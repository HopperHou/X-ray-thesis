"""Isolated FP32 native E50 evaluation/export. Never invokes training entrypoints."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from pathlib import Path
import sys,json,time,types,copy,hashlib,itertools,argparse
from common import read,atomic,sha,now,ROOT

def main():
    import ctypes,signal
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()==1:raise RuntimeError('Runner disappeared before worker startup')
    a=argparse.ArgumentParser();a.add_argument('--task',required=True);a.add_argument('--output',required=True);a.add_argument('--images',type=int,default=1844);a.add_argument('--gpu-uuid',required=True);a.add_argument('--worker-config',required=True);a=a.parse_args()
    cfg=read(a.worker_config);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(out/'yolo_config');os.environ['MPLCONFIGDIR']=str(out/'mpl_config')
    import numpy as np
    import torch,cv2
    torch.set_num_threads(cfg['cpu_threads_per_task']);cv2.setNumThreads(0)
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu_uuid
    assert torch.cuda.device_count()==1,'Only explicitly assigned GPU may be visible'
    torch.cuda.set_device(0)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,cfg['allocator_cap_bytes']/total),0)
    torch.cuda.reset_peak_memory_stats(0)
    free_before,total_before=torch.cuda.mem_get_info(0)
    sys.path.insert(0,cfg['source_root'])
    from ultralytics import YOLO
    import ultralytics.data.build as db
    from ultralytics.data.dataset import YOLODataset,DATASET_CACHE_VERSION
    from ultralytics.data.utils import img2label_paths,load_dataset_cache_file,get_hash
    from ultralytics.models.yolo.detect.val import DetectionValidator
    import ultralytics.engine.validator as base_validator
    from ultralytics.utils.metrics import box_iou
    from v6a2_model import install_v6a2_head
    from v6a2_eval import two_stage_topk
    from full_eval import summarize_ap_matrix
    from ultralytics.utils import ops

    # Refuse native cache repair or download; preserve native cached labels and full rect sorting.
    class ReadOnlyDataset(YOLODataset):
        def get_labels(self):
            self.label_files=img2label_paths(self.im_files)
            cp=Path(self.label_files[0]).parent.with_suffix('.cache')
            cache=load_dataset_cache_file(cp)
            assert cache['version']==DATASET_CACHE_VERSION,'Historical cache version mismatch; no regeneration'
            assert cache['hash']==get_hash(self.label_files+self.im_files),'Historical cache invalid; no regeneration'
            assert len(cache['labels'])==cfg['expected_images']
            labels=copy.deepcopy(cache['labels'])
            assert all(len(x['cls'])==len(x['bboxes']) and not len(x['segments']) for x in labels)
            self.im_files=[x['im_file'] for x in labels]
            return labels
        def cache_labels(self,*args,**kwargs):raise RuntimeError('Forbidden historical cache write')
        def cache_images(self,*args,**kwargs):raise RuntimeError('Forbidden image cache generation')
        def cache_images_to_disk(self,*args,**kwargs):raise RuntimeError('Forbidden image cache generation')
    db.YOLODataset=ReadOnlyDataset
    original_check=base_validator.check_det_dataset
    def no_download(data,*args,**kwargs):
        kwargs['autodownload']=False
        return original_check(data,*args,**kwargs)
    base_validator.check_det_dataset=no_download
    # Dataset loader may delete invalid .npy caches. Block every source-tree write/delete.
    protected=[str(Path(cfg['data_root']).resolve()),str(Path(cfg['source_root']).resolve()),str(Path(cfg['test_root']).resolve()),str(Path(cfg['simplified_root']).resolve())]
    def guard(event,args):
        if event in ('open','os.remove','os.rename','os.rmdir','os.mkdir'):
            paths=[]
            if event=='open':
                path,mode,flags=args
                if isinstance(path,(str,bytes,os.PathLike)) and ((isinstance(mode,str) and any(x in mode for x in 'wax+')) or isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):paths=[path]
            else:paths=[p for p in args[:2] if isinstance(p,(str,bytes,os.PathLike))]
            for p in paths:
                s=os.path.abspath(os.fsdecode(p))
                if any(s==r or s.startswith(r+os.sep) for r in protected):raise PermissionError('Protected historical path write blocked: '+s)
    sys.addaudithook(guard)
    prot=read(Path(cfg['source_root'])/'runtime_protocol.json');task=cfg['models'][a.task]
    ck=Path(task['path']);assert sha(ck)==task['sha256']
    assert sha(prot['a0_checkpoint'])==cfg['a0_sha256']
    started=now();t0=time.monotonic()
    payload=torch.load(ck,map_location='cpu',weights_only=False)
    assert 'model_state' in payload
    detector=YOLO(prot['a0_checkpoint'])
    s=prot['locked_scorer'];install_v6a2_head(detector.model,scorer=tuple(s[k] for k in ('delta_plus','delta_minus','b','k')))
    incompatible=detector.model.load_state_dict(payload['model_state'],strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    del payload
    detector.model.float();head=detector.model.model[-1]
    assert head.end2end
    head.fuse=types.MethodType(lambda self:None,head)
    meta=read(cfg['image_metadata']);records=[];traces={};actual_iouv=[]
    tracefile=(out/'image_trace.jsonl').open('w',encoding='utf8')

    def pairs(gt,pred,thresholds):
        iou=box_iou(gt['bboxes'],pred['bboxes'])
        correct=gt['cls'][:,None]==pred['cls'][None,:]
        iou=(iou*correct).cpu().numpy();tp=np.zeros((len(pred['cls']),len(thresholds)),bool);ps=[]
        for j,th in enumerate(thresholds):
            m=np.array(np.nonzero(iou>=th)).T
            if len(m)>1:
                m=m[iou[m[:,0],m[:,1]].argsort()[::-1]]
                m=m[np.unique(m[:,1],return_index=True)[1]]
                m=m[np.unique(m[:,0],return_index=True)[1]]
            if len(m):tp[m[:,1],j]=True
            ps.append(m)
        return tp,ps

    class TakeLoader:
        def __init__(self,loader,n):
            self.loader=loader;self.n=n;self.dataset=loader.dataset
            # Pilot selects complete native batches; excludes the final partial batch.
            self.indices=set(np.linspace(0,len(loader)-2,n,dtype=int).tolist())
        def __len__(self):return min(len(self.loader),self.n)
        def __iter__(self):
            for i,batch in enumerate(self.loader):
                if i in self.indices:yield batch
        def __getattr__(self,k):return getattr(self.loader,k)

    class CaptureValidator(DetectionValidator):
        def get_dataloader(self,*args,**kwargs):
            loader=super().get_dataloader(*args,**kwargs)
            assert len(loader.dataset)==cfg['expected_images'] and not loader.dataset.augment
            return TakeLoader(loader,a.images//cfg['inference']['batch']) if a.images<cfg['expected_images'] else loader
        def preprocess(self,batch):
            result=super().preprocess(batch)
            assert result['img'].dtype==torch.float32
            self.paths=list(map(str,result['im_file']))
            self.image_hashes=[hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest() for x in result['img']]
            return result
        def postprocess(self,preds):
            raw=preds[1]['one2one'];decoded=head._inference(raw).permute(0,2,1)
            assert decoded.dtype==torch.float32 and torch.isfinite(decoded).all()
            for i,path in enumerate(self.paths):
                x=two_stage_topk(decoded[i,:,:4],decoded[i,:,4:],300,.001)
                traces[path]=x
            output=super().postprocess(preds)
            for path,pred in zip(self.paths,output):
                x=traces[path]
                assert torch.equal(pred['bboxes'],x.final_boxes),'Native Top300 box mismatch'
                assert torch.equal(pred['conf'],x.final_scores),'Native Top300 score mismatch'
                assert torch.equal(pred['cls'].long(),x.final_classes.long()),'Native Top300 class mismatch'
            return output
        def update_metrics(self,preds,batch):
            if not actual_iouv:actual_iouv.extend(self.iouv.cpu().tolist())
            super().update_metrics(preds,batch)
            for bi,pred in enumerate(preds):
                gt=self._prepare_batch(bi,batch);pred=self._prepare_pred(pred);path=str(gt['im_file'])
                assert path in meta
                official=self._process_batch(pred,gt)['tp']
                official=official.cpu().numpy() if isinstance(official,torch.Tensor) else np.asarray(official)
                tp,ps=pairs(gt,pred,actual_iouv)
                assert np.array_equal(official,tp),'Recovered GT matches differ from official matcher'
                g=gt['bboxes'].detach().cpu().numpy();p=pred['bboxes'].detach().cpu().numpy()
                assert np.isfinite(g).all() and np.isfinite(p).all()
                assert ((g[:,2:]-g[:,:2])>0).all(),'Invalid GT box'
                assert ((p[:,2:]-p[:,:2])>=0).all(),'Reversed prediction box'
                matches=np.full((len(g),3),-1,dtype=np.int64)
                for j,k in enumerate((0,5,8)):
                    for gi,pi in ps[k]:matches[gi,j]=pi
                # Official matching uses canvas coordinates. Native inverse is exported separately.
                po=ops.scale_boxes(gt['imgsz'],pred['bboxes'].clone(),gt['ori_shape'],ratio_pad=gt['ratio_pad']).detach().cpu().numpy()
                go=ops.scale_boxes(gt['imgsz'],gt['bboxes'].clone(),gt['ori_shape'],ratio_pad=gt['ratio_pad']).detach().cpu().numpy()
                rec={'path':path,'p':p,'po':po,'g':g,'go':go,'conf':pred['conf'].cpu().numpy(),'pc':pred['cls'].cpu().numpy(),
                     'gc':gt['cls'].cpu().numpy(),'tp':tp,'matches':matches}
                records.append(rec);x=traces.pop(path)
                tr={'path':path,**meta[path],'input_sha256':self.image_hashes[bi],'input_shape':list(gt['imgsz']),
                    'original_shape':list(gt['ori_shape']),'ratio_pad':gt['ratio_pad'],'predictions':len(p),'GT':len(g),
                    'first_slots_sha256':hashlib.sha256(x.first_slots.cpu().numpy().tobytes()).hexdigest(),
                    'coordinate_frame':'official canvas; native clipped original coordinates also exported'}
                tracefile.write(json.dumps(tr,default=lambda v:v.tolist() if hasattr(v,'tolist') else float(v))+'\n');tracefile.flush()
                atomic(out/'progress.json',{'images':len(records),'expected':a.images,'updated':now()})

    results=detector.val(data=prot['data_yaml'],split=cfg['inference']['split'],device=torch.device('cuda:0'),imgsz=1024,batch=cfg['inference']['batch'],
       workers=cfg['workers_per_task'],conf=.001,max_det=300,agnostic_nms=False,plots=False,verbose=False,
       project=str(out),name='native_validation',exist_ok=False,validator=CaptureValidator,cache=False,
       save_json=False,save_txt=False,quantize=32)
    tracefile.close();torch.cuda.synchronize()
    assert len(records)==a.images and len(set(r['path'] for r in records))==a.images
    assert next(detector.model.parameters()).dtype==torch.float32
    arrays={'images':np.array([r['path'] for r in records]),'pred_offsets':np.r_[0,np.cumsum([len(r['p']) for r in records])],
            'gt_offsets':np.r_[0,np.cumsum([len(r['g']) for r in records])]}
    for key,field in [('pred_boxes_input','p'),('pred_boxes_original','po'),('gt_boxes_input','g'),('gt_boxes_original','go'),
                      ('conf','conf'),('pred_cls','pc'),('target_cls','gc'),('tp','tp'),('gt_match_pred_idx','matches')]:
        arrays[key]=np.concatenate([r[field] for r in records],axis=0)
    np.savez_compressed(out/'predictions.npz',**arrays)
    matrix=np.zeros((6,10));ids=np.asarray(results.box.ap_class_index,dtype=int);matrix[ids]=np.asarray(results.box.all_ap)
    atomic(out/'validator_AP.json',{'ap_by_class':matrix.tolist(),'actual_iouv':actual_iouv,'class_ids':ids.tolist()})
    elapsed=time.monotonic()-t0
    atomic(out/'task_completion.json',{'status':'PASS','terminal':True,'started':started,'ended':now(),'pid':os.getpid(),
       'gpu_uuid':a.gpu_uuid,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'images':len(records),'GT':len(arrays['target_cls']),
       'precision':'FP32','batch':cfg['inference']['batch'],'workers':cfg['workers_per_task'],'seconds':elapsed,'images_per_second_end_to_end':len(records)/elapsed,
       'free_bytes_before_model':free_before,'free_bytes_at_completion':torch.cuda.mem_get_info(0)[0],
       'pilot_selection':'Full1844 rect layout retained; uniform fixed native batch indices spanning aspect range; CPU loader traverses full dataset, GPU forwards selected images only' if a.images<cfg['expected_images'] else 'all native batches',
       'numerical_environment':{'torch':torch.__version__,'CUDA':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
           'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
           'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
           'autocast':'FP32 no BF16/FP16 evaluation; validator quantize32','device_name':torch.cuda.get_device_name(0)},
       'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
       'predictions_sha256':sha(out/'predictions.npz'),'trace_sha256':sha(out/'image_trace.jsonl'),
       'protocol':'original native O2O, two Top300, strict score>.001, no NMS; native official matching',
       'historical_protocol_departures':cfg['historical_protocol_departures'],
       'GT_matching_coordinate_frame':'letterboxed input canvas','strict_load':True})

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(1)
