"""Bounded CPU/GPU gates and faithful historical-loop reuse in an isolated run."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,ast,copy,hashlib,json,math,random,signal,sys,time
from pathlib import Path
from common import *

C=read(ROOT/'pipeline_config.yaml')
for folder in ('runtime/yolo','runtime/mpl'):(ROOT/folder).mkdir(parents=True,exist_ok=True)
BASE=Path(C['source_root']);sys.path.insert(0,str(BASE))
import numpy as np
import torch
import full50 as historical
from v6a2_train import model_and_loss,make_loader,move_batch,amp_context,_load_checkpoint,_parameter_hash
from v6a2_model import set_trainable
import adapter
from progress import TrainingReporter
import v6a2_train as original_train
_dataset_check=original_train.check_det_dataset
def no_download(data,*a,**kw):
    kw['autodownload']=False
    return _dataset_check(data,*a,**kw)
original_train.check_det_dataset=no_download

def config():
    v=historical.protocol();v['seed']=43
    assert v['lambda_geo']==0.6070424318313599
    return v

def protection():
    roots=[Path(x).resolve() for x in C['protected_roots']]
    def audit(event,args):
        paths=[]
        if event=='open':
            p,mode,flags=args
            if isinstance(p,(str,bytes,os.PathLike)) and ((isinstance(mode,str) and any(k in mode for k in 'wax+')) or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):paths=[p]
        elif event in ('os.remove','os.rename','os.rmdir','os.mkdir'):
            paths=[x for x in args[:2] if isinstance(x,(str,bytes,os.PathLike))]
        for p in paths:
            q=Path(os.fsdecode(p)).absolute()
            if any(q==r or r in q.parents for r in roots):raise PermissionError('Historical write blocked: '+str(q))
    sys.addaudithook(audit)

def asset_gate():
    for name,want in read(ROOT/'DEPLOYMENT_MANIFEST.json')['files'].items():assert sha(ROOT/name)==want,('Deployment changed',name)
    torch.set_num_threads(4);v=config();obs=historical.verify_sources(v)
    for p,want in C['extra_identities'].items():
        assert sha(p)==want,('Identity mismatch',p)
        obs[p]=want
    reference=read(BASE/'source_hashes.json')
    cache=Path(v['train_list']).parent/'labels/train.cache'
    assert sha(cache)==reference['observed_sha256'][str(cache)]
    from ultralytics.data.utils import img2label_paths
    for split,key in [('train','train_list'),('validation','development_validation_list')]:
        h=hashlib.sha256()
        for p in img2label_paths(Path(v[key]).read_text().splitlines()):
            f=Path(p);h.update(str(f).encode());h.update(b'\0');h.update(f.read_bytes() if f.exists() else b'<absent-negative-label>')
        assert h.hexdigest()==reference['GT'][split]['aggregate_label_sha256'],('GT changed',split)
    for arm in ('B2','AXIS'):
        s=read(BASE/f'runs/seed43/{arm}/status.json')
        assert s['status']=='PASS' and s['total_optimizer_steps']==4150
        assert sha(s['checkpoint'])==s['checkpoint_sha256']
    assert torch.__version__=='2.11.0+cu126'
    atomic(ROOT/'source_hashes.json',{'status':'PASS','observed_sha256':obs,'CUDA_initialized':torch.cuda.is_initialized(),'updated':now()})
    atomic(ROOT/'sanity.json',{'status':'PASS','basis':'CPU preflight plus unchanged source/data; production hooks checked separately'})
    assert not torch.cuda.is_initialized()

def model(device):
    v=config();m,k=model_and_loss(v,device,'box_geo')
    _load_checkpoint(Path(v['m3_checkpoint']),m,Path(v['pilot']['source_root'])/'protocol.m3_b60w30.frozen.json')
    k.geo_variant='axis_center_v1'
    s=read(BASE/'runs/seed43/B2/status.json')
    assert _parameter_hash(m.model[-1].one2one_cv2.state_dict())==s['initial_box_hash']
    assert _parameter_hash(m.model[-1].v6a2_quality_heads.state_dict())==s['initial_q_hash']
    return m,k

def cpu_integration():
    asset_gate();adapted_train_source();m,k=model(torch.device('cpu'));v=config()
    paths=Path(v['train_list']).read_text().splitlines()[:2]
    b={'img':torch.zeros(2,3,128,128),'im_file':paths,'batch_idx':torch.tensor([0.,1.]),'cls':torch.tensor([[0.],[0.]]),'bboxes':torch.tensor([[.5,.5,.3,.2],[.5,.5,.2,.3]])}
    initial=copy.deepcopy(m.state_dict());results={};grads={}
    for name in ('ORIGINAL','AXIS'):
        m.load_state_dict(initial);m.train();set_trainable(m,'box_geo');m.zero_grad(set_to_none=True)
        torch.manual_seed(43);np.random.seed(43);random.seed(43)
        adapter.uninstall() if name=='ORIGINAL' else adapter.install('AXIS')
        loss,audit=k(m(b['img']),b,'box_geo',lambda_geo=v['lambda_geo']*2,epoch=1)
        loss.backward();results[name]=float(loss.detach())
        grads[name]=torch.cat([p.grad.detach().flatten() for p in m.model[-1].one2one_cv2.parameters()])
    assert results['ORIGINAL']==results['AXIS']
    assert torch.equal(grads['ORIGINAL'],grads['AXIS'])
    assert not torch.cuda.is_initialized()
    atomic(ROOT/'CPU_INTEGRATION.json',{'status':'PASS','synthetic_not_research_data':True,'loss_exact':True,'regression_gradient_exact':True,'CUDA_initialized':False,'optimizer_steps':0,'source_hashes_sha256':sha(ROOT/'source_hashes.json')})

def setup_gpu():
    assert_device(C);torch.set_num_threads(4)
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0)
    assert torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark=False
    free,total=torch.cuda.mem_get_info(0)
    cap=min(C['allocator_cap_bytes'],free-6*1024**3)
    assert cap>46*1024**3,('Insufficient shared memory admission',free,cap)
    torch.cuda.set_per_process_memory_fraction(cap/total,0)
    torch.cuda.reset_peak_memory_stats(0)
    return torch.device('cuda:0')

def witness(receipt):
    assert_device(C);torch.cuda.set_device(0);torch.manual_seed(43)
    x=torch.randn(8,8,device='cuda');y=x@x
    with torch.autocast('cuda',dtype=torch.bfloat16):z=x@x
    assert torch.isfinite(y).all() and torch.isfinite(z).all() and z.dtype==torch.bfloat16
    atomic(ROOT/receipt,{'status':'PASS','torch':torch.__version__,'CUDA':torch.version.cuda,'device':torch.cuda.get_device_name(0),'gpu_uuid':C['gpu_uuid'],'seed':43,'shape':list(y.shape),'BF16':True,'updated':now()})
    print('KERNEL_WITNESS_PASS',flush=True)

def get_input_digest(data,k):
    pop=read(config()['pilot']['population_json'])
    patients={str(r['path']):str(r['group']) for r in pop['train']}
    # Verbatim historical digest, without importing any historical main.
    tree=ast.parse((BASE/'reused_pilot_functions.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='input_digest')
    ns={'torch':torch,'hashlib':hashlib,'json':json}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(BASE/'reused_pilot_functions.py'),'exec'),ns)
    return ns['input_digest'](data,patients,k.sources,k.matrix,k.classes)

def check_input(data,k,index):
    ref=json.loads((BASE/'runs/seed43/B2/input_hashes.jsonl').read_text().splitlines()[index])
    observed=get_input_digest(data,k)
    for key,value in zip(('input_sha256','ordered_paths_sha256','ordered_patient_groups_sha256','ordered_eligibility_sha256'),observed):assert ref[key]==value,('Input replay mismatch',index,key)
    assert len(data['im_file'])==ref['actual_batch_size']==180
    return dict(index=index,**{key:ref[key] for key in ('input_sha256','ordered_paths_sha256','actual_batch_size')})

def buffer_state(m):return {n:b.detach().clone() for n,b in m.named_buffers()}
def restore_buffers(m,s):
    with torch.no_grad():
        for n,b in m.named_buffers():b.copy_(s[n])
def vecnorm(gs):return math.sqrt(sum(float(g.detach().cpu().double().square().sum()) for g in gs))

def pilot_calibrate():
    device=setup_gpu();v=config()
    loader=make_loader(v,Path(v['train_list']),augment=True,batch=180,workers=40,shuffle=True,seed=43,pin_memory=True)
    m,k=model(device);initial_buffers=buffer_state(m)
    theta=list(m.model[-1].one2one_cv2.parameters());assert len(theta)==24
    iterator=iter(loader);rows=[];rates=[]
    for index in range(4):
        data=next(iterator);replay=check_input(data,k,index);batch=move_batch(data,device)
        restore_buffers(m,initial_buffers);m.train();set_trainable(m,'box_geo');m.zero_grad(set_to_none=True)
        rng=(torch.get_rng_state(),torch.cuda.get_rng_state(),np.random.get_state(),random.getstate())
        if index<2:
            adapter.uninstall();t0=time.monotonic()
            with amp_context(v,device):old_loss,_=k(m(batch['img']),batch,'box_geo',lambda_geo=v['lambda_geo']*180,epoch=1)
            old_loss.backward();old_value=float(old_loss.detach());old_g=[p.grad.detach().cpu().clone() for p in theta]
            torch.cuda.synchronize();rates.append(time.monotonic()-t0)
            del old_loss;m.zero_grad(set_to_none=True);restore_buffers(m,initial_buffers)
            torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])
        adapter.install('AXIS',capture=True)
        with amp_context(v,device):loss,audit=k(m(batch['img']),batch,'box_geo',lambda_geo=v['lambda_geo']*180,epoch=1)
        axis=adapter.TENSORS.get('axis');iso=adapter.TENSORS.get('isotropic')
        assert axis is not None and len(axis)>0
        ga=torch.autograd.grad(180*v['lambda_geo']*axis.mean(),theta,retain_graph=True)
        gi=torch.autograd.grad(180*v['lambda_geo']*iso.mean(),theta,retain_graph=True)
        na,ni=vecnorm(ga),vecnorm(gi);assert math.isfinite(na) and math.isfinite(ni) and min(na,ni)>0
        del ga,gi
        loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        relative=None
        if index<2:
            diff=math.sqrt(sum(float((p.grad.detach().cpu().double()-g.double()).square().sum()) for p,g in zip(theta,old_g)))
            denom=math.sqrt(sum(float(g.double().square().sum()) for g in old_g));relative=diff/max(denom,1e-300)
            assert abs(float(loss.detach())-old_value)<=1e-6 and relative<=1e-5,('Production integration drift',relative)
            del old_g
        rows.append({**replay,'foreground':len(axis),'axis_norm':na,'isotropic_norm':ni,'ratio':na/ni,'production_gradient_relative_L2':relative})
        del loss,batch,data,axis,iso;adapter.TENSORS.clear();m.zero_grad(set_to_none=True)
    torch.cuda.synchronize();scale=float(np.median([r['ratio'] for r in rows]));assert math.isfinite(scale) and scale>0
    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();free=torch.cuda.mem_get_info()[0]
    assert free>=4*1024**3
    atomic(ROOT/'CALIBRATION.json',{'status':'PASS','seed':43,'scale':scale,'panel':'first four historical seed43 batch180 inputs','rows':rows,'optimizer_steps':0,'BN':'M3 buffers restored before corresponding forwards','precision':'historical BF16 AMP; FP32 centre re-decoding; CPU FP64 norm sums','theta_tensors':24,'quantile':'numpy median linear','validation_test_used':False})
    seconds=max(rates);estimated=2*4150*seconds*1.5+C['evaluation_budget_seconds']
    atomic(ROOT/'GPU_ADMISSION.json',{'status':'PASS','batch':180,'workers':40,'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'free_bytes_at_end':free,'timed_native_steps':rates,'forecast_seconds_including_50percent_margin':estimated,'expected_finish_epoch':time.time()+estimated,'calibration_sha256':sha(ROOT/'CALIBRATION.json'),'optimizer_steps':0,'gpu_uuid':C['gpu_uuid'],'environment':{'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}})

def reuse_pilot():
    setup_gpu()
    assert C['enforce_deadline'] is False
    assert sha(ROOT/'CALIBRATION.json')==C['resume_calibration_sha256']
    assert sha(ROOT/'RESOURCE_pilot.json')==C['resume_resource_sha256']
    cal=read(ROOT/'CALIBRATION.json');resource=read(ROOT/'RESOURCE_pilot.json')
    assert cal['status']=='PASS' and cal['seed']==43 and cal['optimizer_steps']==0
    assert len(cal['rows'])==4 and cal['validation_test_used'] is False
    refs=[json.loads(x) for x in (BASE/'runs/seed43/B2/input_hashes.jsonl').read_text().splitlines()[:4]]
    for i,(row,ref) in enumerate(zip(cal['rows'],refs)):
        assert row['index']==i and row['actual_batch_size']==180
        for key in ('input_sha256','ordered_paths_sha256'):assert row[key]==ref[key]
        assert min(row['axis_norm'],row['isotropic_norm'],row['ratio'])>0 and all(math.isfinite(row[k]) for k in ('axis_norm','isotropic_norm','ratio'))
        if i<2:assert row['production_gradient_relative_L2']==0
    assert cal['scale']==float(np.median([r['ratio'] for r in cal['rows']]))
    assert resource['gpu_uuid']==C['gpu_uuid'] and resource['free_bytes']>=4*1024**3
    assert 'Cannot finish within deadline' in (ROOT/'attempts/attempt01/logs/PILOT_AND_TRAIN_ONLY_CALIBRATION.log').read_text()
    assert read(ROOT/'CPU_INTEGRATION.json')['status']=='PASS'
    atomic(ROOT/'GPU_ADMISSION.json',{'status':'PASS','updated':now(),'basis':'Reuse completed four-batch real-data replay, production-gradient equality and calibration; prior sole failure was now-waived time forecast. No new model forward in this stage.','batch':180,'workers':40,'gpu_uuid':C['gpu_uuid'],'pilot_resource_sha256':sha(ROOT/'RESOURCE_pilot.json'),'calibration_sha256':sha(ROOT/'CALIBRATION.json'),'prior_pilot_peak_allocated_bytes':resource['peak_allocated_bytes'],'prior_pilot_peak_reserved_bytes':resource['peak_reserved_bytes'],'current_free_bytes':torch.cuda.mem_get_info()[0],'time_stop_enabled':False,'optimizer_steps':0})


def save_checkpoint(path,m,arm,v,boxhash,qhash,epoch,seed):
    payload={'model_state':{n:t.detach().cpu() for n,t in m.state_dict().items()},'epoch':epoch,'arm':arm,'seed':seed,'protocol_sha256':sha(BASE/'runtime_protocol.json'),'frozen_protocol_sha256':sha(BASE/'protocol.json'),'source_m3_checkpoint':v['m3_checkpoint'],'source_m3_sha256':historical.EXPECTED_M3_SHA,'initial_box_hash':boxhash,'initial_q_hash':qhash,'extension_config_sha256':sha(ROOT/'pipeline_config.yaml'),'calibration_sha256':sha(ROOT/'CALIBRATION.json'),'training_gpu_uuid':C['gpu_uuid'],'training_host':'hpc5'}
    tmp=path.with_suffix('.pending');torch.save(payload,tmp);tmp.replace(path)

def adapted_train_source():
    tree=ast.parse((BASE/'reused_pilot_functions.py').read_text())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='train')
    src=ast.unparse(fn)
    old="if socket.gethostname() != 'hpc6' or os.environ.get('CUDA_VISIBLE_DEVICES') != value['pilot']['gpu_uuid']:"
    assert src.count(old)==1
    src=src.replace(old,"if socket.gethostname() != 'hpc5' or os.environ.get('CUDA_VISIBLE_DEVICES') != E2_GPU_UUID:")
    tree=ast.parse(src);reference=ast.dump(tree);count=0
    class AddReporter(ast.NodeTransformer):
        def visit_For(self,node):
            nonlocal count
            self.generic_visit(node)
            if isinstance(node.target,ast.Tuple) and [getattr(x,'id',None) for x in node.target.elts]==['step','data']:
                node.body.append(ast.parse('progress_tick(arm,total_steps,epoch,step,started)').body[0]);count+=1
            return node
    tree=ast.fix_missing_locations(AddReporter().visit(tree));assert count==1
    class StripReporter(ast.NodeTransformer):
        def visit_Expr(self,node):
            return None if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='progress_tick' else node
    assert ast.dump(StripReporter().visit(copy.deepcopy(tree)))==reference,'Numerical loop changed beyond approved guard and observer'
    src=ast.unparse(tree)
    atomic(ROOT/'APPROVED_LOOP_ADAPTATION.json',{'base_sha256':sha(BASE/'reused_pilot_functions.py'),'changed_guard':'hpc6/original UUID replaced by explicit hpc5/approved UUID; no guard removed','observer':'one telemetry call after each complete optimizer step; stripped AST equals guard-adapted original','numerical_AST_preserved':True,'adapted_train_function_sha256':hashlib.sha256(src.encode()).hexdigest(),'batch':180,'workers':40,'seed':43})
    return src


def train(arm):
    setup_gpu();assert read(ROOT/'GPU_ADMISSION.json')['status']=='PASS'
    cal=read(ROOT/'CALIBRATION.json');assert sha(ROOT/'CALIBRATION.json')==read(ROOT/'GPU_ADMISSION.json')['calibration_sha256']
    adapter.install(arm,scale=cal['scale']);src=adapted_train_source()
    def make_model(v,device,mode):
        m,k=model_and_loss(v,device,mode);return m,adapter.AuditProxy(k)
    ns=dict(vars(historical));ns.update(ROOT=ROOT,PROTOCOL=BASE/'runtime_protocol.json',FROZEN=BASE/'protocol.json',RULE=BASE/'protocol.json',MANIFEST=BASE/'implementation_manifest.json',E2_GPU_UUID=C['gpu_uuid'],protocol=config,model_and_loss=make_model,save_checkpoint=save_checkpoint,progress_tick=TrainingReporter(),ARMS={arm:('box_geo','axis_center_v1')})
    exec(compile(src,'approved_isolated_historical_train_loop','exec'),ns)
    ns['train'](arm,43)

def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['cpu','witness','pilot','reuse-pilot','train']);a.add_argument('--arm');a.add_argument('--receipt',default='KERNEL_WITNESS.json');args=a.parse_args()
    def interrupted(*unused):raise KeyboardInterrupt('Pipeline worker termination requested')
    signal.signal(signal.SIGTERM,interrupted)
    protection()
    try:
        if args.stage=='cpu':cpu_integration()
        elif args.stage=='witness':witness(args.receipt)
        elif args.stage=='pilot':pilot_calibrate()
        elif args.stage=='reuse-pilot':reuse_pilot()
        else:train(args.arm)
    finally:
        if torch.cuda.is_initialized():
            atomic(ROOT/('RESOURCE_'+args.stage+('_'+args.arm if args.arm else '')+'.json'),{'updated':now(),'gpu_uuid':C['gpu_uuid'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'optimizer_steps':'see arm status; calibration and witness zero','free_bytes':torch.cuda.mem_get_info()[0]})

if __name__=='__main__':main()
