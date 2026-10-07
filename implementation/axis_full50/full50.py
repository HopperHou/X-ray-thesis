"""Final Plan B: fixed AXIS, six fresh paired runs; no performance gate."""
from __future__ import annotations
import argparse, hashlib, json, os, socket, subprocess, time, shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.tal import make_anchors
from full_eval import evaluate, summarize_ap_matrix
from v6a2_loss import aligned_geometry, assigned_geometry_loss, axis_center_term, axis_decoded_pixel_boxes
from v6a2_model import set_trainable, install_v6a2_head
from v6a2_train import _load_checkpoint, _parameter_hash, amp_context, make_loader, model_and_loss, move_batch, sha

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'runtime_protocol.json'
FROZEN=ROOT/'protocol.json'
RULE=FROZEN
MANIFEST=ROOT/'implementation_manifest.json'
SEEDS=(42,43,44)
ARMS={'B2':('box_native','original'),'AXIS':('box_geo','axis_center_v1')}
EXPECTED_M3_SHA='c20481309083b51b68111922c33e68250595cbac2f68018369cb26edf69a2d90'
IOU_KEYS=[f'{.5+.05*j:.2f}' for j in range(10)]

def read_json(p): return json.loads(Path(p).read_text())
def atomic_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)

def protocol():
    v=read_json(PROTOCOL);f=read_json(FROZEN)
    if (f['execution']['budget']!='B' or f['execution']['seeds']!=list(SEEDS) or
        f['training']['global_batch']!=180 or f['training']['workers']!=40 or
        f['training']['steps_per_arm']!=4150 or v['lambda_geo']!=0.6070424318313599 or
        v['lambda_q']!=1 or v['box_optimizer']['epochs']!=50 or v['pilot']['epochs']!=50):
        raise RuntimeError('Frozen execution configuration differs')
    return v

def verify_sources(v):
    m=read_json(MANIFEST);obs={}
    for name,want in m['files'].items():
        p=ROOT/name;actual=sha(p)
        if actual!=want:raise RuntimeError(f'Implementation SHA conflict: {p}')
        obs[str(p)]=actual
    old=read_json(ROOT/'pilot_source_manifest.json')
    for name in ('v6a2_train.py','v6a2_targets.py','v6a2_model.py','v6a2_loss.py','v6a2_eval.py'):
        if sha(ROOT/name)!=old['pilot_source_sha256'][name]:raise RuntimeError(f'Frozen method source changed: {name}')
    import ultralytics
    if ultralytics.__version__!='8.4.83':raise RuntimeError('Ultralytics version changed')
    site=Path(ultralytics.__file__).parent
    for rel,want in v['ultralytics_local_source_sha256'].items():
        p=site/rel
        if sha(p)!=want:raise RuntimeError(f'Installed source conflict: {p}')
        obs[str(p)]=want
    paths={k:Path(v[k]) for k in ('train_list','development_validation_list','eligibility_matrix','data_yaml','a0_checkpoint')}
    paths['population_json']=Path(v['pilot']['population_json'])
    for k,p in paths.items():
        want=old['frozen_data_sha256'][k];actual=sha(p)
        if actual!=want:raise RuntimeError(f'Frozen input SHA conflict: {k}')
        obs[str(p)]=actual
    p=Path(v['m3_checkpoint'])
    if sha(p)!=EXPECTED_M3_SHA:raise RuntimeError('M3 E0 SHA conflict')
    obs[str(p)]=EXPECTED_M3_SHA;obs[str(MANIFEST)]=sha(MANIFEST)
    return obs

def p0():
    for p in ('runs','sanity.json','source_hashes.json','evaluation_check.json'):
        if (ROOT/p).exists():raise RuntimeError(f'Execution directory not fresh: {p}')
    v=protocol();obs=verify_sources(v)
    if socket.gethostname()!='hpc6' or os.environ.get('CUDA_VISIBLE_DEVICES')!=v['pilot']['gpu_uuid']:
        raise RuntimeError('Wrong host/GPU visibility')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA/bf16 unavailable')
    provenance=read_json(Path(v['pilot']['source_root'])/'runs/m3_full_q/result.json')
    if provenance['status']!='PASS' or provenance['checkpoint_sha256']!=EXPECTED_M3_SHA:raise RuntimeError('M3 status conflict')
    pop=read_json(v['pilot']['population_json']);groups={}
    for split,key,n,npats in (('train','train_list',14934,8896),('validation','development_validation_list',1660,990)):
        paths=Path(v[key]).read_text().splitlines();rows=pop[split]
        if len(paths)!=n or len(rows)!=n or len(set(paths))!=n or {r['path'] for r in rows}!=set(paths):raise RuntimeError(f'{split} mapping mismatch')
        groups[split]={str(r['group']) for r in rows}
        if len(groups[split])!=npats or any(not r.get('group') for r in rows):raise RuntimeError(f'{split} patient mismatch')
    if groups['train']&groups['validation']:raise RuntimeError('Patient split overlap')
    matrix=yaml.safe_load(Path(v['eligibility_matrix']).read_text())['matrix']
    if any(matrix[r['source']][c] not in ('E','P','U') for s in ('train','validation') for r in pop[s] for c in v['class_names']):raise RuntimeError('Eligibility invalid')
    from ultralytics.data.utils import img2label_paths
    label_manifest={};cache=Path(v['train_list']).parent/'labels/train.cache'
    if not cache.is_file():raise RuntimeError('Canonical train GT cache missing')
    obs[str(cache)]=sha(cache)
    for split,key in (('train','train_list'),('validation','development_validation_list')):
        h=hashlib.sha256();missing=0
        for p in img2label_paths(Path(v[key]).read_text().splitlines()):
            f=Path(p);h.update(str(f).encode());h.update(b'\0')
            if f.is_file():h.update(f.read_bytes())
            else:missing+=1;h.update(b'<absent-negative-label>')
        label_manifest[split]={'aggregate_label_sha256':h.hexdigest(),'absent_labels':missing}
    if shutil.disk_usage(ROOT).free < 40*1024**3:raise RuntimeError('Insufficient disk space for six final runs and predeclared snapshots')
    atomic_json(ROOT/'source_hashes.json',{'status':'PASS','observed_sha256':obs,'GT':label_manifest,'test_data_read':0})
    atomic_json(ROOT/'p0_provenance.json',{'status':'PASS','m3_sha256':EXPECTED_M3_SHA,'patient_groups':{s:len(g) for s,g in groups.items()},'train_images':14934,'development_images':1660,'patient_overlap':0,'GT':label_manifest,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_uuid':v['pilot']['gpu_uuid'],'test_data_read':0})
    stage('P0_PASS')

def frozen_hash(model):
    return _parameter_hash({n:t for n,t in model.state_dict().items() if 'one2one_cv2' not in n and 'v6a2_quality_heads' not in n})

def verify_freeze(model,expected):
    if frozen_hash(model)!=expected:raise RuntimeError('Frozen parameters/BN buffers changed')
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if not names or any('one2one_cv2' not in n and 'v6a2_quality_heads' not in n for n in names):raise RuntimeError('Trainable scope conflict')
    box_bn=set(model.model[-1].one2one_cv2.modules())
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.training!=(m in box_bn):raise RuntimeError('BN policy conflict')

def save_checkpoint(path,model,arm,v,boxhash,qhash,epoch,seed):
    payload={'model_state':{k:t.detach().cpu() for k,t in model.state_dict().items()},'epoch':epoch,'arm':arm,'seed':seed,'protocol_sha256':sha(PROTOCOL),'frozen_protocol_sha256':sha(FROZEN),'source_m3_checkpoint':v['m3_checkpoint'],'source_m3_sha256':EXPECTED_M3_SHA,'initial_box_hash':boxhash,'initial_q_hash':qhash}
    temp=path.with_suffix('.pending');torch.save(payload,temp);temp.replace(path)

# These functions were copied from the verified pilot; only run length, seed,
# two-arm pairing, checkpoint reporting and freeze validation were adapted.
exec(compile((ROOT/'reused_pilot_functions.py').read_text(),str(ROOT/'reused_pilot_functions.py'),'exec'))
_cpu_toys=p1

def p1():
    torch.set_num_threads(4);_cpu_toys();v=protocol()
    model,criterion=model_and_loss(v,torch.device('cpu'),'box_native')
    _load_checkpoint(Path(v['m3_checkpoint']),model,Path(v['pilot']['source_root'])/'protocol.m3_b60w30.frozen.json')
    model.eval();sample=torch.zeros(1,3,128,128)
    with torch.no_grad():
        initial=model(sample)
        set_trainable(model,'box_geo');model.eval();same=model(sample)
    if not torch.equal(initial[0],same[0]):raise RuntimeError('E0 B2/AXIS inference identity failed')
    if not torch.equal(same[1]['one2one']['scores'],same[1]['one2one']['original_logits']):raise RuntimeError('Scorer is not identity')
    model.train();set_trainable(model,'box_geo');h=frozen_hash(model);verify_freeze(model,h)
    atomic_json(ROOT/'freeze_check.json',{'status':'PASS','BN':'only_one2one_cv2_updates','trainable':'one2one_cv2_and_Q','E0_arm_inference_identity':True,'identity_scorer':True,'verified_method_source_unchanged':True,'actual_batch_factor':'native_once_in_criterion_geo_once_in_caller','lambda_geo':v['lambda_geo'],'test_data_read':0})
    stage('P1_PASS')

def resource_sanity():
    if read_json(ROOT/'sanity.json')['status']!='PASS':raise RuntimeError('CPU checks failed')
    v=protocol();torch.set_num_threads(4)
    loader=make_loader(v,Path(v['train_list']),augment=True,batch=180,workers=40,shuffle=True,seed=42,pin_memory=True)
    data=next(iter(loader));device=torch.device('cuda:0');model,criterion=model_and_loss(v,device,'box_geo');criterion.geo_variant='axis_center_v1'
    _load_checkpoint(Path(v['m3_checkpoint']),model,Path(v['pilot']['source_root'])/'protocol.m3_b60w30.frozen.json')
    batch=move_batch(data,device)
    with amp_context(v,device):loss,audit=criterion(model(batch['img']),batch,'box_geo',lambda_geo=v['lambda_geo']*180,epoch=1)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise RuntimeError('Resource sanity gradients invalid')
    atomic_json(ROOT/'resource_sanity.json',{'status':'PASS','batch':180,'workers':40,'optimizer_steps':0,'loss_finite':True,'gradients_finite':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    stage('RESOURCE_PASS')

def pair():
    results={};initial=set()
    for seed in SEEDS:
        states={a:read_json(ROOT/'runs'/f'seed{seed}'/a/'status.json') for a in ARMS}
        streams={a:[json.loads(line) for line in (ROOT/'runs'/f'seed{seed}'/a/'input_hashes.jsonl').read_text().splitlines()] for a in ARMS}
        for a,s in states.items():
            if s['status']!='PASS' or s['total_optimizer_steps']!=4150 or not s['freeze_verified'] or len(streams[a])!=4150 or sha(Path(s['checkpoint']))!=s['checkpoint_sha256']:raise RuntimeError(f'Incomplete/corrupt {seed}/{a}')
            initial.add((s['initial_box_hash'],s['initial_q_hash']))
        for j,(b,a) in enumerate(zip(streams['B2'],streams['AXIS']),1):
            if {k:v for k,v in b.items() if k!='arm'}!={k:v for k,v in a.items() if k!='arm'}:raise RuntimeError(f'Pair mismatch seed{seed} step{j}')
        results[str(seed)]={'status':'PASS','paired_steps':4150}
    if len(initial)!=1:raise RuntimeError('Six initial checkpoints differ')
    atomic_json(ROOT/'pairing_check.json',{'status':'PASS','seeds':results,'total_optimizer_steps':24900,'same_M3_E0':True,'freeze_verified_all_epochs':True,'test_data_read':0});stage('P2_PASS')

def eval_all():
    if read_json(ROOT/'pairing_check.json')['status']!='PASS':raise RuntimeError('Pairing not passed')
    v=protocol();verify_sources(v);s=v['locked_scorer'];params=tuple(s[k] for k in ('delta_plus','delta_minus','b','k'))
    jobs=[('A0-R',Path(v['a0_checkpoint']))]+[(f'{a}_seed{seed}',ROOT/'runs'/f'seed{seed}'/a/'epoch_50.pt') for seed in SEEDS for a in ARMS]
    checked={}
    for name,checkpoint in jobs:
        stage('P3_EVALUATION',model=name)
        out=ROOT/'evaluations'/name
        r=evaluate(v,checkpoint,out,params,device='cuda:0',batch=8,workers=8,expected_images=1660)
        matrix=read_json(out/'per_iou_ap.json');ap=np.asarray(matrix['ap_by_class'])
        if r['status']!='PASS' or r['images']!=1660 or r['TEST_DATA_READ']!=0 or ap.shape!=(6,10) or not np.isfinite(ap).all():raise RuntimeError(f'Evaluation invalid: {name}')
        atomic_json(out/'eval_result.json',{**r,'name':name,'epoch':None if name=='A0-R' else 50,'ap_by_class_iou':ap.tolist(),'metric_unit':'fraction','patient_groups':990})
        checked[name]={'status':'PASS','images':1660,'checkpoint_sha256':sha(checkpoint),'complete_AP_matrix':True}
    atomic_json(ROOT/'evaluation_check.json',{'status':'PASS','models':checked,'test_data_read':0});stage('P3_PASS_ALL_GPU_TASKS_FINISHED')

def stage(name,**kw):
    atomic_json(ROOT/'pipeline_status.json',{'stage':name,'timestamp':time.time(),**kw})
    (ROOT/'EXPERIMENT_TRACKER.md').write_text('# Plan B execution tracker\n\nCurrent stage: '+name+'\n\n'+json.dumps(kw,ensure_ascii=False)+'\n\nDetailed run status: runs/seed{42,43,44}/{B2,AXIS}/status.json\n')

def finish(status,error=None):
    if status not in ('VALID_COMPLETE','INVALID','INCOMPLETE'):raise ValueError(status)
    atomic_json(ROOT/'completion_status.json',{'status':status,'error':error,'test_data_read':0,'training_steps_expected':24900,'checkpoint_rule':'E50_only'})
    if status!='VALID_COMPLETE':
        (ROOT/'REPORT.md').write_text(f'# Plan B full-training report\n\nStatus: {status}\n\nExecution error: {error}\n\nNo scientific STOP/GO is inferred from an execution failure.\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['p0','p1','resource','train','pair','eval','invalid','incomplete']);p.add_argument('--arm',choices=list(ARMS));p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--error',default='pipeline interrupted');args=p.parse_args()
    try:
        if args.stage=='train':
            if args.arm is None or args.seed is None:raise ValueError('arm and seed required')
            stage('P2_TRAINING',seed=args.seed,arm=args.arm);torch.set_num_threads(4);train(args.arm,args.seed)
        elif args.stage=='resource':resource_sanity()
        elif args.stage=='invalid':finish('INVALID',args.error)
        elif args.stage=='incomplete':finish('INCOMPLETE',args.error)
        elif args.stage=='eval':eval_all()
        else:globals()[args.stage]()
    except BaseException as e:
        finish('INCOMPLETE' if isinstance(e,KeyboardInterrupt) else 'INVALID',f'{args.stage}: {type(e).__name__}: {e}')
        raise

if __name__=='__main__':main()
