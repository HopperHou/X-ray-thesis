"""Exact official AP patient-resampling, conditional on the seven fixed models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json, csv, multiprocessing as mp
import numpy as np
from ultralytics.utils.metrics import ap_per_class
from full50 import ROOT, SEEDS, read_json, atomic_json, protocol, stage, finish

NAMES=['A0-R']+[f'{a}_seed{s}' for s in SEEDS for a in ('B2','AXIS')]
RECORDS=[]
PATIENT_IMAGES=[]

def ap_matrix(z,pred_indices=None,gt_indices=None):
    p=np.arange(len(z['conf'])) if pred_indices is None else pred_indices
    g=np.arange(len(z['target_cls'])) if gt_indices is None else gt_indices
    output=ap_per_class(z['tp'][p],z['conf'][p],z['pred_cls'][p],z['target_cls'][g],plot=False)
    ap,classes=output[5],output[6]
    matrix=np.zeros((6,10),dtype=np.float64)
    matrix[classes]=ap
    if not np.isfinite(matrix).all():raise RuntimeError('Nonfinite bootstrap AP')
    return matrix

def indices(z,images,key):
    offsets=z[key];return np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in images])

def draw_job(item):
    j,draw=item
    images=np.concatenate([PATIENT_IMAGES[int(k)] for k in draw])
    # All models are reordered to the same image universe before this stage.
    matrices=[]
    for z in RECORDS:
        matrices.append(ap_matrix(z,indices(z,images,'pred_offsets'),indices(z,images,'gt_offsets')))
    return j,np.asarray(matrices)

def reorder(z,paths):
    old={str(p):i for i,p in enumerate(z['images'])}
    order=[old[p] for p in paths]
    pi=indices(z,order,'pred_offsets');gi=indices(z,order,'gt_offsets')
    return {'images':np.asarray(paths),'tp':z['tp'][pi],'conf':z['conf'][pi],'pred_cls':z['pred_cls'][pi],
        'target_cls':z['target_cls'][gi],
        'pred_offsets':np.cumsum([0]+[z['pred_offsets'][i+1]-z['pred_offsets'][i] for i in order]),
        'gt_offsets':np.cumsum([0]+[z['gt_offsets'][i+1]-z['gt_offsets'][i] for i in order])}

def scalars(matrix):
    macro=np.asarray(matrix).mean(axis=-2)
    return np.stack((macro.mean(axis=-1),macro[...,0],macro[...,5],macro[...,6],macro[...,7],macro[...,8],macro[...,9],macro[...,5:].mean(axis=-1)),axis=-1)

METRICS=['mAP50_95','AP50','AP75','AP80','AP85','AP90','AP95','H']

def main():
    global RECORDS,PATIENT_IMAGES
    stage('P4_PATIENT_BOOTSTRAP')
    if read_json(ROOT/'evaluation_check.json')['status']!='PASS':raise RuntimeError('Official evaluation incomplete')
    v=protocol();pop=read_json(v['pilot']['population_json'])['validation'];paths=Path(v['development_validation_list']).read_text().splitlines()
    bypath={str(r['path']):str(r['group']) for r in pop};patients=sorted(set(bypath.values()))
    if len(patients)!=990:raise RuntimeError('Bootstrap patient count mismatch')
    PATIENT_IMAGES=[np.asarray([i for i,p in enumerate(paths) if bypath[p]==patient],dtype=np.int64) for patient in patients]
    observed=[];reports=[]
    for name in NAMES:
        out=ROOT/'evaluations'/name
        with np.load(out/'bootstrap_records.npz',allow_pickle=False) as inp:z={k:inp[k] for k in inp.files}
        if len(z['images'])!=1660 or len(set(z['images']))!=1660 or set(z['images'])!=set(paths):raise RuntimeError('Incomplete bootstrap image universe')
        # Verify unsampled exported matching statistics exactly reproduce official AP.
        expected=np.asarray(read_json(out/'per_iou_ap.json')['ap_by_class'])
        if not np.allclose(ap_matrix(z),expected,rtol=0,atol=1e-12):raise RuntimeError(f'Bootstrap export differs from official AP: {name}')
        RECORDS.append(reorder(z,paths));observed.append(expected);reports.append(read_json(out/'eval_result.json'))
    # GT has to be identical for every model, including negative images.
    for z in RECORDS[1:]:
        if not np.array_equal(z['gt_offsets'],RECORDS[0]['gt_offsets']) or not np.array_equal(z['target_cls'],RECORDS[0]['target_cls']):raise RuntimeError('GT changed between models')
    observed=np.asarray(observed);vals=scalars(observed)
    raw=[]
    for i,name in enumerate(NAMES):raw.append({'name':name,**dict(zip(METRICS,vals[i].tolist()))})
    summary={'metric_unit':'fraction','models':raw,'comparisons':{},'seed_summary':{},'exposed_development':True,'no_external_validation':True}
    for arm in ('B2','AXIS'):
        a=np.stack([vals[NAMES.index(f'{arm}_seed{s}')] for s in SEEDS])
        summary['seed_summary'][arm]={'mean':dict(zip(METRICS,a.mean(0).tolist())),'sample_SD':dict(zip(METRICS,a.std(0,ddof=1).tolist()))}
    comparisons={}
    for label,left,right in (('AXIS_minus_A0-R','AXIS','A0-R'),('AXIS_minus_B2','AXIS','B2'),('B2_minus_A0-R','B2','A0-R')):
        pairs=[(NAMES.index(f'{left}_seed{s}'),0 if right=='A0-R' else NAMES.index(f'{right}_seed{s}')) for s in SEEDS]
        comparisons[label]=pairs
        a=np.stack([vals[l]-vals[r] for l,r in pairs])
        summary['comparisons'][label]={'by_seed':{str(s):dict(zip(METRICS,row.tolist())) for s,row in zip(SEEDS,a)},'mean':dict(zip(METRICS,a.mean(0).tolist())),'sample_SD':dict(zip(METRICS,a.std(0,ddof=1).tolist()))}
    atomic_json(ROOT/'comparison_summary.json',summary)
    with (ROOT/'per_class.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','class_id','class','AP50','mAP50_95',*[f'AP{50+j*5}' for j in range(10)]])
        for n,matrix in zip(NAMES,observed):
            for c,row in enumerate(matrix):w.writerow([n,c,v['class_names'][c],row[0],row.mean(),*row])
    with (ROOT/'convergence.csv').open('w',newline='') as f:
        fields=['seed','arm','epoch','steps','mean_loss','mean_native_box','mean_q','mean_geo']
        w=csv.DictWriter(f,fields);w.writeheader()
        for s in SEEDS:
            for a in ('B2','AXIS'):
                rows=[json.loads(t) for t in (ROOT/'runs'/f'seed{s}'/a/'train_metrics.jsonl').read_text().splitlines()]
                for e in range(1,51):
                    rr=[r for r in rows if r['epoch']==e]
                    if len(rr)!=83:raise RuntimeError('Convergence epoch incomplete')
                    w.writerow({'seed':s,'arm':a,'epoch':e,'steps':83,**{f'mean_{k}':np.mean([r[k] for r in rr]) for k in ('loss','native_box','q','geo')}})
    rng=np.random.Generator(np.random.PCG64(20260930));draws=rng.integers(0,990,size=(10000,990),dtype=np.int32)
    matrices=np.empty((10000,7,6,10),dtype=np.float64)
    # fork shares immutable evaluation arrays; workers perform CPU AP only.
    with mp.get_context('fork').Pool(12) as pool:
        for done,(j,m) in enumerate(pool.imap_unordered(draw_job,enumerate(draws),chunksize=4),1):
            matrices[j]=m
            if done%100==0:
                atomic_json(ROOT/'bootstrap_progress.json',{'completed':done,'total':10000,'GPU_tasks_finished':True})
                print(f'Bootstrap {done}/10000',flush=True)
    np.savez_compressed(ROOT/'bootstrap_samples.npz',ap_by_model_class_iou=matrices)
    bootvals=scalars(matrices)
    bs={'status':'PASS','resamples':10000,'rng':'PCG64','seed':20260930,'patients':990,'unit':'patient_group_all_images','recompute_official_AP':True,'official_stat_identity_PASS':True,'conditional_on_fixed_models_and_exposed_development':True,'does_not_correct_selection_bias':True,'metric_unit':'fraction','comparisons':{}}
    for label,pairs in comparisons.items():
        diffs=np.stack([bootvals[:,l]-bootvals[:,r] for l,r in pairs],axis=1)
        def ci(x):return {k:np.quantile(x[:,j],[.025,.975]).tolist() for j,k in enumerate(METRICS)}
        bs['comparisons'][label]={'by_seed':{str(s):ci(diffs[:,j]) for j,s in enumerate(SEEDS)},'three_seed_mean_difference':ci(diffs.mean(1))}
    atomic_json(ROOT/'bootstrap_summary.json',bs)
    lines=['# B2 / AXIS Plan B full-training report','','Status: VALID_COMPLETE','','Six runs completed E50; 4,150 optimizer steps per arm. E50 is the sole primary checkpoint.','','## Official development metrics','','|Model|AP50 (%)|mAP50–95 (%)|H (%)|','|---|---:|---:|---:|']
    for r in raw:lines.append(f"|{r['name']}|{r['AP50']*100:.6f}|{r['mAP50_95']*100:.6f}|{r['H']*100:.6f}|")
    lines+=['','## All fixed metrics (%)','','|Model|'+ '|'.join(METRICS)+'|','|---|'+ '|'.join(['---:']*len(METRICS))+'|']
    for r in raw:lines.append('|'+r['name']+'|'+ '|'.join(f'{r[k]*100:.6f}' for k in METRICS)+'|')
    lines+=['','## Three-seed mean and sample SD (%)','','|Arm / statistic|'+ '|'.join(METRICS)+'|','|---|'+ '|'.join(['---:']*len(METRICS))+'|']
    for arm in ('B2','AXIS'):
        for stat in ('mean','sample_SD'):lines.append('|'+arm+' / '+stat+'|'+ '|'.join(f"{summary['seed_summary'][arm][stat][k]*100:.6f}" for k in METRICS)+'|')
    lines+=['','## Paired differences (percentage points)','','|Comparison|3-seed mean mAP difference|Patient-bootstrap conditional 95% CI|','|---|---:|---:|']
    for label in comparisons:
        diff=summary['comparisons'][label]['mean']['mAP50_95']*100;lo,hi=np.array(bs['comparisons'][label]['three_seed_mean_difference']['mAP50_95'])*100
        lines.append(f'|{label}|{diff:+.6f}|[{lo:+.6f}, {hi:+.6f}]|')
    lines+=['','## Interpretation boundary','','AXIS − A0-R evaluates the complete system. AXIS − B2 evaluates the additional axis geometry term. Report all signs, all seeds, and sample SD; no result direction changes VALID_COMPLETE.','','Development has been repeatedly exposed. These conditional confidence intervals do not remove model-selection bias or establish independent external generalization. Test data were not read. Seed42 E50 is the predeclared representative; no best epoch or seed selection.','','## Files','','See comparison_summary.json, bootstrap_summary.json, per_class.csv, convergence.csv, pairing_check.json, evaluation_check.json and gpu_holder_restore_status.json.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    restoration=read_json(ROOT/'gpu_holder_restore_status.json')
    if restoration['status'] not in ('RESTORED','ALREADY_RUNNING'):raise RuntimeError('GPU holder restoration was not verified')
    finish('VALID_COMPLETE');stage('P0_P5_COMPLETE')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        finish('INCOMPLETE' if isinstance(e,KeyboardInterrupt) else 'INVALID',f'P4: {type(e).__name__}: {e}')
        raise
