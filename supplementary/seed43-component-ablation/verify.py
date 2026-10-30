"""Read-only verification of the released evidence; Python standard library only.

Does not reconstruct predictions, decode radiographs, or verify remote weights.
The separate COLLECTION_VERIFICATION.json records those server-side checks.
"""
from pathlib import Path
import csv, hashlib, json, math, statistics

P = Path(__file__).resolve().parent
def j(name): return json.loads((P/name).read_text(encoding='utf-8'))
def csvrows(name):
    with (P/name).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def close(a,b): assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),(a,b)

for name,digest in j('SHA256SUMS.json').items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
main=j('MAIN_CHECKPOINTS.json')
assert len(main['records'])==8
assert {r['id'] for r in main['records']}=={'A0-R','M3 E18'}|{f'{a} seed {s}' for a in ['B2','AXIS'] for s in [42,43,44]}
for r in main['records']:
    assert r['original_path']==r['relative_path_base']+r['original_relative_path']
    assert len(r['sha256'])==64 and set(r['sha256'])<=set('0123456789abcdef')
    for source in r['sources']:
        path=P.parents[1]/source['repository_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['source_file_sha256']
    s=json.loads((P.parents[1]/r['sources'][0]['repository_path']).read_text())
    if r['id']=='M3 E18':
        s=s['source_m3_e0']
        assert s['expected_sha256']==r['sha256'] and s['path']==r['original_path']
    else:
        assert s['checkpoint_sha256']==r['sha256'] and s['checkpoint']==r['original_path']
arms=['B2','AXIS','O_ONLY','O_ISOTROPIC']
streams={arm:[json.loads(x) for x in (P/f'training/{arm}/input_hashes.jsonl').read_text().splitlines()] for arm in arms}
base=streams['B2']
assert len(base)==4150
for arm,rows in streams.items():
    assert len(rows)==4150
    for a,b in zip(base,rows):
        assert {k:v for k,v in a.items() if k!='arm'}=={k:v for k,v in b.items() if k!='arm'}
    status=j(f'training/{arm}/status.json')
    assert status['total_optimizer_steps']==4150 and status['freeze_verified']
    for k in ['initial_box_hash','initial_q_hash']:
        assert status[k]==j('training/B2/status.json')[k]
cal=j('original/CALIBRATION.json'); samples=csvrows('CALIBRATION_SAMPLE_LIST.csv')
assert len(samples)==720 and len({s['training_image'] for s in samples})==720
for k,row in enumerate(cal['rows']):
    paths=[s['original_training_path'] for s in samples if int(s['batch'])==k+1]
    assert len(paths)==180
    assert hashlib.sha256(json.dumps(paths,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==row['ordered_paths_sha256']
    assert row['input_sha256']==base[k]['input_sha256']
    close(row['axis_norm']/row['isotropic_norm'],row['ratio'])
close(cal['scale'],statistics.median(r['ratio'] for r in cal['rows']))
raw=csvrows('original/E2_SEED43_RESULTS.csv'); cells=csvrows('original/E2_CLASS_IOU_RESULTS.csv')
assert len(cells)==480
for split,n in [('validation',1660),('internal_test',1844)]:
    manifest=[r for r in csvrows('EVALUATION_IMAGE_MANIFEST.csv') if r['partition']==split]
    assert len(manifest)==n and len({r['training_image'] for r in manifest})==n
    for arm in arms:
        a=j(f'evaluation/{split}/{arm}/validator_AP.json')['ap_by_class']
        assert len(a)==6 and all(len(r)==10 for r in a)
        sub=[r for r in cells if r['partition']==split and r['arm']==arm]
        assert len(sub)==60
        for r in sub:
            k=round((float(r['IoU_threshold'])-.5)/.05)
            close(r['AP_fraction'],a[int(r['class_index'])][k])
        r=next(r for r in raw if r['partition']==split and r['arm']==arm)
        close(r['mAP50_percent'],100*statistics.mean(x[0] for x in a))
        close(r['mAP50_95_percent'],100*statistics.mean(x for row in a for x in row))
for r in csvrows('HISTORICAL_VS_A100.csv'):
    old=j(f"historical_evaluation/{r['partition']}/{r['configuration']}/result.json")
    close(r['historical_percent'],100*old[r['metric']])
    close(r['delta_percentage_points'],float(r['A100_percent'])-float(r['historical_percent']))
print(json.dumps({'status':'PASS','training_streams':4,'updates_per_stream':4150,
 'calibration_images':720,'AP_matrices':8,'AP_cells':480,'evaluation_images':3504,
 'scope':'Released hashes, ordering, calibration arithmetic, AP exports and historical differences; not model inference or remote checkpoint verification.'},indent=2))
