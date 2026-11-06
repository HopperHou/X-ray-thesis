from pathlib import Path
import hashlib,json

root=Path(__file__).resolve().parent
m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for name,sha in m['sha256'].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=sha:raise SystemExit('FAIL '+name)
for item in m['sections']:
    p=root/item['files'][1]
    if '\\label{'+item['label']+'}' not in p.read_text(encoding='utf-8'):raise SystemExit('FAIL section '+item['label'])
print(json.dumps({'status':'PASS','files':len(m['sha256']),'sections':len(m['sections'])}))
