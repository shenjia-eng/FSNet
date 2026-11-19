"""Transactional checkpoints, model candidates, and training history."""
import os,json,hashlib,math,fcntl,io,copy,random,shutil
import numpy as np
from pathlib import Path
from contextlib import contextmanager
import torch
from selection import pool,select,strict_mae,f1_best
VERSION='composite-archive-v5'
POINTS=('after_model','after_history','after_full','after_manifest','before_current','after_current','after_gc')
def fail(message):raise ValueError(message)
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def strict_rows(rows):
 if not rows:fail('Empty history')
 for epoch,row in enumerate(rows,1):
  if set(row)!=set(('epoch','mae','rmse','f1','prediction')):fail('History fields')
  if type(row['epoch']) is not int or row['epoch']!=epoch:fail('Noncontinuous integer epoch')
  for k in ('mae','rmse','f1'):
   v=row[k]
   if type(v) not in (int,float) or not math.isfinite(v) or v<0:fail('Invalid metric')
  if row['f1']>1:fail('Invalid f1')
  if set(row['prediction'])!=set(('path','sha256')):fail('Prediction receipt')
def tensors_equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(tensors_equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(tensors_equal(x,y) for x,y in zip(a,b))
 return a==b
def freeze(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:freeze(v) for k,v in x.items()}
 if isinstance(x,list):return [freeze(v) for v in x]
 if isinstance(x,tuple):return tuple(freeze(v) for v in x)
 return copy.deepcopy(x)
class Archive:
 def __init__(self,root,identity):
  self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True);self.identity=copy.deepcopy(identity);self._verified={};self._models={}
  for d in ('objects','commits','gc','recovery'): (self.root/d).mkdir(exist_ok=True)
 def syncdir(self,p):
  fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 @contextmanager
 def lock(self):
  with (self.root/'archive.lock').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
 def atomic(self,p,data):
  if shutil.disk_usage(self.root).free<3*2**30+len(data)+64*2**20:fail('Insufficient disk space for atomic write')
  temp=p.with_name(p.name+'.tmp')
  with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.replace(temp,p);self.syncdir(p.parent)
 def object(self,kind,data):
  name=f'objects/{kind}_{sha(data)}'
  p=self.root/name
  if p.exists():
   if p.read_bytes()!=data:fail('Object collision')
  else:self.atomic(p,data)
  return {'path':name,'sha256':sha(data),'bytes':len(data)}
 def tensor_object(self,kind,payload):
  f=io.BytesIO();torch.save(freeze(payload),f);return self.object(kind,f.getvalue())
 def verify_ref(self,ref):
  if set(ref)!=set(('path','sha256','bytes')):fail('Object reference fields')
  raw=self.root/ref['path'];p=raw.resolve()
  if not p.is_relative_to(self.root) or raw.is_symlink():fail('Path escape')
  st=p.stat();signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
  if type(ref['bytes']) is not int or st.st_size!=ref['bytes']:fail('Object length')
  key=(ref['path'],ref['sha256'],ref['bytes'])
  if self._verified.get(key)!=signature:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
   if h.hexdigest()!=ref['sha256']:fail('Object SHA')
   after=p.stat()
   if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)!=signature:fail('Object changed while validating')
   self._verified[key]=signature
  return p,signature
 def checked(self,ref):
  p,_=self.verify_ref(ref);return p.read_bytes()
 def tensor_load(self,ref):return torch.load(io.BytesIO(self.checked(ref)),map_location='cpu',weights_only=False)
 def prediction(self,receipt):
  if not isinstance(receipt['path'],str):fail('Prediction path')
  p=(self.root/receipt['path']).resolve()
  if not p.is_relative_to(self.root):fail('Prediction path outside archive root')
  if sha(p.read_bytes())!=receipt['sha256']:fail('Prediction SHA')
 def current_ref(self):
  p=self.root/'CURRENT'
  if not p.exists():
   if any((self.root/'commits').iterdir()) or any((self.root/'objects').iterdir()):fail('Missing CURRENT while checkpoint files exist; recovery required')
   return None
  return json.loads(p.read_bytes())
 def load(self,with_full=True):
  ref=self.current_ref()
  if ref is None:return None
  return self.load_manifest(ref,with_full)
 def project_history(self,history):
  strict_rows(history);return copy.deepcopy(history)
 def load_manifest(self,ref,with_full=True):
  m=json.loads(self.checked(ref))
  if m['version']!=VERSION or m['identity']!=self.identity:fail('Archive identity/version')
  history=json.loads(self.checked(m['history']));strict_rows(history)
  detailed=json.loads(self.checked(m['detailed_history']))
  if self.project_history(detailed)!=history:fail('Detailed/selection history mismatch')
  if m['epoch']!=len(history):fail('Commit epoch')
  for row in history:self.prediction(row['prediction'])
  expected=pool(history)
  wanted={str(x['epoch']) for x in expected}
  if set(m['models'])!=wanted:fail('Pool epochs mismatch')
  roles={'mae':strict_mae(history)['epoch'],'f1':f1_best(history)['epoch'],'joint':select(history)['epoch']}
  if m['roles']!=roles:fail('Role mismatch')
  for row in expected:
   refmodel=m['models'][str(row['epoch'])];_,signature=self.verify_ref(refmodel)
   key=(refmodel['sha256'],signature)
   if key not in self._models:
    obj=self.tensor_load(refmodel)
    if obj['format']!='model_only_candidate' or obj['identity']!=self.identity or obj['row']!=row or obj['detailed_row_sha256']!=sha(canonical(detailed[row['epoch']-1])):fail('Candidate identity/history mismatch')
    if not isinstance(obj['model'],dict) or not obj['model']:fail('Model state missing')
    if self.identity.get('gradnorm_protocol'):
     from gradnorm_integration import state_digest
     from gradnorm_core import GradNormBalancer
     gn=obj.get('gradnorm');expected=detailed[row['epoch']-1]['training']['gradnorm']
     if gn is None or state_digest(gn)!=expected['state_sha256']:fail('Candidate GradNorm binding')
     checker=GradNormBalancer('cpu',gn['config']);checker.load_state_dict(gn)
    self._models[key]=(copy.deepcopy(obj['row']),obj['detailed_row_sha256'])
   elif self._models[key]!=(row,sha(canonical(detailed[row['epoch']-1]))):fail('Cached candidate/history mismatch')
  result={'ref':ref,'manifest':m,'history':history,'detailed_history':detailed}
  self.verify_ref(m['last'])
  if with_full:
   full=self.tensor_load(m['last'])
   self.validate_full(full,history)
   if full['history']!=detailed:fail('Full detailed history mismatch')
   result['full']=full
   if str(m['epoch']) in m['models']:
    model=self.tensor_load(m['models'][str(m['epoch'])])['model']
    if not tensors_equal(model,full['model']):fail('Last model/pool disagree')
  return result
 def validate_full(self,full,history):
  needed={'format','model','criterion','optimizer','optimizer_signature','scheduler','epoch','best_mae','best_epoch','rng','history','identity','amp'}
  if not needed.issubset(full) or full['format']!='fsnet-training-v2':fail('Incomplete full state')
  if full['identity']!=self.identity or full['epoch']!=len(history)-1 or self.project_history(full['history'])!=history:fail('Full epoch/history/identity')
  ref=strict_mae(history)
  if full['best_mae']!=ref['mae'] or full['best_epoch']!=ref['epoch']-1:fail('Full strict MAE metadata')
  if not isinstance(full['rng'],dict) or not {'python','numpy','torch','cuda'}.issubset(full['rng']):fail('Incomplete RNG')
  if not isinstance(full['model'],dict) or not full['model']:fail('Empty model')
  if not isinstance(full['scheduler'],dict) or full['scheduler'].get('epoch')!=len(history)-1:fail('Scheduler epoch')
  opt=full['optimizer']
  if not isinstance(opt,dict) or set(opt)!=set(('state','param_groups')) or not opt['param_groups']:fail('Optimizer structure')
  ids=[i for g in opt['param_groups'] for i in g.get('params',[])]
  if not ids or len(ids)!=len(set(ids)) or not set(opt['state']).issubset(ids) or not opt['state']:fail('Optimizer parameter identities')
  for g in opt['param_groups']:
   if not {'lr','betas','eps','weight_decay','params'}.issubset(g):fail('Adam group')
  for state in opt['state'].values():
   if not {'step','exp_avg','exp_avg_sq'}.issubset(state):fail('Adam moments')
   if not all(torch.is_tensor(state[k]) for k in ('step','exp_avg','exp_avg_sq')):fail('Adam moment types')
   if state['exp_avg'].shape!=state['exp_avg_sq'].shape:fail('Adam moment shape')
   if not all(torch.isfinite(state[k]).all().item() for k in ('step','exp_avg','exp_avg_sq')):fail('Nonfinite Adam')
  rng=full['rng']
  try:
   random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
   torch.Generator(device='cpu').set_state(rng['torch'])
  except Exception as exc:raise ValueError('Invalid RNG state') from exc
  if not isinstance(rng['cuda'],list) or any(not torch.is_tensor(t) or t.dtype!=torch.uint8 or t.ndim!=1 for t in rng['cuda']):fail('CUDA RNG structure')
  if self.identity.get('gradnorm_protocol'):
   from gradnorm_integration import state_digest
   from gradnorm_core import GradNormBalancer
   gn=full.get('gradnorm')
   if gn is None or state_digest(gn)!=full['history'][-1]['training']['gradnorm']['state_sha256']:fail('Full GradNorm history binding')
   checker=GradNormBalancer('cpu',gn['config']);checker.load_state_dict(gn)
  if full.get('archive_history_sha256') not in (None,sha(canonical(full['history']))):fail('Full detailed history digest')
  if full.get('archive_selection_sha256') not in (None,sha(canonical(history))):fail('Full selection history digest')
 def commit(self,row,full,previous_sha,hook=lambda _:None):
  with self.lock():
   old=self.load(with_full=False)
   actual=old['ref']['sha256'] if old else None
   if actual!=previous_sha:fail('Stale previous commit')
   history=(old['history'] if old else [])+[copy.deepcopy(row)];strict_rows(history)
   self.validate_full(full,history);self.prediction(row['prediction'])
   if old and canonical(full['history'][:-1])!=canonical(old['detailed_history']):fail('Detailed history prefix changed')
   wanted=pool(history);available=dict(old['manifest']['models']) if old else {}
   if row['epoch'] in {x['epoch'] for x in wanted}:
    available[str(row['epoch'])]=self.tensor_object('model',{'format':'model_only_candidate','identity':self.identity,'row':row,'detailed_row_sha256':sha(canonical(full['history'][-1])),'model':full['model'],**({'gradnorm':full['gradnorm']} if 'gradnorm' in full else {})})
   hook('after_model')
   models={str(x['epoch']):available[str(x['epoch'])] for x in wanted}
   h=self.object('history',canonical(history));dh=self.object('detail',canonical(full['history']));hook('after_history')
   payload=dict(full);payload['archive_history_sha256']=dh['sha256'];payload['archive_selection_sha256']=h['sha256']
   last=self.tensor_object('full',payload);hook('after_full')
   manifest={'version':VERSION,'identity':self.identity,'epoch':len(history),'previous_commit_sha256':previous_sha,
    'history':h,'detailed_history':dh,'models':models,'last':last,'roles':{'mae':strict_mae(history)['epoch'],'f1':f1_best(history)['epoch'],'joint':select(history)['epoch']},
    'historical_manifests':'evidence_only_after_gc'}
   data=canonical(manifest);name=f'commits/epoch_{len(history):04d}_{sha(data)}.json'
   p=self.root/name
   if p.exists():
    if p.read_bytes()!=data:fail('Commit collision')
   else:self.atomic(p,data)
   new={'path':name,'sha256':sha(data),'bytes':len(data)};hook('after_manifest');hook('before_current')
   self.atomic(self.root/'CURRENT',canonical(new));hook('after_current')
   return self.load(with_full=False)
 def resolve_model(self,role):
  state=self.load(with_full=False);m=state['manifest'];epoch=m['roles'][role]
  return self.tensor_load(m['models'][str(epoch)])
 def gc(self,expected_sha,hook=lambda _:None):
  with self.lock():
   state=self.load(with_full=False)
   if state['ref']['sha256']!=expected_sha:fail('Stale GC')
   m=state['manifest'];keep={m['history']['path'],m['detailed_history']['path'],m['last']['path']}|{r['path'] for r in m['models'].values()}
   removed=[]
   for p in sorted((self.root/'objects').iterdir()):
    rel=p.relative_to(self.root).as_posix()
    if rel not in keep and p.is_file():
     removed.append({'path':rel,'sha256':sha(p.read_bytes())})
   log={'current_commit_sha256':expected_sha,'removal_intent':removed,'history_not_rollback_capable':True}
   event=canonical(log);event_path=self.root/'gc'/f'{expected_sha}_{sha(event)}.json'
   if event_path.exists():
    if event_path.read_bytes()!=event:fail('GC record collision')
   else:self.atomic(event_path,event)
   for item in removed:(self.root/item['path']).unlink()
   self.syncdir(self.root/'objects');hook('after_gc')
   return self.load(with_full=False)


 def recover_initial_prepared(self,expected_ref,validate_objects,hook=lambda _:None):
  if not callable(validate_objects):fail('Explicit model/optimizer validation required')
  with self.lock():
   current=self.root/'CURRENT';published=current.exists()
   if published:
    if json.loads(current.read_bytes())!=expected_ref:fail('Recovery CURRENT conflict')
    state=self.load(with_full=True)
   else:
    entries=list((self.root/'commits').iterdir())
    if len(entries)!=1 or entries[0].resolve()!=(self.root/expected_ref['path']).resolve():fail('Expected one matching initial checkpoint record')
    if any((self.root/'gc').iterdir()):fail('GC indicates prior execution')
    state=self.load_manifest(expected_ref,with_full=True)
   m=state['manifest']
   if m['epoch']!=1 or m['previous_commit_sha256'] is not None:fail('Not sole initial complete commit')
   validate_objects(state['full'])
   intent={'action':'recover_missing_initial_pointer','manifest':expected_ref,'identity':self.identity}
   data=canonical(intent);path=self.root/'recovery'/('intent_'+sha(data)+'.json')
   if path.exists():
    if path.read_bytes()!=data:fail('Recovery intent conflict')
   else:self.atomic(path,data)
   hook('after_recovery_intent')
   if not published:self.atomic(current,canonical(expected_ref))
   hook('after_recovery_current')
   result=self.load(with_full=True)
   done=canonical({'intent_sha256':sha(data),'current':result['ref'],'status':'published_and_revalidated'})
   p=self.root/'recovery'/('completed_'+sha(data)+'.json')
   if p.exists():
    if p.read_bytes()!=done:fail('Recovery completion conflict')
   else:self.atomic(p,done)
   return result
