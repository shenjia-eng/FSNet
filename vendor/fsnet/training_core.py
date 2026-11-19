"""Optimization, loss routing, RNG state, and atomic checkpoints."""
from pathlib import Path
import copy, hashlib, json, math, os, random, shutil
import numpy as np
import torch

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def atomic_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with t.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def atomic_torch(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with t.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def atomic_copy(source,dest):
    p=Path(dest);t=p.with_name(p.name+'.tmp');shutil.copyfile(source,t)
    with t.open('rb') as f:os.fsync(f.fileno())
    os.replace(t,p)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():raise RuntimeError('CUDA RNG restore requires CUDA')
        torch.cuda.set_rng_state_all(state['cuda'])

def seed_all(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    seed=torch.initial_seed()%(2**32);np.random.seed(seed);random.seed(seed)

def checked_loss(losses,weights):
    missing={k for k,v in weights.items() if v!=0 and k not in losses}
    unexpected=set(losses)-set(weights)
    if missing or unexpected:raise ValueError(f'Loss routing mismatch: missing={sorted(missing)} unweighted={sorted(unexpected)}')
    if any(not math.isfinite(float(v)) or v<0 for v in weights.values()):raise ValueError('Invalid loss coefficient')
    parts={k:losses[k]*weights[k] for k in losses}
    total=sum(parts.values())
    if not torch.is_tensor(total) or not total.requires_grad:raise ValueError('Loss detached or empty')
    if not torch.isfinite(total).all():raise FloatingPointError('Nonfinite loss')
    return total,{k:float(v.detach()) for k,v in losses.items()},{k:float(v.detach()) for k,v in parts.items()}

def build_optimizer(model,criterion,args):
    rates={'head':args.lr,'backbone':args.lr_backbone,'proj':args.lr_proj,'ffparser':args.lr_ffparser,'criterion':args.lr}
    bins={k:[] for k in rates};names={k:[] for k in rates};seen=set()
    for name,p in list(model.named_parameters())+[('criterion.'+n,p) for n,p in criterion.named_parameters()]:
        if id(p) in seen:raise ValueError('Duplicate optimizer tensor')
        seen.add(id(p))
        key=('criterion' if name.startswith('criterion.') else 'ffparser' if 'ffparser' in name else
             'proj' if name.startswith('backbone.proj') else 'backbone' if name.startswith('backbone.') else 'head')
        bins[key].append(p);names[key].append(name)
    groups=[dict(params=bins[k],param_names=names[k],name=k,lr=float(rates[k]),
                 base_lr=float(rates[k]),weight_decay=float(args.weight_decay)) for k in rates if bins[k]]
    if args.weight_decay<0 or any(g['lr']<=0 for g in groups):raise ValueError('Invalid optimizer settings')
    # Includes temporarily frozen tensors: their state/group identity remains stable at unfreeze.
    return torch.optim.Adam(groups,lr=args.lr,weight_decay=args.weight_decay)

def group_signature(optimizer):
    return [{'name':g['name'],'param_names':list(g['param_names']),
             'shapes':[list(p.shape) for p in g['params']]} for g in optimizer.param_groups]

class BudgetSchedule:
    """Explicit epoch-indexed warmup+cosine; no hidden scheduler/optimizer rebinding."""
    def __init__(self,optimizer,total_epochs,warmup_epochs=0,min_factor=.01):
        if total_epochs<2 or not 0<=warmup_epochs<total_epochs or not 0<min_factor<=1:
            raise ValueError('Invalid scheduler budget')
        self.optimizer=optimizer;self.total=int(total_epochs);self.warmup=int(warmup_epochs);self.minimum=float(min_factor)
        self.epoch=None
    def set_epoch(self,epoch):
        if not 0<=epoch<self.total:raise ValueError('Epoch outside scheduler budget')
        if epoch<self.warmup:factor=(epoch+1)/self.warmup
        else:
            progress=(epoch-self.warmup)/(self.total-self.warmup-1) if self.total-self.warmup>1 else 1.
            factor=self.minimum+(1-self.minimum)*(1+math.cos(math.pi*progress))/2
        for g in self.optimizer.param_groups:g['lr']=g['base_lr']*factor
        self.epoch=epoch
        return {g['name']:g['lr'] for g in self.optimizer.param_groups}
    def state_dict(self):
        return {'total_epochs':self.total,'warmup_epochs':self.warmup,'min_factor':self.minimum,'epoch':self.epoch}
    def load_state_dict(self,state):
        if (state['total_epochs'],state['warmup_epochs'],state['min_factor'])!=(self.total,self.warmup,self.minimum):
            raise ValueError('Scheduler protocol mismatch')
        if state['epoch'] is not None:self.set_epoch(state['epoch'])

def configure_training_mode(model,freeze_backbone):
    model.train()
    for name,p in model.named_parameters():
        if name.startswith('backbone.'):p.requires_grad_(not freeze_backbone)
    if freeze_backbone and getattr(model,'_freeze_backbone_bn',True):model.backbone.eval()

def improved_mae(candidate,best):
    if not math.isfinite(candidate):raise FloatingPointError('Nonfinite selection metric')
    return candidate<best # strict improvement; exact ties retain earlier checkpoint.

def initialize_model(model,path,allowed_missing=(),allowed_unexpected=()):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    state=ck.get('model',ck);current=model.state_dict()
    missing=set(current)-set(state);extra=set(state)-set(current)
    if any(not any(k.startswith(p) for p in allowed_missing) for k in missing):raise ValueError('Undeclared new tensors: '+str(sorted(missing)))
    if any(not any(k.startswith(p) for p in allowed_unexpected) for k in extra):raise ValueError('Undeclared dropped tensors: '+str(sorted(extra)))
    reshaped=[]
    for k in current.keys()&state.keys():
        if current[k].shape!=state[k].shape and k.endswith('.gamma') and current[k].numel()==state[k].numel()==1:
            state[k]=state[k].reshape(current[k].shape);reshaped.append(k)
    incompatible=[k for k in current.keys()&state.keys() if current[k].shape!=state[k].shape]
    if incompatible:raise ValueError('Shape mismatch: '+str(incompatible))
    model.load_state_dict({k:v for k,v in state.items() if k in current},strict=False)
    return {'parent':str(path),'sha256':sha256(path),'parent_epoch':ck.get('epoch'),
            'mode':'model_only_new_stage','missing':sorted(missing),'dropped':sorted(extra),
            'optimizer_reset':True,'scheduler_reset':True,'scalar_gamma_reshape_preserving_values':reshaped}

def checkpoint_payload(model,criterion,optimizer,schedule,epoch,best_mae,best_epoch,history,identity):
    return {'format':'fsnet-training-v2','model':model.state_dict(),'criterion':criterion.state_dict(),
            'optimizer':optimizer.state_dict(),'optimizer_signature':group_signature(optimizer),
            'scheduler':schedule.state_dict(),'epoch':epoch,'best_mae':best_mae,'best_epoch':best_epoch,
            'rng':rng_state(),'history':history,'identity':identity,'amp':None,
            **({'gradnorm':criterion.gradnorm.state_dict()} if hasattr(criterion,'gradnorm') else {})}

def recover_checkpoint(path,model,criterion,optimizer,schedule,identity,restore_random=True):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    required={'format','model','criterion','optimizer','optimizer_signature','scheduler','epoch','best_mae','best_epoch','rng','history','identity','amp'}
    if not required.issubset(ck) or ck['format']!='fsnet-training-v2':raise ValueError('Not a full v2 resume checkpoint')
    if ck['identity']!=identity:raise ValueError('Config/source/data identity mismatch; use a new experiment')
    if ck['optimizer_signature']!=group_signature(optimizer):raise ValueError('Optimizer tensor identity mismatch')
    model.load_state_dict(ck['model'],strict=True);criterion.load_state_dict(ck['criterion'],strict=True)
    optimizer.load_state_dict(ck['optimizer']);schedule.load_state_dict(ck['scheduler'])
    if restore_random:restore_rng(ck['rng'])
    return ck

def export_history(path,history):
    p=Path(path);t=p.with_name(p.name+'.tmp')
    with t.open('w',encoding='utf-8') as f:
        for row in history:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def commit_epoch(run_dir,payload,is_best):
    rd=Path(run_dir);last=rd/'last.pth'
    atomic_torch(last,payload)
    if is_best:atomic_copy(last,rd/'best.pth')
    export_history(rd/'epochs.jsonl',payload['history'])

def reconcile_after_resume(run_dir,checkpoint):
    rd=Path(run_dir)
    # last.pth is the authoritative transaction. Repair derived outputs after an interrupted save.
    if checkpoint['best_epoch']==checkpoint['epoch']:atomic_copy(rd/'last.pth',rd/'best.pth')
    else:
        best=torch.load(rd/'best.pth',map_location='cpu',weights_only=False)
        if best.get('identity')!=checkpoint['identity'] or best.get('epoch')!=checkpoint['best_epoch']:
            raise ValueError('Missing or mismatched best checkpoint')
    for row in checkpoint['history']:
        if sha256(rd/row['prediction_file'])!=row['prediction_sha256']:
            raise ValueError('Missing/corrupt committed prediction artifact')
    export_history(rd/'epochs.jsonl',checkpoint['history'])
