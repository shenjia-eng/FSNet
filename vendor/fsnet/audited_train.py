"""FSNet stage training and evaluation driver."""
from pathlib import Path
import argparse,fcntl,json,os,sys,time,platform,math,contextlib,shutil
from datetime import datetime,timezone
ROOT=Path(os.environ.get('FSNET_OUTPUT_ROOT', 'outputs'))
HERE=Path(__file__).resolve().parent
TMP=ROOT/'tmp/gradnorm_fsnet_20261017_v2';TMP.mkdir(parents=True,exist_ok=True)
os.environ.update(TMPDIR=str(TMP),MPLCONFIGDIR=str(TMP/'matplotlib'),TORCH_HOME=str(ROOT/'tmp/torch'),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.path.insert(0,str(HERE/'source'))
import numpy as np
import torch
import train as legacy
from training_core import *
from metrics import evaluate_loader,evaluation_collate
from variants import apply_variant
from crowd_datasets.SHHA.SHHA import load_data
from torchvision import transforms
from util.misc import collate_fn_crowd

def runtime_source_hash():
    return canonical_hash({str(p.relative_to(HERE)):sha256(p) for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in p.parts})

def safe_run_path(path):
    p=Path(path).resolve();allowed=(ROOT/'04_运行/formal_single_seed_20261016').resolve()
    if not p.is_relative_to(allowed) or p==allowed:raise ValueError('Run directory must be a named child of run root')
    return p

def args_from(cfg):
    a=legacy.get_args_parser().parse_args([])
    unknown=set(cfg['args'])-set(vars(a))-{'backbone_pretrained'}
    if unknown:raise ValueError('Unknown model/training args: '+str(sorted(unknown)))
    vars(a).update(cfg['args'])
    if a.resume or a.init_weights:raise ValueError('Use top-level initialization')
    if a.uncertainty_weighting:raise ValueError('Uncertainty weighting is unsupported by this driver')
    if a.point_loss_coef<=0:raise ValueError('Positive point loss coefficient is mandatory')
    if getattr(a,'freeze_modules',''):raise ValueError('Additional freeze_modules not supported')
    return a

def build(cfg,device):
    a=args_from(cfg)
    if getattr(a,'backbone_pretrained',True):
        declared=cfg.get('backbone_initialization')
        if not declared:raise ValueError('Explicit backbone initialization identity required')
        import models.backbone as bb
        actual=Path(bb.models.model_paths[a.backbone]).resolve()
        if actual!=Path(declared['path']).resolve() or sha256(actual)!=declared['sha256']:
            raise ValueError('ImageNet backbone path/hash mismatch')
    model,criterion=legacy.build_model(a,training=True)
    apply_variant(model,cfg['variant'],a.seed)
    if 'loss_point' not in criterion.weight_dict or 'loss_points' in criterion.weight_dict:
        raise ValueError('Point loss routing is not fixed')
    model._freeze_backbone_bn=cfg.get('freeze_backbone_bn',True)
    from diagnostics import attach_diagnostics
    attach_diagnostics(criterion)
    model.to(device);criterion.to(device)
    from gradnorm_core import GradNormBalancer
    criterion.gradnorm=GradNormBalancer(device,cfg["gradnorm"])
    criterion.gradnorm_enabled=True
    return a,model,criterion

def gradient_summary(model):
    result={}
    for prefix in ('backbone.','regression.','classification.','ffparser','spatial_stem.','freq_spatial_gate.','det_freq_attn.'):
        ps=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        grads=[p.grad.detach() for p in ps if p.grad is not None]
        norm=math.sqrt(sum(float(g.double().square().sum()) for g in grads))
        result[prefix]={'trainable_tensors':len(ps),'tensors_with_grad':len(grads),'l2_norm':norm}
    return result

def train_epoch(model,criterion,optimizer,loader,device,epoch,freeze_epochs,clip_norm):
    configure_training_mode(model,epoch<freeze_epochs);criterion.train()
    signature=[(n,id(p),tuple(p.shape)) for n,p in model.named_parameters()]
    from diagnostics import DiagnosticAccumulator,spectral_stats,first_step_update
    diag=DiagnosticAccumulator();spectral_before=spectral_stats(model);initial_params=None
    totals={};samples_seen=steps=clipped_steps=0;grad_acc={};start=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for samples,targets in loader:
        samples=samples.to(device);targets=[{k:v.to(device) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        raw=criterion(model(samples),targets);total,raw_values,weighted=checked_loss(raw,criterion.weight_dict)
        diag.add(criterion,targets)
        total.backward()
        g=gradient_summary(model)
        if any(len(t['point']) for t in targets):
            rg=g['regression.']
            if rg['tensors_with_grad']==0 or not math.isfinite(rg['l2_norm']):raise RuntimeError('Regression gradient missing/nonfinite')
        parameters=[p for group in optimizer.param_groups for p in group['params']]
        norm=torch.nn.utils.clip_grad_norm_(parameters,clip_norm if clip_norm>0 else float('inf'),error_if_nonfinite=True)
        clipped_steps+=int(clip_norm>0 and float(norm)>clip_norm)
        if steps==0:initial_params={n:p.detach().clone() for n,p in model.named_parameters() if 'ffparser' in n or n=='classification.output.weight'}
        optimizer.step()
        if steps==0:first_update=first_step_update(model,initial_params)
        n=len(targets);samples_seen+=n;steps+=1
        values={'loss':float(total.detach()),'gradient_norm_before_clip':float(norm)}
        values.update({'raw/'+k:v for k,v in raw_values.items()})
        values.update({'weighted/'+k:v for k,v in weighted.items()})
        for k,v in values.items():totals[k]=totals.get(k,0.)+v*n
        for k,v in g.items():grad_acc[k]=grad_acc.get(k,0.)+v['l2_norm']
    if not steps:raise ValueError('Empty training epoch')
    if signature!=[(n,id(p),tuple(p.shape)) for n,p in model.named_parameters()]:
        raise RuntimeError('Parameters changed after optimizer construction')
    if device.type=='cuda':torch.cuda.synchronize()
    return {'diagnostics':diag.result(),'spectral_before':spectral_before,'spectral_after':spectral_stats(model),'first_step_parameter_update':first_update,'averages':{k:v/samples_seen for k,v in totals.items()},'effective_crops_seen':samples_seen,'steps':steps,
            'gradient_l2_mean':{k:v/steps for k,v in grad_acc.items()},'clipped_steps':clipped_steps,'gradient_clip_fraction':clipped_steps/steps,'seconds':time.perf_counter()-start,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else 0}

class ManifestValidationDataset(torch.utils.data.Dataset):
    def __init__(self,records):
        self.records=records;self.img_list=[r['image'] for r in records]
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index];image,points=load_data((r['image'],r['annotation']),False)
        target={'point':torch.tensor(points,dtype=torch.float32),'labels':torch.ones(len(points),dtype=torch.long),
                'image_id':torch.tensor([index],dtype=torch.long)}
        return self.transform(image),[target]

def read_data_manifest(cfg):
    path=Path(cfg['data_manifest']);data=json.loads(path.read_text())
    if sha256(path)!=cfg['data_manifest_sha256']:raise ValueError('Data manifest hash mismatch')
    for role,records in data.items():
        if role not in ('train','validation','evaluation'):continue
        seen=set()
        for r in records:
            if r['image'] in seen:raise ValueError('Duplicate image in '+role)
            seen.add(r['image'])
            for key in ('image','annotation'):
                if sha256(r[key])!=r[key+'_sha256']:raise ValueError('Input hash mismatch: '+r[key])
    if not data.get('train') or not data.get('validation'):raise ValueError('Explicit train and validation records required')
    return data

def protocol_from(cfg):
    p=cfg['evaluation']
    if set(p)!={'score_threshold','match_radius_px','coordinate_frame','bounds'}:raise ValueError('Evaluation protocol must be explicit')
    if not 0<=p['score_threshold']<=1 or p['match_radius_px']<=0:raise ValueError('Invalid evaluation threshold/radius')
    return p

def identity_for(cfg):
    return {'config_sha256':canonical_hash(cfg),'source_sha256':runtime_source_hash(),
            'data_manifest_sha256':cfg['data_manifest_sha256'],'protocol':'fsnet-training-v2'}

def record_environment():
    return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'precision':'FP32','tf32':False}

def configure_runtime(seed):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    seed_all(seed)

def run_training(cfg,resume=False,preflight_report=None):
    if not cfg.get('approved_for_training',False):raise ValueError('Training is disabled in the configuration')
    if not torch.cuda.is_available():raise RuntimeError('Training requires CUDA')
    if not preflight_report:raise ValueError('A passed GPU preflight report for this exact configuration/source is required')
    gate=json.loads(Path(preflight_report).read_text())
    if gate.get('status')!='passed' or gate.get('identity')!=identity_for(cfg):
        raise ValueError('GPU preflight is failed, stale or belongs to another configuration')
    if gate.get('environment')!=record_environment():raise ValueError('GPU/runtime changed since preflight')
    if cfg['args'].get('backbone_pretrained') is not True:raise ValueError('A pretrained backbone is required')
    device=torch.device('cuda');rd=safe_run_path(cfg['run_dir'])
    if not resume:
        if rd.exists() and any(rd.iterdir()):raise ValueError('Refusing to overwrite an existing run')
        rd.mkdir(parents=True,exist_ok=True)
    elif not (rd/'last.pth').is_file():raise FileNotFoundError('No last checkpoint')
    configure_runtime(cfg['args']['seed'])
    data=read_data_manifest(cfg);protocol=protocol_from(cfg);identity=identity_for(cfg)
    a,model,criterion=build(cfg,device)
    optimizer=build_optimizer(model,criterion,a)
    schedule=BudgetSchedule(optimizer,cfg['stage_epochs'],**cfg['schedule'])
    freeze=int(cfg['freeze_backbone_epochs'])
    if not 0<=freeze<cfg['stage_epochs']:raise ValueError('Invalid freeze budget')
    history=[];best=float('inf');best_epoch=-1;start_epoch=0
    if resume:
        ck=recover_checkpoint(rd/'last.pth',model,criterion,optimizer,schedule,identity)
        reconcile_after_resume(rd,ck);history=ck['history'];best=ck['best_mae'];best_epoch=ck['best_epoch'];start_epoch=ck['epoch']+1
    else:
        init=cfg['initialization']
        if init['mode']=='imagenet':
            inheritance={'mode':'imagenet','optimizer_reset':True,'backbone_initialization':cfg['backbone_initialization']}
        elif init['mode']=='stage_parent':
            if sha256(init['path'])!=init['sha256']:raise ValueError('Parent SHA mismatch')
            inheritance=initialize_model(model,init['path'],init['allowed_missing_prefixes'],init['allowed_unexpected_prefixes'])
        else:raise ValueError('Unknown initialization mode')
        atomic_json(rd/'inheritance.json',inheritance)
    os.environ['SHHA_TRAIN_LIST']=cfg['train_list']
    os.environ['SHHA_TEST_LIST']=cfg['validation_list']
    train_set,_=legacy.build_dataset(args=a)(legacy._resolve_data_roots(a))
    actual={str(k):str(v) for k,v in train_set.img_map.items()}
    expected={r['image']:r['annotation'] for r in data['train']}
    if actual!=expected:raise ValueError('Training loader differs from the manifest')
    valset=ManifestValidationDataset(data['validation'])
    valloader=torch.utils.data.DataLoader(valset,batch_size=1,shuffle=False,num_workers=0,collate_fn=evaluation_collate)
    atomic_json(rd/'config.json',cfg)
    atomic_json(rd/'runtime.json',{'identity':identity,'environment':record_environment(),'optimizer_groups':[
        {k:g[k] for k in ('name','lr','base_lr','weight_decay','param_names')} for g in optimizer.param_groups],
        'selection':'strict minimum validation MAE; exact ties keep earlier epoch','loss_weights':criterion.weight_dict})
    if not resume:
        atomic_json(rd/'progress.json',{'status':'initial_validation','completed_epochs':0,'total_epochs':cfg['stage_epochs']})
        print('Initial validation started',flush=True)
        initial=evaluate_loader(model,valloader,device,rd/'predictions/initial.jsonl.gz',protocol,valset.img_list)
        atomic_json(rd/'initial_validation.json',initial)
    from diagnostics import fixed_probe
    committed_epochs=len(history)
    try:
        for epoch in range(start_epoch,cfg['stage_epochs']):
            if shutil.disk_usage(ROOT).free<3*2**30:raise RuntimeError('Disk guard: less than3GiB free; preserve checkpoints and stop')
            epoch_started=datetime.now(timezone.utc).isoformat()
            atomic_json(rd/'progress.json',{'status':'training_epoch','epoch':epoch+1,'completed_epochs':epoch,'total_epochs':cfg['stage_epochs']})
            print(f'Epoch {epoch+1}/{cfg["stage_epochs"]} started',flush=True)
            lrs=schedule.set_epoch(epoch)
            # A fresh epoch-specific generator seeds order and workers; restart at epoch boundary reproduces it.
            generator=torch.Generator().manual_seed(int(a.seed)+epoch*100003)
            loader=torch.utils.data.DataLoader(train_set,batch_size=a.batch_size,shuffle=True,drop_last=True,
                num_workers=a.num_workers,collate_fn=collate_fn_crowd,worker_init_fn=seed_worker,generator=generator,persistent_workers=False)
            tr=train_epoch(model,criterion,optimizer,loader,device,epoch,freeze,a.clip_max_norm)
            pred_rel=f'predictions/epoch_{epoch+1:04d}.jsonl.gz'
            val=evaluate_loader(model,valloader,device,rd/pred_rel,protocol,valset.img_list)
            probe=fixed_probe(model,criterion,history[-1].get('fixed_probe') if history else None)
            candidate=val['metrics']['mae'];is_best=improved_mae(candidate,best)
            if is_best:best=candidate;best_epoch=epoch
            row={'run_id':cfg['run_id'],'stage_id':cfg['stage_id'],'seed':a.seed,'variant':cfg['variant'],
                 'started_utc':epoch_started,'completed_utc':datetime.now(timezone.utc).isoformat(),
                 'epoch_zero_based':epoch,'stage_epoch_completed':epoch+1,
                 'global_epoch_completed':cfg['global_epoch_offset']+epoch+1,'training':tr,'learning_rates':lrs,'fixed_probe':probe,
                 'weight_decay':{g['name']:g['weight_decay'] for g in optimizer.param_groups},
                 'backbone_frozen':epoch<freeze,'validation':val['metrics'],'validation_seconds':val['seconds'],
                 'prediction_file':pred_rel,'prediction_sha256':val['prediction_sha256'],
                 'raw_prediction_file':('predictions/'+val['raw_prediction_file']) if val['raw_prediction_file'] else None,
                 'raw_prediction_sha256':val['raw_prediction_sha256'],
                 'raw_prediction_bytes':val['raw_prediction_bytes'],
                 'best_mae':best,'best_epoch_zero_based':best_epoch,'is_new_best':is_best,'evaluation_protocol':protocol}
            history.append(row)
            payload=checkpoint_payload(model,criterion,optimizer,schedule,epoch,best,best_epoch,history,identity)
            commit_epoch(rd,payload,is_best)
            committed_epochs=len(history)
            print(json.dumps({'epoch':epoch+1,'mae':candidate,'best_mae':best,'train_s':tr['seconds'],'val_s':val['seconds'],'peak_gib':tr['gpu_peak_allocated_bytes']/2**30},ensure_ascii=False),flush=True)
            atomic_json(rd/'progress.json',{'status':'running','completed_epochs':epoch+1,'total_epochs':cfg['stage_epochs'],'best_mae':best})
        endpoint={}
        for label in ['best','last']:
            saved=torch.load(rd/(label+'.pth'),map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'],strict=True)
            ev=evaluate_loader(model,valloader,device,rd/f'endpoint/{label}.jsonl.gz',protocol,valset.img_list,save_raw=True)
            expected=history[saved['epoch']]['validation']
            if ev['metrics']!=expected:raise RuntimeError('Endpoint re-evaluation mismatch: '+label)
            endpoint[label]={'epoch_zero_based':saved['epoch'],**ev}
            del saved
        atomic_json(rd/'endpoint_metrics.json',endpoint)
        atomic_json(rd/'manifest.json',{'status':'complete','identity':identity,'completed_epochs':len(history),
                                       'best_sha256':sha256(rd/'best.pth'),'last_sha256':sha256(rd/'last.pth'),
                                       'best_mae':best,'best_epoch_zero_based':best_epoch})
        atomic_json(rd/'progress.json',{'status':'complete','completed_epochs':len(history)})
    except BaseException as exc:
        atomic_json(rd/'failure.json',{'error':type(exc).__name__+': '+str(exc),'last_fully_returned_commit_epochs':committed_epochs,'recovery_source':'last.pth authoritative; repair derived outputs on resume'})
        raise

def run_evaluation(cfg,checkpoint,split,output_dir):
    if not torch.cuda.is_available():raise RuntimeError('Real-data evaluation requires GPU')
    configure_runtime(cfg['args']['seed'])
    out=Path(output_dir).resolve()
    if not out.is_relative_to(ROOT/'05_结果/formal_single_seed_20261016'):raise ValueError('Evaluation output outside result root')
    if out.exists() and any(out.iterdir()):raise ValueError('Refusing to overwrite evaluation')
    data=read_data_manifest(cfg);protocol=protocol_from(cfg);identity=identity_for(cfg)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('identity')!=identity:raise ValueError('Checkpoint does not belong to this configuration')
    device=torch.device('cuda');a,model,criterion=build(cfg,device);model.load_state_dict(ck['model'],strict=True)
    ds=ManifestValidationDataset(data[split])
    loader=torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=evaluation_collate)
    result=evaluate_loader(model,loader,device,out/'predictions.jsonl.gz',protocol,ds.img_list)
    result.update(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),split=split,identity=identity,evaluation_protocol=protocol,environment=record_environment())
    atomic_json(out/'metrics.json',result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--mode',choices=['train','evaluate'],required=True);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--preflight-report');ap.add_argument('--checkpoint');ap.add_argument('--split',choices=['validation','evaluation'],default='evaluation');ap.add_argument('--output-dir')
    ns=ap.parse_args();cfg=json.loads(Path(ns.config).read_text())
    lockpath=ROOT/'05_结果/固定队列_20261014/queue.lock'
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ns.mode=='train':run_training(cfg,ns.resume,ns.preflight_report)
        else:
            if not ns.checkpoint or not ns.output_dir:ap.error('Evaluation needs --checkpoint and --output-dir')
            run_evaluation(cfg,ns.checkpoint,ns.split,ns.output_dir)
if __name__=='__main__':main()

baseline_train_epoch=train_epoch
from gradnorm_integration import train_epoch
