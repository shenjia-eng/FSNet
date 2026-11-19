"""FSNet training-history adapter."""
import copy,math
from archive import Archive,fail,canonical,sha,freeze
import torch

REQUIRED={'run_id','stage_id','seed','variant','started_utc','completed_utc','epoch_zero_based',
'stage_epoch_completed','global_epoch_completed','training','learning_rates','fixed_probe','weight_decay',
'backbone_frozen','validation','validation_seconds','prediction_file','prediction_sha256',
'raw_prediction_file','raw_prediction_sha256','raw_prediction_bytes','best_mae',
'best_epoch_zero_based','is_new_best','evaluation_protocol'}

class FSNetArchive(Archive):
 def __init__(self,root,identity,context):
  super().__init__(root,identity)
  self.context=copy.deepcopy(context)
  if set(context)!= {'run_id','stage_id','seed','variant','global_epoch_offset','evaluation_protocol'}:fail('Explicit stage context required')
 def project_history(self,history):
  if not history:fail('Empty detailed history')
  rows=[];best=math.inf;best_epoch=-1
  for epoch,row in enumerate(history,1):
   if not REQUIRED.issubset(row):fail('Incomplete detailed history')
   for k in ['run_id','stage_id','seed','variant','evaluation_protocol']:
    if row[k]!=self.context[k]:fail('Detailed stage identity '+k)
   if any(type(row[k]) is not int for k in ['epoch_zero_based','stage_epoch_completed','global_epoch_completed','seed']):fail('Detailed integer metadata')
   if row['epoch_zero_based']!=epoch-1 or row['stage_epoch_completed']!=epoch or row['global_epoch_completed']!=self.context['global_epoch_offset']+epoch:fail('Detailed epoch mismatch')
   if any(not isinstance(row[k],dict) for k in ['training','learning_rates','fixed_probe','weight_decay','validation']):fail('Detailed records missing')
   met=row['validation'];improved=met['mae']<best
   if improved:best=met['mae'];best_epoch=epoch-1
   if type(row['is_new_best']) is not bool or row['is_new_best']!=improved or row['best_mae']!=best or row['best_epoch_zero_based']!=best_epoch:fail('Detailed strict MAE metadata')
   rows.append({'epoch':epoch,'mae':met['mae'],'rmse':met['rmse'],'f1':met['f1'],
                'prediction':{'path':row['prediction_file'],'sha256':row['prediction_sha256']}})
  # Fail nonfinite/unserializable extension records without dropping any fields.
  canonical(history)
  if self.identity.get('gradnorm_protocol'):
   from gradnorm_core import GradNormBalancer
   for row in history:
    tr=row['training'];gn=tr['gradnorm']
    if gn['model_steps']!=row['stage_epoch_completed']*259:fail('GradNorm model update count')
    if gn['model_steps']!=gn['update_steps']+gn['empty_steps']+gn['disabled_steps']+gn['warmup_steps']:fail('GradNorm counter identity')
    if gn['disabled_steps']!=0:fail('GradNorm dynamic weighting is disabled')
    checker=GradNormBalancer(config=gn['config'])
    checker.check_bounded_weights(torch.tensor(gn['weights_end'],dtype=torch.float32))
    if gn['warmup_steps']!=min(gn['model_steps'],gn['config']['warmup_steps']):fail('Warmup count')
    if gn['boundary_steps']>gn['update_steps']:fail('Boundary count')
    self.prediction({'path':tr['gradnorm_batch_file'],'sha256':tr['gradnorm_batch_sha256']})
  from archive import strict_rows
  strict_rows(rows)
  return rows
 def commit_full(self,full,previous_sha,hook=lambda _:None):
  return self.commit(self.project_history(full['history'])[-1],full,previous_sha,hook)

def validate_optimizer_objects(full,model,criterion,optimizer,schedule,identity,core):
 if full['identity']!=identity:fail('Actual FSNet identity')
 if hasattr(criterion,'gradnorm'):
  if 'gradnorm' not in full:fail('Missing GradNorm full state')
  from gradnorm_integration import validate_state_for
  validate_state_for(criterion,full['gradnorm'])
 if full['optimizer_signature']!=core.group_signature(optimizer):fail('Actual optimizer signature')
 for actual,saved in [(model.state_dict(),full['model']),(criterion.state_dict(),full['criterion'])]:
  if set(actual)!=set(saved):fail('Actual model/criterion keys')
  for name,t in actual.items():
   v=saved[name]
   if not torch.is_tensor(v) or v.shape!=t.shape or v.dtype!=t.dtype:fail('Actual tensor shape/dtype '+name)
   if (v.is_floating_point() or v.is_complex()) and not torch.isfinite(v).all().item():fail('Actual nonfinite tensor '+name)
 state=full['optimizer'];groups=state['param_groups']
 if len(groups)!=len(optimizer.param_groups):fail('Actual optimizer group count')
 all_ids=[]
 for sg,ag in zip(groups,optimizer.param_groups):
  for k in ['name','param_names','base_lr','weight_decay','betas','eps','amsgrad']:
   if sg[k]!=ag[k]:fail('Actual Adam group metadata '+k)
  if len(sg['params'])!=len(ag['params']):fail('Actual Adam parameter count')
  for pid,p in zip(sg['params'],ag['params']):
   all_ids.append(pid)
   if pid not in state['state']:continue
   st=state['state'][pid]
   for k in ['exp_avg','exp_avg_sq']:
    if k not in st or st[k].shape!=p.shape or st[k].dtype!=p.dtype or not torch.isfinite(st[k]).all().item():fail('Actual Adam moment shape/dtype/finite')
   if not torch.is_tensor(st['step']) or st['step'].numel()!=1 or not torch.isfinite(st['step']).all().item() or st['step'].item()<0:fail('Actual Adam step')
 if len(all_ids)!=len(set(all_ids)) or not set(state['state']).issubset(all_ids):fail('Actual Adam ids')
 sc=full['scheduler']
 if (sc['total_epochs'],sc['warmup_epochs'],sc['min_factor'])!=(schedule.total,schedule.warmup,schedule.minimum):fail('Actual scheduler protocol')
 if sc['epoch']!=full['epoch']:fail('Actual scheduler epoch')

def restore_fsnet_full(full,model,criterion,optimizer,schedule,identity,core,restore_random=True):
 validate_optimizer_objects(full,model,criterion,optimizer,schedule,identity,core)
 model.load_state_dict(full['model'],strict=True);criterion.load_state_dict(full['criterion'],strict=True)
 optimizer.load_state_dict(full['optimizer']);schedule.load_state_dict(full['scheduler'])
 if hasattr(criterion,'gradnorm'):criterion.gradnorm.load_state_dict(full['gradnorm'])
 if restore_random:core.restore_rng(full['rng'])
 return full
