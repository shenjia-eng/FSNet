"""Learning diagnostics for fixed training crops."""
from pathlib import Path
from types import MethodType
import math
import torch
import torch.nn.functional as F
import numpy as np
from training_core import rng_state,restore_rng
from metrics import select_predictions,match_counts,summarize
PROBE=Path('fixed_crop_probe.pth')
def attach_diagnostics(c):
    original=c.loss_labels
    def labels(self,outputs,targets,indices,num_points):
        result=original(outputs,targets,indices,num_points)
        with torch.no_grad():
            logits=outputs['pred_logits'];p=logits.softmax(-1)[...,1]
            target=torch.zeros(p.shape,dtype=torch.long,device=p.device)
            xy=[];matched=[];pairs=[]
            for i,(idx,j) in enumerate(indices):
                target[i,idx]=1;matched.append(p[i,idx])
                if len(idx):xy.append((outputs['pred_points'][i,idx]-targets[i]['point'][j]).norm(dim=-1))
                pairs.append([[int(g),int(k)] for g,k in zip(j.cpu(),idx.cpu())])
            ce=F.cross_entropy(logits.reshape(-1,2),target.reshape(-1),weight=self.empty_weight,reduction='none').reshape(p.shape)
            positive=target.bool();den=self.empty_weight[target].sum()
            pos_num=float(ce[positive].sum());neg_num=float(ce[~positive].sum());den=float(den)
            assert math.isclose((pos_num+neg_num)/den,float(result['loss_ce']),rel_tol=2e-5,abs_tol=1e-7)
            self._learning_diag={'positive_ce_numerator':pos_num,'negative_ce_numerator':neg_num,'ce_denominator':den,
                'matched_probabilities':torch.cat(matched).cpu().numpy(),
                'background_probability_quantiles':torch.quantile(p[~positive],torch.tensor([.5,.9,.99,1.],device=p.device)).cpu().numpy() if (~positive).any() else np.zeros(4),
                'matched_distances':torch.cat(xy).cpu().numpy() if xy else np.empty(0),
                'above_threshold':int((p>.5).sum()),'queries':p.numel()}
        return result
    c.loss_labels=MethodType(labels,c)
class DiagnosticAccumulator:
    def __init__(self):self.rows=[];self.points=0;self.empty=0;self.crops=0
    def add(self,c,targets):
        self.rows.append(c._learning_diag);counts=[len(t['point']) for t in targets]
        self.points+=sum(counts);self.empty+=sum(n==0 for n in counts);self.crops+=len(counts)
    def result(self):
        def quant(key):
            z=np.concatenate([x[key] for x in self.rows]);return np.quantile(z,[0,.1,.5,.9,.99,1]).tolist() if len(z) else None
        den=sum(x['ce_denominator'] for x in self.rows)
        return {'target_points_seen':self.points,'empty_crops':self.empty,'empty_crop_fraction':self.empty/self.crops,
            'queries_seen':sum(x['queries'] for x in self.rows),'above_threshold_train_mode':sum(x['above_threshold'] for x in self.rows),
            'positive_ce_numerator':sum(x['positive_ce_numerator'] for x in self.rows),
            'negative_ce_numerator':sum(x['negative_ce_numerator'] for x in self.rows),'ce_denominator':den,
            'matched_probability_quantiles_0_10_50_90_99_100':quant('matched_probabilities'),
            'matched_distance_quantiles_0_10_50_90_99_100':quant('matched_distances'),
            'mean_batch_background_quantiles_50_90_99_100':np.mean([x['background_probability_quantiles'] for x in self.rows],axis=0).tolist()}
def spectral_stats(m):
    return {n:float(p.detach().norm()) for n,p in m.named_parameters() if 'ffparser' in n}
def first_step_update(m,before):
    return {n:{'before_norm':float(before[n].norm()),'update_norm':float((p.detach()-before[n]).norm())} for n,p in m.named_parameters() if n in before}
@torch.no_grad()
def fixed_probe(m,c,previous=None):
    old_rng=rng_state();modes=[(module,module.training) for module in m.modules()]
    try:
        m.eval();b=torch.load(PROBE,map_location='cpu',weights_only=False,mmap=True)
        x=b['x'].cuda();targets=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in t.items()} for t in b['targets']]
        out=m(x);indices=c.matcher(out,targets);prob=out['pred_logits'].softmax(-1)[...,1];rows=[];pairs=[];matched=[]
        for i,t in enumerate(targets):
            idx,j=indices[i];pairs.append({str(int(g)):int(k) for g,k in zip(j.cpu(),idx.cpu())});matched.extend(prob[i,idx].cpu().tolist())
            pts,_,_=select_predictions(out['pred_points'][i].cpu().numpy(),prob[i].cpu().numpy(),x.shape[-1],x.shape[-2],.5)
            gt=t['point'].cpu().numpy();tp,fp,fn=match_counts(pts,gt,8)
            rows.append({'gt_count':len(gt),'pred_count':len(pts),'tp':tp,'fp':fp,'fn':fn})
        old=previous.get('target_to_query') if previous else None
        changed=sum(v!=old[i].get(k) for i,d in enumerate(pairs) for k,v in d.items()) if old is not None else None
        n=sum(len(z) for z in pairs)
        return {'scope':'fixed20 training crops','metrics':summarize(rows),
            'matched_probability_quantiles':np.quantile(matched,[0,.1,.5,.9,1]).tolist(),'target_to_query':pairs,
            'matched_query_change_fraction_vs_previous':changed/n if changed is not None and n else None}
    finally:
        for module,mode in modes:module.training=mode
        restore_rng(old_rng)
