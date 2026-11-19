"""Composite checkpoint selection."""
import math
def valid_history(history):
    if not history:raise ValueError('No trained epochs')
    epochs=[x['epoch'] for x in history]
    if epochs!=list(range(1,len(history)+1)):raise ValueError('Continuous unique trained epochs required')
    for x in history:
        if not all(isinstance(x[k],(int,float)) and math.isfinite(x[k]) for k in ('mae','rmse','f1')):raise ValueError('Nonfinite metrics')
        if x['mae']<0 or x['rmse']<0 or not 0<=x['f1']<=1:raise ValueError('Metric domain')
def strict_mae(history):
    valid_history(history)
    return min(history,key=lambda x:(x['mae'],x['epoch']))
def f1_best(history):
    valid_history(history)
    return min(history,key=lambda x:(-x['f1'],x['mae'],x['rmse'],x['epoch']))
def select(history,candidates=None):
    ref=strict_mae(history)
    candidates=history if candidates is None else candidates
    eligible=[x for x in candidates if x['mae']<=ref['mae']*1.05 and x['rmse']<=ref['rmse']*1.05]
    if not eligible:raise ValueError('Pool missing a guaranteed eligible reference')
    return min(eligible,key=lambda x:(-x['f1'],x['mae'],x['rmse'],x['epoch']))
def dominates(a,b):
    weak=a['mae']<=b['mae'] and a['rmse']<=b['rmse'] and a['f1']>=b['f1']
    strict=a['mae']<b['mae'] or a['rmse']<b['rmse'] or a['f1']>b['f1']
    same=all(a[k]==b[k] for k in ('mae','rmse','f1'))
    return weak and (strict or (same and a['epoch']<b['epoch']))
def pool(history):
    valid_history(history)
    frontier=[x for x in history if not any(dominates(y,x) for y in history)]
    keep={x['epoch']:x for x in frontier}
    ref=strict_mae(history);fb=f1_best(history)
    keep[ref['epoch']]=ref
    keep[fb['epoch']]=fb
    result=[keep[k] for k in sorted(keep)]
    if select(history,result)['epoch']!=select(history)['epoch']:raise AssertionError('Pool selection differs')
    return result
