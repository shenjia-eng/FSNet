"""One evaluator for learning curves AND final evaluation; pixel coordinates are half-open."""
from pathlib import Path
import gzip,json,math,os,time,zipfile,contextlib,hashlib,shutil
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from source.engine import compute_count_metrics,flatten_count_metrics_for_log
from training_core import rng_state,restore_rng,sha256
from util.misc import collate_fn_crowd

def evaluation_collate(batch):
    if len(batch)!=1:raise ValueError('Evaluation batch must contain one image')
    image,targets=batch[0]
    if image.ndim!=3 or len(targets)!=1:raise ValueError('Evaluation needs one full image, not patches')
    target=dict(targets[0]);target['original_size_hw']=torch.tensor(image.shape[-2:],dtype=torch.long)
    return collate_fn_crowd([(image,[target])])

def select_predictions(points,scores,width,height,threshold):
    points=np.asarray(points,dtype=np.float64).reshape(-1,2);scores=np.asarray(scores,dtype=np.float64).reshape(-1)
    if len(points)!=len(scores) or not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid or nonfinite model prediction')
    if width<=0 or height<=0 or not 0<=threshold<=1:raise ValueError('Invalid prediction filter')
    confident=scores>threshold
    inside=(points[:,0]>=0)&(points[:,0]<width)&(points[:,1]>=0)&(points[:,1]<height)
    keep=confident&inside
    return points[keep],scores[keep],{'queries':len(scores),'above_threshold':int(confident.sum()),
                                      'out_of_bounds_above_threshold':int((confident&~inside).sum())}

def match_counts(pred,gt,radius):
    pred=np.asarray(pred,dtype=np.float64).reshape(-1,2);gt=np.asarray(gt,dtype=np.float64).reshape(-1,2)
    if radius<=0 or not np.isfinite(pred).all() or not np.isfinite(gt).all():raise ValueError('Invalid point matching input')
    if len(pred)==0 or len(gt)==0:return 0,len(pred),len(gt)
    neighbors=cKDTree(gt).query_ball_point(pred,r=radius)
    indptr=np.zeros(len(pred)+1,dtype=np.int64)
    for i,ns in enumerate(neighbors):indptr[i+1]=indptr[i]+len(ns)
    indices=np.asarray([j for ns in neighbors for j in sorted(ns)],dtype=np.int64)
    graph=csr_matrix((np.ones(len(indices),dtype=np.int8),indices,indptr),shape=(len(pred),len(gt)))
    tp=int((maximum_bipartite_matching(graph,perm_type='column')>=0).sum())
    return tp,len(pred)-tp,len(gt)-tp

def summarize(rows):
    if not rows:raise ValueError('Empty evaluation set')
    count=flatten_count_metrics_for_log(compute_count_metrics([r['pred_count'] for r in rows],[r['gt_count'] for r in rows]))
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    count.update(tp=tp,fp=fp,fn=fn,
                 precision=tp/(tp+fp) if tp+fp else 0.,
                 recall=tp/(tp+fn) if tp+fn else 0.,
                 f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    gt_counts=np.asarray([r['gt_count'] for r in rows],dtype=np.float64)
    if len(gt_counts)<2 or np.var(gt_counts)==0:count['r2']=None
    # Store undefined R2/Pearson values as null.
    return {k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in count.items()}

@torch.no_grad()
def evaluate_loader(model,loader,device,output_file,protocol,image_paths=None,save_raw=False):
    if protocol['coordinate_frame']!='input_pixels' or protocol['bounds']!='half_open':
        raise ValueError('Unsupported coordinate protocol')
    if loader.batch_size!=1:raise ValueError('Evaluation requires batch_size=1 with explicit original-size metadata')
    if image_paths is None or len(image_paths)!=len(loader.dataset):raise ValueError('Explicit image paths required')
    if not isinstance(loader.sampler,torch.utils.data.SequentialSampler):raise ValueError('Evaluation must be sequential')
    old_rng=rng_state();modes=[(m,m.training) for m in model.modules()];model.eval()
    p=Path(output_file);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    raw_path=p.with_name(p.name.replace('.jsonl.gz','.all_queries.npz'))
    raw_tmp=raw_path.with_name(raw_path.name+'.tmp')
    rows=[];query_hashes=[];started=time.perf_counter()
    try:
        with gzip.open(tmp,'wt',encoding='utf-8') as f, (zipfile.ZipFile(raw_tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) if save_raw else contextlib.nullcontext()) as archive:
            for i,(samples,targets) in enumerate(loader):
                if not torch.is_tensor(samples) or samples.ndim!=4 or samples.shape[0]!=1:raise ValueError('Unexpected evaluation tensor')
                if shutil.disk_usage(p.parent).free<3*2**30+512*2**20:raise RuntimeError('Disk boundary during evaluation; preserve partial files')
                padded_h,padded_w=samples.shape[-2:]
                if 'original_size_hw' not in targets[0]:raise ValueError('Missing original image dimensions before padding')
                h,w=map(int,targets[0]['original_size_hw'].tolist())
                if h>padded_h or w>padded_w:raise ValueError('Invalid original dimensions')
                outputs=model(samples.to(device))
                scores=torch.softmax(outputs['pred_logits'].float(),dim=-1)[0,:,1].cpu().numpy()
                raw_points=outputs['pred_points'][0].float().cpu().numpy()
                points,kept_scores,filters=select_predictions(raw_points,scores,w,h,protocol['score_threshold'])
                # Preserve every float32 query, including low confidence and out-of-bounds.
                member=f'image_{i:06d}.npy'
                raw_array=np.column_stack((raw_points,scores)).astype(np.float32,copy=False)
                query_hashes.append({'image_index':i,'shape':list(raw_array.shape),'dtype':str(raw_array.dtype),'sha256':hashlib.sha256(raw_array.tobytes()).hexdigest()})
                if save_raw:
                    with archive.open(member,'w',force_zip64=True) as binary:
                        np.lib.format.write_array(binary,raw_array,allow_pickle=False)
                gt=targets[0]['point'].cpu().numpy().reshape(-1,2)
                if not np.isfinite(gt).all() or ((gt[:,0]<0)|(gt[:,0]>=w)|(gt[:,1]<0)|(gt[:,1]>=h)).any():
                    raise ValueError('Ground-truth point outside input pixel frame: '+str(image_paths[i]))
                tp,fp,fn=match_counts(points,gt,protocol['match_radius_px'])
                row={'image_path':str(image_paths[i]),'image_index':i,'width':w,'height':h,'padded_width':padded_w,'padded_height':padded_h,
                     'raw_archive':raw_path.name if save_raw else None,'raw_member':member if save_raw else None,
                     'foreground_probability_quantiles':np.quantile(scores,[0,.5,.9,.99,.999,1]).tolist(),
                     'foreground_probability_histogram':np.histogram(scores,bins=np.linspace(0,1,101))[0].tolist(),
                     'raw_columns':['x','y','foreground_probability'],
                     'gt_count':len(gt),'pred_count':len(points),'tp':tp,'fp':fp,'fn':fn,
                     'gt_points':gt.tolist(),'pred_points':points.tolist(),'pred_scores':kept_scores.tolist(),
                     'filter_counts':filters,'prediction_scope':'score_above_threshold_and_in_bounds','protocol':protocol}
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                rows.append({k:row[k] for k in ['gt_count','pred_count','tp','fp','fn']})
        for pending in ([raw_tmp] if save_raw else [])+[tmp]:
            with pending.open('rb') as durable:os.fsync(durable.fileno())
        if save_raw:os.replace(raw_tmp,raw_path)
        os.replace(tmp,p)
        directory=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        result=summarize(rows)
        return {'all_query_semantic_sha256':query_hashes,'metrics':result,'seconds':time.perf_counter()-started,'prediction_sha256':sha256(p),'images':len(rows),
                'raw_prediction_file':raw_path.name if save_raw else None,'raw_prediction_sha256':sha256(raw_path) if save_raw else None,
                'raw_prediction_bytes':raw_path.stat().st_size if save_raw else 0,
                'raw_prediction_scope':'all_queries_float32_xy_and_foreground_probability' if save_raw else 'selected_points_plus_probability_summary'}
    finally:
        for m,mode in modes:m.training=mode
        restore_rng(old_rng)
