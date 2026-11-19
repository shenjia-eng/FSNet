"""FSNet training integration. Model and GradNorm Adam are strictly separate."""
import json, math, os, gzip, hashlib, time
from pathlib import Path
import torch
from gradnorm_core import GradNormBalancer, detached_copy, require

def state_digest(state):
    h=hashlib.sha256()
    def walk(v):
        if torch.is_tensor(v):
            x=v.detach().cpu().contiguous()
            h.update(b"tensor");h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b"dict")
            for k in sorted(v,key=str):walk(k);walk(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode());h.update(str(len(v)).encode())
            for x in v:walk(x)
        else:h.update(type(v).__name__.encode());h.update(repr(v).encode())
    walk(state);return h.hexdigest()

def validate_state_for(criterion,state):
    checker=GradNormBalancer("cpu",criterion.gradnorm.config)
    checker.load_state_dict(state)

def summary(b):
    state=b.state_dict()
    return dict(state_sha256=state_digest(state),weights_end=state["weights"].tolist(),
        effective_point_coefficient=b.config["base_point_coefficient"]*float(state["weights"][1]/state["weights"][0]),
        reference_losses=state["reference"].tolist(),reference_initialized=state["reference_initialized"],
        model_steps=b.model_steps,update_steps=b.update_steps,empty_steps=b.empty_steps,
        disabled_steps=b.disabled_steps,projection_steps=b.projection_steps,config=b.config,
        warmup_steps=b.warmup_steps,reference_count=b.reference_count,
        reference_sum=state["reference_sum"].tolist(),boundary_steps=b.boundary_steps)

def train_epoch(model,criterion,optimizer,loader,device,epoch,freeze_epochs,clip_norm):
    import audited_train as d
    from diagnostics import DiagnosticAccumulator,spectral_stats,first_step_update
    from training_core import configure_training_mode,sha256
    configure_training_mode(model,epoch<freeze_epochs);criterion.train()
    bal=criterion.gradnorm
    shared=dict(model.named_parameters()).get(bal.config["reference_parameter"])
    require(shared is not None and shared.requires_grad,"Invalid GradNorm reference")
    require(all(id(p)!=id(bal.weights) for group in optimizer.param_groups for p in group["params"]),"Task weight leaked into model optimizer")
    signature=[(n,id(p),tuple(p.shape)) for n,p in model.named_parameters()]
    diag=DiagnosticAccumulator();spectral_before=spectral_stats(model);initial_params=None
    totals={};samples_seen=steps=clipped_steps=0;grad_acc={};start=time.perf_counter()
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
    path=getattr(criterion,"gradnorm_batch_path",None);log=None;file=None
    if path:
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        for old in [path,path.with_name(path.name+".tmp")]:
            if old.exists():
                backup=path.parent/"uncommitted_attempts";backup.mkdir(exist_ok=True)
                os.replace(old,backup/(old.name+"."+str(time.time_ns())))
        file=path.with_name(path.name+".tmp").open("wb")
        log=gzip.GzipFile(filename="",fileobj=file,mode="wb",mtime=0)
    try:
        for samples,targets in loader:
            samples=samples.to(device);targets=[{k:v.to(device) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            raw=criterion(model(samples),targets)
            require(set(raw)=={"loss_ce","loss_point"} and criterion.weight_dict=={"loss_ce":1,"loss_point":0.0002},"Unexpected auxiliary loss or loss routing")
            total,gn_before=bal.begin(raw["loss_ce"],raw["loss_point"],shared,
                any(len(t["point"]) for t in targets),getattr(criterion,"gradnorm_enabled",True))
            diag.add(criterion,targets);total.backward()
            require(bal.weights.grad is None,"Model backward changed task-weight gradients")
            g=d.gradient_summary(model)
            if any(len(t["point"]) for t in targets):
                rg=g["regression."]
                require(rg["tensors_with_grad"]>0 and math.isfinite(rg["l2_norm"]),"Regression gradient missing/nonfinite")
            parameters=[p for group in optimizer.param_groups for p in group["params"]]
            norm=torch.nn.utils.clip_grad_norm_(parameters,clip_norm if clip_norm>0 else float("inf"),error_if_nonfinite=True)
            clipped=clip_norm>0 and float(norm)>clip_norm;clipped_steps+=int(clipped)
            if steps==0:initial_params={n:p.detach().clone() for n,p in model.named_parameters() if "ffparser" in n or n=="classification.output.weight"}
            optimizer.step()
            gn=bal.finish_after_model_step()
            if steps==0:first_update=first_step_update(model,initial_params)
            n=len(targets);samples_seen+=n;steps+=1
            values={"loss":float(total.detach()),"gradient_norm_before_clip":float(norm),
                    "raw/loss_ce":gn["raw_loss_cls"],"raw/loss_point":gn["raw_loss_point"],
                    "weighted/loss_ce":gn["weighted_losses"][0],"weighted/loss_point":gn["weighted_losses"][1],
                    "gradnorm/a":gn["weights_before"][0],"gradnorm/b":gn["weights_before"][1],
                    "gradnorm/effective_point_coefficient":gn["effective_point_coefficient_before"]}
            for k,v in values.items():totals[k]=totals.get(k,0.)+v*n
            for k,v in g.items():grad_acc[k]=grad_acc.get(k,0.)+v["l2_norm"]
            if log:
                ld=getattr(criterion,"_learning_diag",{})
                foreground_background={k:float(ld[k]) for k in ["positive_ce_numerator","negative_ce_numerator","ce_denominator"] if k in ld}
                record=dict(gn,classification_components=foreground_background,batch_index=steps-1,effective_crops=n,gradient_norm_before_clip=float(norm),clipped=bool(clipped))
                log.write((json.dumps(record,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode())
        require(steps>0,"Empty training epoch")
        require(signature==[(n,id(p),tuple(p.shape)) for n,p in model.named_parameters()],"Model parameter identity changed")
        if log:
            log.close();file.flush();os.fsync(file.fileno());file.close()
            os.replace(path.with_name(path.name+".tmp"),path)
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        if device.type=="cuda":torch.cuda.synchronize()
        return {"diagnostics":diag.result(),"spectral_before":spectral_before,"spectral_after":spectral_stats(model),
            "first_step_parameter_update":first_update,"averages":{k:v/samples_seen for k,v in totals.items()},
            "effective_crops_seen":samples_seen,"steps":steps,
            "gradient_l2_mean":{k:v/steps for k,v in grad_acc.items()},"clipped_steps":clipped_steps,
            "gradient_clip_fraction":clipped_steps/steps,"seconds":time.perf_counter()-start,
            "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
            "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved() if device.type=="cuda" else 0,
            "gradnorm":summary(bal),"gradnorm_batch_file":("gradnorm_batches/"+path.name) if path else None,
            "gradnorm_batch_sha256":sha256(path) if path else None}
    finally:
        if log and not log.closed:log.close()
        if file and not file.closed:file.close()
