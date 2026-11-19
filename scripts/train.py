"""Five-stage FSNet training for seed42."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
from runtime import (read_json, sha256, source_path, records_for, load_driver,
                     validation_dataset, validate_training_splits)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", required=True, help="Directory with the five stage JSON configs")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--imagenet", required=True, help="Official vgg16_bn-6c64b313.pth")
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-root")
    ap.add_argument("--seed", type=int, choices=[42], default=42)
    ap.add_argument("--resume", action="store_true", help="Resume this wrapper's CURRENT full state")
    ap.add_argument("--validate-only", action="store_true", help="Verify data/configs without building a model")
    ns = ap.parse_args()
    configs = [read_json(p) for p in sorted(Path(ns.stages).glob("F*.json"))]
    expected = ["F00_base150", "F01_fpn50", "F02_both50", "F03_gate_context50", "F04_detse50"]
    if [c["stage_id"] for c in configs] != expected or [c["stage_epochs"] for c in configs] != [150,50,50,50,50]:
        raise ValueError("Expected the five-stage150+50+50+50+50 recipe")
    if any(c["args"]["seed"] != ns.seed or c["variant"] != "fsnet" for c in configs):
        raise ValueError("Use the released seed42 FSNet configs")
    counts = validate_training_splits(read_json(ns.manifest), configs)
    imagenet = Path(ns.imagenet).resolve()
    if sha256(imagenet) != configs[0]["backbone_initialization"]["sha256"]:
        raise ValueError("ImageNet initialization checksum mismatch")
    records = {k: records_for(ns.manifest, k, ns.data_root) for k in ("train", "validation")}
    if ns.validate_only:
        print(json.dumps({"status":"validated", **counts, "stages":expected}))
        return
    out = Path(ns.output).resolve()
    if out.exists() and any(out.iterdir()) and not ns.resume:
        raise ValueError("Output exists; use a new directory or explicit --resume")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    source = source_path(ns.source_root)
    d = load_driver(source, out)
    import torch
    from training_core import (BudgetSchedule, build_optimizer, seed_worker, initialize_model,
                               checkpoint_payload, atomic_json, export_history)
    from fsnet_archive import FSNetArchive, restore_fsnet_full
    from gradnorm_core import GradNormBalancer, VERSION
    from models import backbone
    from metrics import evaluate_loader, evaluation_collate
    if not torch.cuda.is_available():
        raise ValueError("The published training recipe requires CUDA")
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    backbone.models.model_paths["vgg16_bn"] = str(imagenet)
    train_list, val_list = out / "train.list", out / "validation.list"
    for role, p in (("train", train_list), ("validation", val_list)):
        content = "".join(r["image"] + "\t" + r["annotation"] + "\n" for r in records[role])
        if p.exists() and p.read_text(encoding="utf-8") != content:
            raise ValueError("Existing data list changed: " + str(p))
        p.write_text(content, encoding="utf-8")
    os.environ["SHHA_TRAIN_LIST"] = str(train_list)
    os.environ["SHHA_TEST_LIST"] = str(val_list)
    previous = None
    for index, original in enumerate(configs):
        cfg = copy.deepcopy(original)
        stage = cfg["stage_id"]
        rd = out / stage
        rd.mkdir(exist_ok=True)
        cfg["args"]["data_root"] = str(Path(ns.data_root).resolve())
        cfg["args"]["data_roots"] = str(Path(ns.data_root).resolve())
        cfg["backbone_initialization"]["path"] = str(imagenet)
        identity = {"protocol":"fsnet-portable-seed42-v1", "gradnorm_protocol":VERSION,
                    "stage":stage, "original_config_sha256":__import__('hashlib').sha256(
                        json.dumps(original,sort_keys=True).encode()).hexdigest(),
                    "manifest_sha256":sha256(ns.manifest), "source_sha256":d.runtime_source_hash()}
        context = {"run_id":"portable_s42_"+stage, "stage_id":stage, "seed":42,
                   "variant":"fsnet", "global_epoch_offset":cfg["global_epoch_offset"],
                   "evaluation_protocol":cfg["evaluation"]}
        archive = FSNetArchive(rd, identity, context)
        state = archive.load(with_full=False) if (rd / "CURRENT").exists() else None
        if state and state["manifest"]["epoch"] == cfg["stage_epochs"]:
            previous = archive
            print(stage + " already complete", flush=True)
            continue
        d.configure_runtime(42)
        a, model, criterion = d.build(cfg, torch.device("cuda"))
        optimizer = build_optimizer(model, criterion, a)
        schedule = BudgetSchedule(optimizer, cfg["stage_epochs"], **cfg["schedule"])
        train, _ = d.legacy.build_dataset(args=a)(d.legacy._resolve_data_roots(a))
        if train.img_map != {r["image"]:r["annotation"] for r in records["train"]}:
            raise ValueError("Loader and manifest training records differ")
        val = validation_dataset(records["validation"])
        loader = torch.utils.data.DataLoader(val,batch_size=1,shuffle=False,num_workers=0,collate_fn=evaluation_collate)
        history = []
        if state:
            state = archive.load(with_full=True)
            restore_fsnet_full(state["full"],model,criterion,optimizer,schedule,identity,__import__("training_core"))
            history = state["detailed_history"]
        elif index:
            parent_state = previous.load(with_full=False)
            chosen = parent_state["manifest"]["roles"]["joint"]
            reference = parent_state["manifest"]["models"][str(chosen)]
            parent = previous.tensor_load(reference)
            if parent["gradnorm"]["config"] != cfg["gradnorm"]:
                raise ValueError("Parent GradNorm configuration changed")
            checker = GradNormBalancer("cpu",cfg["gradnorm"])
            checker.load_state_dict(parent["gradnorm"])
            initialize_model(model,previous.root/reference["path"],
                             cfg["initialization"]["allowed_missing_prefixes"],
                             cfg["initialization"]["allowed_unexpected_prefixes"])
            criterion.gradnorm = GradNormBalancer("cuda",cfg["gradnorm"],parent["gradnorm"]["weights"])
        atomic_json(rd/"portable_config.json",cfg)
        for epoch in range(len(history),cfg["stage_epochs"]):
            started = now()
            rates = schedule.set_epoch(epoch)
            generator = torch.Generator().manual_seed(42 + epoch * 100003)
            train_loader = torch.utils.data.DataLoader(train,batch_size=a.batch_size,shuffle=True,
                drop_last=True,num_workers=a.num_workers,collate_fn=d.collate_fn_crowd,
                worker_init_fn=seed_worker,generator=generator,persistent_workers=False)
            criterion.gradnorm_batch_path = rd/f"gradnorm_batches/epoch_{epoch+1:04d}.jsonl.gz"
            training = d.train_epoch(model,criterion,optimizer,train_loader,torch.device("cuda"),
                                     epoch,cfg["freeze_backbone_epochs"],a.clip_max_norm)
            if training["steps"] != 259 or training["effective_crops_seen"] != 8288:
                raise ValueError("Actual epoch budget differs from the published recipe")
            prediction = f"predictions/epoch_{epoch+1:04d}.jsonl.gz"
            ev = evaluate_loader(model,loader,torch.device("cuda"),rd/prediction,
                                 cfg["evaluation"],val.img_list,save_raw=False)
            best = min((r["validation"]["mae"] for r in history),default=float("inf"))
            improved = ev["metrics"]["mae"] < best
            best_epoch = epoch if improved else min(range(len(history)),key=lambda i:history[i]["validation"]["mae"])
            best = min(best,ev["metrics"]["mae"])
            row = {**{k:context[k] for k in ("run_id","stage_id","seed","variant")},
                "started_utc":started,"completed_utc":now(),"epoch_zero_based":epoch,
                "stage_epoch_completed":epoch+1,"global_epoch_completed":cfg["global_epoch_offset"]+epoch+1,
                "training":training,"learning_rates":rates,"fixed_probe":{},
                "weight_decay":{g["name"]:g["weight_decay"] for g in optimizer.param_groups},
                "backbone_frozen":epoch<cfg["freeze_backbone_epochs"],"validation":ev["metrics"],
                "validation_seconds":ev["seconds"],"prediction_file":prediction,
                "prediction_sha256":ev["prediction_sha256"],"raw_prediction_file":None,
                "raw_prediction_sha256":None,"raw_prediction_bytes":0,"best_mae":best,
                "best_epoch_zero_based":best_epoch,"is_new_best":improved,
                "evaluation_protocol":cfg["evaluation"],"all_query_semantic_sha256":ev["all_query_semantic_sha256"]}
            history.append(row)
            full = checkpoint_payload(model,criterion,optimizer,schedule,epoch,best,best_epoch,history,identity)
            state = archive.commit_full(full,state["ref"]["sha256"] if state else None)
            archive.gc(state["ref"]["sha256"])
            export_history(rd/"epochs.jsonl",history)
            print(json.dumps({"stage":stage,"epoch":epoch+1,"mae":best,
                              "joint_epoch":state["manifest"]["roles"]["joint"]}),flush=True)
        previous = archive
        del model, criterion, optimizer, schedule, train, val, loader
        torch.cuda.empty_cache()
    state = previous.load(with_full=False)
    selected = previous.resolve_model("joint")
    e = state["manifest"]["roles"]["joint"]
    from training_core import atomic_torch
    atomic_torch(out/"fsnet_seed42_joint.pth",{"model":selected["model"],"seed":42,
                 "stage_epoch":e,"global_epoch":300+e})
    atomic_json(out/"complete.json",{"status":"training_complete", "selected_stage_epoch":e,
                 "selected_global_epoch":300+e})


if __name__ == "__main__":
    main()
