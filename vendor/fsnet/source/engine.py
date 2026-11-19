# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable, Optional

import torch

import util.misc as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)

# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    log_batch_interval: int = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if log_batch_interval > 0 and (i + 1) % log_batch_interval == 0:
            print(
                "[epoch %d] batch %d  loss=%.4f" % (epoch, i + 1, loss_value),
                flush=True,
            )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

DEFAULT_COUNT_TOLS = (1, 3, 5, 10)
DEFAULT_RATIO_TOLS = (0.05, 0.10, 0.15, 0.20)
DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def _empty_count_metrics(count_tol=5, ratio_tol=0.10):
    nan = float("nan")
    return {
        "n": 0,
        "mae": nan,
        "me": nan,
        "mse": nan,
        "rmse": nan,
        "median_ae": nan,
        "p90_ae": nan,
        "p95_ae": nan,
        "max_ae": nan,
        "over_count_rate": nan,
        "under_count_rate": nan,
        "exact_count_rate": nan,
        "pearson_r": nan,
        "accuracy": nan,
        "r2": nan,
        "acc_at_count_tol": nan,
        "acc_at_ratio_tol": nan,
        "acc_at_count_tols": {k: nan for k in DEFAULT_COUNT_TOLS},
        "acc_at_ratio_tols": {k: nan for k in DEFAULT_RATIO_TOLS},
        "count_tol": count_tol,
        "ratio_tol": ratio_tol,
    }


def compute_count_metrics(
    pred_counts,
    gt_counts,
    count_tol=5,
    ratio_tol=0.10,
    count_tols=None,
    ratio_tols=None,
):
    """Per-image count metrics: MAE/ME/RMSE/MSE, robust errors, multi-tier accuracies."""
    preds = np.asarray(pred_counts, dtype=np.float64)
    gts = np.asarray(gt_counts, dtype=np.float64)
    if preds.size == 0:
        return _empty_count_metrics(count_tol=count_tol, ratio_tol=ratio_tol)

    count_tols = tuple(count_tols or DEFAULT_COUNT_TOLS)
    ratio_tols = tuple(ratio_tols or DEFAULT_RATIO_TOLS)
    signed = preds - gts
    abs_err = np.abs(signed)
    rel_err = abs_err / np.maximum(gts, 1.0)
    sq_err = signed ** 2

    pearson_r = float("nan")
    if preds.size >= 2 and np.std(preds) > 0 and np.std(gts) > 0:
        pearson_r = float(np.corrcoef(preds, gts)[0, 1])

    # Formula (13): mean per-image relative accuracy; empty images (g=0) use max(g,1)=1.
    per_image_acc = 1.0 - abs_err / np.maximum(gts, 1.0)
    accuracy = float(np.mean(per_image_acc))

    # Formula (15): coefficient of determination (R²).
    r2 = float("nan")
    if preds.size >= 2:
        ss_res = float(np.sum(sq_err))
        ss_tot = float(np.sum((gts - np.mean(gts)) ** 2))
        if ss_tot == 0:
            r2 = 1.0 if ss_res == 0 else float("nan")
        else:
            r2 = float(1.0 - ss_res / ss_tot)

    return {
        "n": int(preds.size),
        "mae": float(np.mean(abs_err)),
        "me": float(np.mean(signed)),
        "mse": float(np.mean(sq_err)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "median_ae": float(np.median(abs_err)),
        "p90_ae": float(np.percentile(abs_err, 90)),
        "p95_ae": float(np.percentile(abs_err, 95)),
        "max_ae": float(np.max(abs_err)),
        "over_count_rate": float(np.mean(signed > 0)),
        "under_count_rate": float(np.mean(signed < 0)),
        "exact_count_rate": float(np.mean(signed == 0)),
        "pearson_r": pearson_r,
        "accuracy": accuracy,
        "r2": r2,
        "acc_at_count_tol": float(np.mean(abs_err <= count_tol)),
        "acc_at_ratio_tol": float(np.mean(rel_err <= ratio_tol)),
        "acc_at_count_tols": {k: float(np.mean(abs_err <= k)) for k in count_tols},
        "acc_at_ratio_tols": {k: float(np.mean(rel_err <= k)) for k in ratio_tols},
        "count_tol": count_tol,
        "ratio_tol": ratio_tol,
    }


def format_count_metrics(metrics, threshold=None):
    """Human-readable multi-line summary for eval scripts."""
    lines = []
    if threshold is not None:
        lines.append(f"threshold:               {float(threshold):.2f}")
    lines.extend([
        f"n:                       {metrics['n']}",
        f"MAE  (平均绝对误差):     {metrics['mae']:.4f}",
        f"ME   (有符号平均偏差):   {metrics['me']:+.4f}  (+多估 / -少估)",
        f"MSE  (均方误差):         {metrics['mse']:.4f}",
        f"RMSE (均方根误差):       {metrics['rmse']:.4f}",
        f"Median AE:               {metrics['median_ae']:.4f}",
        f"P90 AE:                  {metrics['p90_ae']:.4f}",
        f"P95 AE:                  {metrics['p95_ae']:.4f}",
        f"Max AE:                  {metrics['max_ae']:.4f}",
        f"Pearson r:               {metrics['pearson_r']:.4f}",
        f"Accuracy (eq.13):        {metrics['accuracy']:.4f}",
        f"R^2 (eq.15):             {metrics['r2']:.4f}",
        f"Over-count rate:         {metrics['over_count_rate'] * 100:.2f}%",
        f"Under-count rate:        {metrics['under_count_rate'] * 100:.2f}%",
        f"Exact-count rate:        {metrics['exact_count_rate'] * 100:.2f}%",
        f"Acc@±{metrics['count_tol']}棵:            {metrics['acc_at_count_tol'] * 100:.2f}%",
        f"Acc@±{metrics['ratio_tol'] * 100:.0f}%:           {metrics['acc_at_ratio_tol'] * 100:.2f}%",
    ])
    for k, v in sorted(metrics.get("acc_at_count_tols", {}).items()):
        if k == metrics.get("count_tol"):
            continue
        lines.append(f"Acc@±{k}棵:               {v * 100:.2f}%")
    for k, v in sorted(metrics.get("acc_at_ratio_tols", {}).items()):
        if abs(k - float(metrics.get("ratio_tol", -1))) < 1e-9:
            continue
        lines.append(f"Acc@±{k * 100:.0f}%:              {v * 100:.2f}%")
    return "\n".join(lines)


def best_threshold_by_mae(threshold_metrics):
    """Pick threshold row with lowest MAE from evaluate_crowd_count_metrics_at_thresholds()."""
    valid = [m for m in threshold_metrics if m.get("n", 0) > 0 and math.isfinite(m.get("mae", float("nan")))]
    if not valid:
        return None
    return min(valid, key=lambda m: m["mae"])


# --- Training log / summary helpers (validation = test.list) ---

TRAINING_SCALAR_METRICS = (
    "mae", "me", "mse", "rmse", "median_ae", "p90_ae", "p95_ae", "max_ae",
    "over_count_rate", "under_count_rate", "exact_count_rate", "pearson_r",
    "accuracy", "r2", "acc_at_count_tol", "acc_at_ratio_tol",
)
TRAINING_BEST_MIN = ("mae", "mse", "rmse", "median_ae", "p90_ae", "p95_ae", "max_ae")
TRAINING_BEST_MAX = (
    "pearson_r", "accuracy", "r2", "exact_count_rate", "acc_at_count_tol", "acc_at_ratio_tol",
)
TRAINING_BEST_MIN_ABS = ("me",)


def flatten_count_metrics_for_log(metrics: dict) -> dict:
    """JSON-serializable flat dict for run_log eval_json lines."""
    out = {}
    for k, v in metrics.items():
        if k in ("acc_at_count_tols", "acc_at_ratio_tols"):
            prefix = "acc_count" if k == "acc_at_count_tols" else "acc_ratio"
            for tk, tv in sorted(v.items()):
                key = f"{prefix}_{int(tk)}" if prefix == "acc_count" else f"{prefix}_{tk:g}"
                out[key] = float(tv)
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v) if k != "n" else int(v)
        elif k in ("count_tol", "ratio_tol", "n"):
            out[k] = v
    return out


def serialize_eval_json(metrics: dict, epoch: int, eval_time: float) -> str:
    import json
    payload = flatten_count_metrics_for_log(metrics)
    payload["epoch"] = int(epoch)
    payload["eval_time"] = float(eval_time)
    return "eval_json:" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_eval_json_line(line: str) -> Optional[dict]:
    import json
    line = line.strip()
    if not line.startswith("eval_json:"):
        return None
    try:
        return json.loads(line[len("eval_json:"):])
    except json.JSONDecodeError:
        return None


def _tb_tag_for_metric(key: str) -> str:
    if key.startswith("acc_count_"):
        return f"metric/acc_at_count_tol_{key.split('_', 2)[2]}"
    if key.startswith("acc_ratio_"):
        return f"metric/acc_at_ratio_tol_{key.split('_', 2)[2]}"
    return f"metric/{key}"


def log_validation_metrics(writer, log_file, step: int, epoch: int, metrics: dict, eval_time: float) -> None:
    """Write full validation metrics to run_log (eval_json) and TensorBoard."""
    log_file.write(serialize_eval_json(metrics, epoch, eval_time) + "\n")
    flat = flatten_count_metrics_for_log(metrics)
    for key, val in flat.items():
        if key in ("epoch", "eval_time", "n", "count_tol", "ratio_tol"):
            continue
        if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            continue
        tag = _tb_tag_for_metric(key)
        log_file.write(f"{tag}@{step}: {float(val)}\n")
        if writer is not None:
            writer.add_scalar(tag, float(val), step)


def _epoch_approx(eval_index: int, eval_freq: int, epoch: Optional[int] = None) -> int:
    if epoch is not None:
        return int(epoch)
    return (eval_index + 1) * eval_freq


def _pick_best_eval(evals: list[dict], key: str, mode: str):
    finite = [
        (i, e) for i, e in enumerate(evals)
        if key in e and isinstance(e[key], (int, float)) and math.isfinite(float(e[key]))
    ]
    if not finite:
        return None
    if mode == "min":
        idx, row = min(finite, key=lambda x: float(x[1][key]))
    elif mode == "max":
        idx, row = max(finite, key=lambda x: float(x[1][key]))
    elif mode == "min_abs":
        idx, row = min(finite, key=lambda x: abs(float(x[1][key])))
    else:
        return None
    return idx, row


def summarize_best_during_training(evals: list[dict], eval_freq: int) -> dict:
    """Build best/last validation metric summary from parsed eval records."""
    if not evals:
        return {}
    last = evals[-1]
    out: dict = {
        "metric_split": "validation (test.list)",
        "n_evals": len(evals),
        "last_eval": {},
        "best_during_training": {},
    }
    for key in TRAINING_SCALAR_METRICS:
        if key in last:
            out["last_eval"][key] = last[key]
    for key in sorted(last.keys()):
        if key.startswith("acc_count_") or key.startswith("acc_ratio_"):
            out["last_eval"][key] = last[key]

    specs = [(k, "min") for k in TRAINING_BEST_MIN]
    specs += [(k, "max") for k in TRAINING_BEST_MAX]
    specs += [(k, "min_abs") for k in TRAINING_BEST_MIN_ABS]
    for key, mode in specs:
        picked = _pick_best_eval(evals, key, mode)
        if picked is None:
            continue
        idx, row = picked
        epoch = row.get("epoch")
        entry = {
            "value": float(row[key]),
            "epoch_approx": _epoch_approx(idx, eval_freq, epoch),
            "criterion": mode,
        }
        out["best_during_training"][key] = entry
        if key == "mae":
            out["best_mae_during_training"] = entry["value"]
            out["best_mae_epoch_approx"] = entry["epoch_approx"]
        if key == "acc_at_count_tol":
            out["best_acc_at_count_tol_during_training"] = entry["value"]
            out["best_acc_at_count_tol_epoch_approx"] = entry["epoch_approx"]
        if key == "acc_at_ratio_tol":
            out["best_acc_at_ratio_tol_during_training"] = entry["value"]
            out["best_acc_at_ratio_tol_epoch_approx"] = entry["epoch_approx"]

    for key in sorted({k for e in evals for k in e if k.startswith("acc_count_") or k.startswith("acc_ratio_")}):
        picked = _pick_best_eval(evals, key, "max")
        if picked is None:
            continue
        idx, row = picked
        epoch = row.get("epoch")
        out["best_during_training"][key] = {
            "value": float(row[key]),
            "epoch_approx": _epoch_approx(idx, eval_freq, epoch),
            "criterion": "max",
        }

    best_mae = _pick_best_eval(evals, "mae", "min")
    if best_mae is not None:
        _, row = best_mae
        for bias_key in ("over_count_rate", "under_count_rate"):
            if bias_key in row:
                out["best_during_training"][bias_key] = {
                    "value": float(row[bias_key]),
                    "epoch_approx": _epoch_approx(best_mae[0], eval_freq, row.get("epoch")),
                    "criterion": "at_best_mae",
                }

    if "mse" in last:
        out["last_eval_rmse"] = last["mse"]
    if "rmse" in last:
        out["last_eval_rmse"] = last["rmse"]
    if "mae" in last:
        out["last_eval_mae"] = last["mae"]
    if last.get("count_tol") is not None:
        out["count_tol"] = last.get("count_tol", 5)
    if last.get("ratio_tol") is not None:
        out["ratio_tol"] = last.get("ratio_tol", 0.10)
    return out


@torch.no_grad()
def _collect_count_scores(model, data_loader, device, vis_dir=None, threshold=0.5):
    """Run model once; return per-image foreground scores and gt counts."""
    model.eval()
    score_batches = []
    gt_counts = []
    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        gt_cnt = int(targets[0]['point'].shape[0])
        score_batches.append(outputs_scores.detach().cpu())
        gt_counts.append(gt_cnt)
        if vis_dir is not None:
            mask = outputs_scores > threshold
            pts = outputs_points[mask]
            points = pts.detach().cpu().numpy().tolist()
            vis(samples, targets, [points], vis_dir)
    return score_batches, gt_counts


def pred_counts_at_threshold(score_batches, threshold):
    return [int((scores > threshold).sum().item()) for scores in score_batches]


@torch.no_grad()
def _collect_count_predictions(model, data_loader, device, vis_dir=None, threshold=0.5):
    model.eval()
    pred_counts = []
    gt_counts = []
    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        gt_cnt = int(targets[0]['point'].shape[0])
        mask = outputs_scores > threshold
        predict_cnt = int(mask.sum())
        pred_counts.append(predict_cnt)
        gt_counts.append(gt_cnt)
        if vis_dir is not None:
            pts = outputs_points[mask]
            points = pts.detach().cpu().numpy().tolist()
            vis(samples, targets, [points], vis_dir)
    return pred_counts, gt_counts


@torch.no_grad()
def evaluate_crowd_count_metrics(
    model,
    data_loader,
    device,
    vis_dir=None,
    threshold=0.5,
    count_tol=5,
    ratio_tol=0.10,
    count_tols=None,
    ratio_tols=None,
):
    pred_counts, gt_counts = _collect_count_predictions(
        model, data_loader, device, vis_dir=vis_dir, threshold=threshold
    )
    return compute_count_metrics(
        pred_counts,
        gt_counts,
        count_tol=count_tol,
        ratio_tol=ratio_tol,
        count_tols=count_tols,
        ratio_tols=ratio_tols,
    )


@torch.no_grad()
def evaluate_crowd_count_metrics_at_thresholds(
    model,
    data_loader,
    device,
    thresholds=None,
    count_tol=5,
    ratio_tol=0.10,
    count_tols=None,
    ratio_tols=None,
):
    """One forward pass; metrics for each score threshold."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    score_batches, gt_counts = _collect_count_scores(model, data_loader, device)
    out = []
    for threshold in thresholds:
        pred_counts = pred_counts_at_threshold(score_batches, threshold)
        metrics = compute_count_metrics(
            pred_counts,
            gt_counts,
            count_tol=count_tol,
            ratio_tol=ratio_tol,
            count_tols=count_tols,
            ratio_tols=ratio_tols,
        )
        metrics["threshold"] = float(threshold)
        out.append(metrics)
    return out


# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    metrics = evaluate_crowd_count_metrics(model, data_loader, device, vis_dir=vis_dir)
    return metrics["mae"], metrics["rmse"]