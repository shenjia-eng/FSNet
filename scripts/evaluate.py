"""Put FSNet pretrained models to work on individual images or datasets."""
import argparse
import gzip
import json
from pathlib import Path
from runtime import (read_json, sha256, source_path, records_for, configure_torch,
                     build_for_inference, validation_dataset)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--split", choices=["train", "validation"], default="validation")
    ap.add_argument("--data-root")
    ap.add_argument("--image", help="Single-image prediction; no annotation needed")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--source-root")
    ap.add_argument("--threads", type=int, default=2)
    ns = ap.parse_args()
    if bool(ns.image) == bool(ns.manifest):
        ap.error("Choose exactly one of --image or --manifest")
    if ns.manifest and not ns.data_root:
        ap.error("--manifest requires --data-root")
    out = Path(ns.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    source_path(ns.source_root)
    import torch
    import numpy as np
    from metrics import evaluate_loader, evaluation_collate, select_predictions, match_counts, summarize
    torch.set_num_threads(ns.threads)
    configure_torch()
    model, cfg = build_for_inference(ns.config, ns.checkpoint, torch.device(ns.device))
    protocol = dict(score_threshold=.5, match_radius_px=8., coordinate_frame="input_pixels", bounds="half_open")
    if ns.image:
        from PIL import Image
        from torchvision import transforms
        from util.misc import nested_tensor_from_tensor_list
        im = Image.open(ns.image).convert("RGB")
        x = transforms.functional.normalize(transforms.functional.to_tensor(im), [.485,.456,.406], [.229,.224,.225])
        batch = nested_tensor_from_tensor_list([x]).to(ns.device)
        with torch.no_grad():
            pred = model(batch)
        points, scores, filters = select_predictions(
            pred["pred_points"][0].float().cpu().numpy(),
            pred["pred_logits"].float().softmax(-1)[0,:,1].cpu().numpy(),
            im.width, im.height, .5)
        result = dict(image=Path(ns.image).name, width=im.width, height=im.height,
                      pred_count=len(points), pred_points=points.tolist(),
                      pred_scores=scores.tolist(), filter_counts=filters, protocol=protocol)
        (out / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        records = records_for(ns.manifest, ns.split, ns.data_root)
        ds = validation_dataset(records)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                            num_workers=0, collate_fn=evaluation_collate)
        result = evaluate_loader(model, loader, torch.device(ns.device), out / "predictions.jsonl.gz",
                                 protocol, ds.img_list, save_raw=False)
        # All radii use this one frozen set of predictions and the same checkpoint.
        with gzip.open(out / "predictions.jsonl.gz", "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        by_radius = {}
        for radius in (8, 12, 16):
            derived = []
            for row in rows:
                tp, fp, fn = match_counts(row["pred_points"], np.asarray(row["gt_points"], dtype=np.float32), radius)
                derived.append(dict(pred_count=row["pred_count"], gt_count=row["gt_count"], tp=tp, fp=fp, fn=fn))
            by_radius[str(radius)] = summarize(derived)
        result.update(split=ns.split, metrics_by_radius=by_radius,
                      manifest_sha256=sha256(ns.manifest), evaluation_protocol=protocol)
    result.update(checkpoint_sha256=sha256(ns.checkpoint), config_sha256=sha256(ns.config),
                  device=ns.device, torch=torch.__version__)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result.get("metrics_by_radius", {"pred_count": result.get("pred_count")}), indent=2))


if __name__ == "__main__":
    main()
