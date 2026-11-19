"""Paths, data loading, and model initialization for FSNet."""
from pathlib import Path
import hashlib
import json
import os
import sys
import types


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_path(override=None):
    p = Path(override).resolve() if override else Path(__file__).resolve().parents[1] / "vendor/fsnet"
    if not (p / "source/models/fsnet.py").is_file():
        raise ValueError("Missing source file: " + str(p))
    sys.path[:0] = [str(p), str(p / "source")]
    return p


def records_for(manifest, split, data_root, verify=True):
    document = read_json(manifest)
    records = document[split] if split in document else document["splits"][split]
    root = Path(data_root).resolve()
    result = []
    for original in records:
        row = dict(original)
        for key in ("image", "annotation"):
            rel = Path(row[key])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Public manifest requires relative paths: " + str(rel))
            p = (root / rel).resolve()
            if not p.is_relative_to(root) or not p.is_file():
                raise ValueError("Missing dataset file: " + str(p))
            if verify and row.get(key + "_sha256") and sha256(p) != row[key + "_sha256"]:
                raise ValueError("Dataset checksum mismatch: " + str(p))
            row[key] = str(p)
        result.append(row)
    if not result:
        raise ValueError("Empty split: " + split)
    return result


def validate_training_splits(document, configs):
    default = {"id": "mt-set-2074-230-v1", "train": 2074, "validation": 230}
    spec = configs[0].get("dataset_split", default)
    if any(c.get("dataset_split", default) != spec for c in configs):
        raise ValueError("Stage configurations use different dataset splits")
    if document.get("split_id", default["id"]) != spec["id"]:
        raise ValueError("Dataset split does not match stage configurations")
    splits = document.get("splits", document)
    counts = {key: len(splits[key]) for key in ("train", "validation")}
    if any(counts[key] != spec[key] for key in counts):
        raise ValueError("Dataset sizes do not match stage configurations")
    hashes = {key: [r["image_sha256"] for r in splits[key]] for key in counts}
    if any(len(set(hashes[key])) != counts[key] for key in counts):
        raise ValueError("Duplicate images within a dataset split")
    if set(hashes["train"]) & set(hashes["validation"]):
        raise ValueError("Training and validation images must be disjoint")
    return counts


def load_driver(source, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    filename = source / "audited_train.py"
    text = filename.read_text(encoding="utf-8")
    marker = "ROOT=Path(os.environ.get('FSNET_OUTPUT_ROOT', 'outputs'))"
    if text.count(marker) != 1:
        raise ValueError("Expected exactly one output-root assignment in the training driver")
    module = types.ModuleType("audited_train")
    module.__file__ = str(filename)
    module.__dict__["_portable_root"] = str(output)
    sys.modules[module.__name__] = module
    exec(compile(text.replace(marker, "ROOT=Path(_portable_root)"), str(filename), "exec"), module.__dict__)
    return module


def configure_torch(seed=42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from training_core import seed_all
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(seed)


def build_for_inference(config, checkpoint, device):
    import torch
    from types import SimpleNamespace
    from models import build_model
    from variants import apply_variant
    cfg = read_json(config)
    args = dict(cfg.get("args", cfg.get("model_args", cfg)))
    args["backbone_pretrained"] = False
    # Allocate learned FFT grids at the training crop size before strict loading.
    # This does not resize input images during evaluation.
    args["infer_fixed_side"] = int(args.get("patch_side", 128))
    model = build_model(SimpleNamespace(**args), training=False)
    apply_variant(model, cfg.get("variant", "fsnet"), args.get("seed", 42))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved.get("model", saved), strict=True)
    return model.to(device).eval(), cfg


def validation_dataset(records):
    import torch
    from torchvision import transforms
    from crowd_datasets.SHHA.SHHA import load_data

    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.img_list = [r["image"] for r in records]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ])

        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            row = records[index]
            image, points = load_data((row["image"], row["annotation"]), False)
            target = {"point": torch.tensor(points, dtype=torch.float32),
                      "labels": torch.ones(len(points), dtype=torch.long),
                      "image_id": torch.tensor([index], dtype=torch.long)}
            return self.transform(image), [target]
    return Dataset()
