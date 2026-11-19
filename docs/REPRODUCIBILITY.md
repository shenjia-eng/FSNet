# Training and evaluation

Explore FSNet with five pretrained models and ready-to-use evaluation commands.

## Installation

Use Linux and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Training requires CUDA. Evaluation supports CUDA and CPU (`--device cpu`).

## Data and pretrained models

Download the three MT-Set ZIP files and a model checkpoint from [Releases](https://github.com/shenjia-eng/FSNet/releases/tag/v1.0.0). Extract all ZIP files into the same directory and place checkpoints in `weights/`. See [DATASETS.md](DATASETS.md) for the directory structure and data format.

| Seed | Configuration | Checkpoint |
|---:|---|---|
|42|`configs/fsnet_seed42.json`|`weights/fsnet_seed42.pth`|
|123|`configs/fsnet_seed123.json`|`weights/fsnet_seed123.pth`|
|456|`configs/fsnet_seed456.json`|`weights/fsnet_seed456.pth`|
|789|`configs/fsnet_seed789.json`|`weights/fsnet_seed789.pth`|
|2026|`configs/fsnet_seed2026.json`|`weights/fsnet_seed2026.pth`|

## Evaluation

```bash
python scripts/evaluate.py --config configs/fsnet_seed42.json --checkpoint weights/fsnet_seed42.pth --manifest data/data_manifest.json --split validation --data-root /path/to/mt-set --output outputs/validation42 --device cuda
```

Use the corresponding configuration and checkpoint for each seed. Predictions and metrics are saved to the output directory.

Images are loaded as RGB with ImageNet normalization and padded to a multiple of 128. Predictions with foreground probability strictly greater than 0.5 and coordinates inside the original image bounds are retained. Counting uses the number of retained points. Localization metrics use one-to-one point matching at radii of 8, 12, and 16 pixels; the main results use 12 pixels.

## Training: mt-set-2074-228-v2

The training CLI supports seed42. Download [VGG16-BN](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth) for initialization and run:

```bash
python scripts/train.py --stages configs/stages_v2/42 --manifest data/data_manifest.json --data-root /path/to/mt-set --imagenet /path/to/vgg16_bn-6c64b313.pth --output outputs/train42_v2 --seed 42
```

| Stage | Epochs | Configuration |
|---|---:|---|
|F00|150|`F00_base150.json`|
|F01|50|`F01_fpn50.json`|
|F02|50|`F02_both50.json`|
|F03|50|`F03_gate_context50.json`|
|F04|50|`F04_detse50.json`|

Stage configurations are in `configs/stages_v2/42/`. Training uses batches of eight images with four random 128×128 crops per image. Learning rates, loss weights, GradNorm settings, and stage parameters are defined in the configuration files.

## Download checksums

Run the following command with all downloaded files in one directory:

```bash
python scripts/verify_assets.py /path/to/downloads
```

Expected SHA256 values are listed in `SHA256SUMS` and `release_assets.json`.
