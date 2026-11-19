# Third-party notices

Keep the copyright and license notices in the source files when redistributing this package. The code license does not grant rights to independently obtained datasets or model weights.

## Code license

Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.

Academic research only. See [LICENSE](LICENSE) for the complete terms and bundled dependency notices.

## DETR

The backbone, matching, training-engine and utility modules retain Facebook copyright notices. Some files explicitly identify adapted code from [DETR](https://github.com/facebookresearch/detr), distributed under the Apache License 2.0. Preserve those file-level notices and the Apache license text in `licenses/Apache-2.0.txt`.

## DFFormer and timm

The optional DFFormer comparison component uses `dfformer_upstream.py`, which retains the Apache License 2.0 notice and Copyright 2022 Garena Online Private Limited. Its header also credits implementations adapted from [timm](https://github.com/huggingface/pytorch-image-models). The DFFormer source identifies the [official DFFormer repository](https://github.com/okojoalg/dfformer). Preserve the source notices and `licenses/Apache-2.0.txt` whenever distributing this component.

## PyTorch, torchvision and other dependencies

PyTorch, torchvision, NumPy, SciPy, Pillow, OpenCV, PyYAML and other separately installed dependencies retain their own licenses. The bundled LICENSE includes notices for PyTorch, torchvision, OpenCV and Pillow. Installing a dependency does not change its license. The VGG-derived files retain their existing Facebook copyright notices.

## Research attribution

The frequency-filter implementation acknowledges GFNet-style complex frequency filtering in its source comments. Other optional spectral comparisons are identified in code as inspired by DFFormer, AFNO and FNet.

## Data and trained weights

MT-Set data and the released trained weights have separate usage terms, described with those assets. External datasets remain subject to their original providers' terms. The license for this source code must not be interpreted as relicensing external imagery.
