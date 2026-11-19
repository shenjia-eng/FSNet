# Datasets

## MT-Set

MT-Set contains 2,985 satellite-image tiles and 2,515 JSON annotation files with 201,189 tree-crown point annotations. There are 470 images without a corresponding annotation JSON. Most images are 800×800 pixels; the loader converts images to RGB.

The dataset is available under [CC BY-NC 4.0](../LICENSE_DATA.md).

## Download and extract

Download all three ZIP files from [Releases](https://github.com/shenjia-eng/FSNet/releases/tag/v1.0.0) and extract each into the same directory. They form one MT-Set collection and create the following structure:

```text
mt-set/
  cropped_images_800(Giant cypress)/
  cropped_images_800(Tibetan juniper)/
  cropped_images_800_Tibetan juniper/
  cropped_images_800_fir_merged/
  cropped_images_800_spruce_merged/
  images_800_Giant_cypress/
  images_800_Tibetan_juniper/
```

| ZIP file | PNG images | Annotation JSON files |
|---|---:|---:|
|`mt-set-v2-part01.zip`|983|983|
|`mt-set-v2-part02.zip`|1148|850|
|`mt-set-v2-part03.zip`|854|682|

Each ZIP is independently extractable. Pass the resulting `mt-set/` folder as `--data-root`.

## Dataset splits

| Manifest key | Images | Points |
|---|---:|---:|
|`train`|2074|102679|
|`validation`|228|10175|

Image assignments are listed in `data/splits/train.tsv` and `data/splits/validation.tsv`. `data/data_manifest.json` records image and annotation paths, checksums, dimensions, and point counts.

## Collection contents

| Original collection directory | PNG images | Annotation JSON files |
|---|---:|---:|
|`cropped_images_800(Giant cypress)`|1232|1232|
|`cropped_images_800(Tibetan juniper)`|1225|784|
|`cropped_images_800_Tibetan juniper`|186|157|
|`cropped_images_800_fir_merged`|176|176|
|`cropped_images_800_spruce_merged`|108|108|
|`images_800_Giant_cypress`|33|33|
|`images_800_Tibetan_juniper`|25|25|
|**Original collection total**|**2985**|**2515**|

## Annotation format

Images are stored as PNG files and annotations as JSON files. Point coordinates are expressed in pixels relative to the original image. Annotation filenames share the corresponding image's basename. The data loader reads point annotations from the JSON `shapes` entries.

## External dataset sources

- **NeonTreeEvaluation:** [dataset v0.2.2](https://doi.org/10.5281/zenodo.5914554), [benchmark project](https://github.com/weecology/NeonTreeEvaluation_package), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reference: Weinstein et al., PLOS Computational Biology 17(7):e1009180, 2021. [DOI](https://doi.org/10.1371/journal.pcbi.1009180).
- **MillionTrees / Yosemite:** [MillionTrees dataset description](https://github.com/weecology/MillionTrees/blob/main/docs/datasets.md#chen-and-shang-2022), [Yosemite Tree Dataset](https://github.com/nightonion/yosemite-tree-dataset). Reference: Chen and Shang, *Transformer for Tree Counting in Aerial Images*, Remote Sensing 14(3):476, 2022. [DOI](https://doi.org/10.3390/rs14030476). Obtain these images under the original providers' terms.
