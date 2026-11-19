import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import json
import re
import zlib


def _image_id_from_filename(img_path: str, index: int, multi_root: bool = False) -> int:
    """Stable int id for vis/collate; supports e.g. row_23_col_25 - 副本.png (not only split('_')[-1])."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    m = re.search(r"row_(\d+)_col_(\d+)", stem, re.I)
    if m:
        base = int(m.group(1)) * 10_000 + int(m.group(2))
    else:
        m = re.search(r"col_(\d+)", stem, re.I)
        if m:
            base = int(m.group(1))
        else:
            nums = re.findall(r"\d+", stem)
            base = int(nums[-1]) if nums else int(index)
    if not multi_root:
        return int(base)
    p = os.path.normcase(os.path.normpath(img_path))
    h = zlib.crc32(p.encode("utf-8", errors="surrogateescape")) & 0xFFFFFFFF
    return int((base * 1_000_003 + h % 1_000_003) % (2**31)) or int(index)


class SHHA(Dataset):
    def __init__(
        self,
        data_root,
        transform=None,
        train=False,
        patch=False,
        flip=False,
        fixed_side: int = 0,
        patch_side: int = 128,
    ):
        if isinstance(data_root, (list, tuple)):
            roots = [os.path.normpath(str(r).strip()) for r in data_root if str(r).strip()]
        else:
            roots = [os.path.normpath(str(data_root).strip())]
        if not roots:
            raise ValueError("SHHA: empty data_root")
        self.root_path = roots[0]
        self._multi_root = len(roots) > 1
        # 列表文件放在 data_root 下：把写好内容的 .txt 改名为 .list 即可。
        # 每行：<图片路径> <空格> <标注文件路径>（标注可为 .txt 点坐标，或 .json 等，见 load_data）
        self.train_lists = os.environ.get("SHHA_TRAIN_LIST", "train.list")
        self.eval_list = os.environ.get("SHHA_TEST_LIST", "test.list")
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []

        def _resolve_path(root, p: str) -> str:
            p = p.strip().strip('"').strip("'")
            p = p.replace('\\', '/')
            if os.path.isabs(p):
                return os.path.normpath(p)
            return os.path.normpath(os.path.join(root, p))

        def _open_list_file(root, list_path: str):
            # 允许 list 文件本身为绝对路径
            primary = _resolve_path(root, list_path)
            alt = _resolve_path(
                root,
                "shanghai_tech_part_a_train.list" if train else "shanghai_tech_part_a_test.list",
            )
            for candidate in (primary, alt):
                if os.path.exists(candidate):
                    return open(candidate, encoding="utf-8")
            raise FileNotFoundError(
                f"List file not found. Tried: {primary} and {alt}. "
                f"请在 data_root ({root}) 下放置 train.list / test.list（或由 .txt 改名而来）。"
            )

        # loads the image/gt pairs from one or more roots (each with train.list / test.list)
        for root in roots:
            for _, train_list in enumerate(self.img_list_file):
                train_list = train_list.strip()
                with _open_list_file(root, train_list) as fin:
                    for line in fin:
                        if len(line) < 2:
                            continue
                        raw = line.strip()
                        if not raw or raw.startswith("#"):
                            continue
                        # Prefer tab: paths may contain spaces (e.g. "Tibetan juniper").
                        if "\t" in raw:
                            parts = raw.split("\t", 1)
                        else:
                            parts = raw.split(maxsplit=1)
                        if len(parts) != 2:
                            continue
                        img_p = _resolve_path(root, parts[0])
                        gt_p = _resolve_path(root, parts[1])
                        self.img_map[img_p] = gt_p
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip
        self.patch_side = int(patch_side) if patch_side else 128
        self.fixed_side = int(fixed_side) if fixed_side else 0

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        if self.fixed_side > 0:
            w0, h0 = img.size
            s = self.fixed_side
            if w0 < 1 or h0 < 1:
                raise ValueError(f"Invalid image size {w0}x{h0}: {img_path}")
            if w0 != s or h0 != s:
                sx = s / float(w0)
                sy = s / float(h0)
                img = img.resize((s, s), Image.BILINEAR)
                if point.size > 0:
                    point = np.asarray(point, dtype=np.float64, copy=True)
                    point[:, 0] *= sx
                    point[:, 1] *= sy
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        # Flip the full image in pixel-center coordinates BEFORE cropping.
        # Pixel-center coordinates use [0,W-1] x [0,H-1].
        if self.train and self.flip and random.random() > 0.5:
            img = torch.as_tensor(img, dtype=torch.float32).flip(-1)
            point = np.asarray(point, dtype=np.float64).copy()
            point[:, 0] = (int(img.shape[-1]) - 1.0) - point[:, 0]

        # Half-open crops follow the actual tensor slice; x=end_w is outside.
        if self.train and self.patch:
            img, point = random_crop(img, point, patch_side=self.patch_side)
            point = [torch.tensor(p, dtype=torch.float32) for p in point]

        # patch mode: point is a list (one entry per 128-crop). Full image: one ndarray (N,2) -> wrap as [point].
        if not self.train:
            point = [point]
        elif not self.patch:
            point = [np.asarray(point, dtype=np.float64)]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = _image_id_from_filename(img_path, index, self._multi_root)
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


def _imread_bgr_unicode(path: str):
    """cv2.imread() often fails on Windows when path contains non-ASCII (e.g. Chinese)."""
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image (corrupt/unsupported format?): {path}")
    return img


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = _imread_bgr_unicode(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    if gt_path.lower().endswith(".json"):
        # utf-8-sig strips BOM; read whole string so we can give a clear error path
        with open(gt_path, encoding="utf-8-sig") as f_label:
            raw = f_label.read()
        if not raw.strip():
            data = {"shapes": []}
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON label (empty save, truncated file, or not UTF-8 JSON): {gt_path}\n"
                    f"JSONDecodeError: {e}"
                ) from e
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "point":
                continue
            pts = shape.get("points") or []
            if not pts or not pts[0] or len(pts[0]) < 2:
                continue
            x, y = float(pts[0][0]), float(pts[0][1])
            points.append([x, y])
    else:
        with open(gt_path, encoding="utf-8") as f_label:
            for line in f_label:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                points.append([x, y])

    if len(points) == 0:
        pts_arr = np.zeros((0, 2), dtype=np.float64)
    else:
        pts_arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return img, pts_arr

# random crop augumentation
def random_crop(img, den, num_patch=4, patch_side=128):
    half_h = int(patch_side)
    half_w = int(patch_side)
    if img.size(1) < half_h or img.size(2) < half_w:
        raise ValueError(
            'random_crop: image %dx%d smaller than patch_side=%d'
            % (img.size(2), img.size(1), patch_side)
        )
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] < end_w) & (den[:, 1] >= start_h) & (den[:, 1] < end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den