# dataset.py
import os
import re
from copy import deepcopy
from functools import partial
from glob import glob
from hashlib import sha1
from typing import Callable, Iterable, Optional, Tuple, Union, Dict, List

import cv2
import imgaug.augmenters as iaa
import numpy as np
from glog import logger
from joblib import Parallel, cpu_count, delayed
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

import aug


# -------------------------
# utils: subsampling buckets
# -------------------------
def subsample(
    data: Iterable,
    bounds: Tuple[float, float],
    hash_fn: Callable,
    n_buckets: int = 100,
    salt: str = "",
    verbose: bool = True,
):
    data = list(data)
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt, hash_fn=hash_fn)

    lower_bound, upper_bound = [x * n_buckets for x in bounds]
    msg = f"Subsampling buckets from {lower_bound} to {upper_bound}, total buckets number is {n_buckets}"
    if salt:
        msg += f"; salt is {salt}"
    if verbose:
        logger.info(msg)

    return np.array([sample for bucket, sample in zip(buckets, data) if lower_bound <= bucket < upper_bound])


def hash_from_paths(x: Tuple[str, str], salt: str = "") -> str:
    path_a, path_b = x
    names = "".join(map(os.path.basename, (path_a, path_b)))
    return sha1(f"{names}_{salt}".encode()).hexdigest()


def split_into_buckets(data: Iterable, n_buckets: int, hash_fn: Callable, salt: str = ""):
    hashes = map(partial(hash_fn, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])


def _read_img(path: str):
    img = cv2.imread(path)
    if img is None:
        logger.warning(f"Can not read image {path} with OpenCV, switching to scikit-image")
        img = imread(path)
        # skimage may return RGB; keep as-is (consistent within dataset)
    return img


def add_two_to_number(file_path: str) -> str:
    match = re.search(r"(\d+)(?=\.\w+$)", file_path)
    if match:
        number = int(match.group())
        new_number = number + 2
        return re.sub(r"(\d+)(?=\.\w+$)", str(new_number), file_path)
    return file_path


# -------------------------
# Dataset
# -------------------------
class PairedDataset(Dataset):
    """
    Returns dict:
      {
        'a': blur tensor (C,H,W) float32 [0,1]
        'b': sharp tensor (C,H,W) float32 [0,1]
        'c': sub tensor (1,H/2,W/2) float32 [0,1]
        'd': sub_target tensor (1,H/2,W/2) float32 [0,1]
        'e': edge_sub tensor (1,H/2,W/2) float32 [0,1]
        'b_path': sharp path (str)
      }
    Compatible with your train.py/test.py + models/models.py(DeblurModel.get_input).
    """

    def __init__(
        self,
        files_a: List[str],
        files_b: List[str],
        files_c: List[str],
        transform_fn: Callable,
        normalize_fn: Callable,
        corrupt_fn: Optional[Callable] = None,
        preload: bool = False,
        preload_size: int = 0,
        filter: bool = True,
        fft_hpf: bool = True,
        verbose: bool = True,
    ):
        assert len(files_a) == len(files_b) == len(files_c), (
            len(files_a),
            len(files_b),
            len(files_c),
        )

        self.files_a = files_a
        self.files_b = files_b
        self.files_c = files_c

        self.verbose = verbose
        self.corrupt_fn = corrupt_fn
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn

        # NOTE: original code used height=1280//2, width=720//2 (swapped).
        # Keep it as-is for backward compatibility.
        self.sub_resize = iaa.Resize({"height": 1280 // 2, "width": 720 // 2})

        self.filter = filter
        self.fft_hpf = fft_hpf

        # preload caches (optional)
        self.preload = bool(preload)
        self.preload_size = int(preload_size) if preload_size else 0
        self._cache_a: Optional[List[np.ndarray]] = None
        self._cache_b: Optional[List[np.ndarray]] = None

        logger.info(f"Dataset has been created with {len(self.files_a)} samples")

        if self.preload:
            self._cache_a = self._bulk_preload(self.files_a, preload_size=self.preload_size)
            self._cache_b = self._bulk_preload(self.files_b, preload_size=self.preload_size)

    def _bulk_preload(self, paths: List[str], preload_size: int):
        jobs = [delayed(self._preload_one)(p, preload_size=preload_size) for p in paths]
        jobs = tqdm(jobs, desc="preloading images", disable=not self.verbose)
        return Parallel(n_jobs=cpu_count(), backend="threading")(jobs)

    @staticmethod
    def _preload_one(path: str, preload_size: int):
        img = _read_img(path)
        if preload_size:
            h, w = img.shape[:2]
            h_scale = preload_size / h
            w_scale = preload_size / w
            scale = max(h_scale, w_scale)
            img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
            assert min(img.shape[:2]) >= preload_size, f"weird img shape: {img.shape}"
        return img

    # -------------------------
    # subframe 만들기 (경로/이미지 모두 지원)
    # -------------------------
    @staticmethod
    def calculate_sigma_color(image: np.ndarray, k: float = 0.01) -> float:
        color_diff = np.abs(image - np.mean(image, axis=(0, 1)))
        sigma_color = np.std(color_diff, axis=(0, 1))
        sigma_color = float(np.mean(k * sigma_color))
        return sigma_color

    @staticmethod
    def radius_filter(image: np.ndarray, radius: int = 25) -> np.ndarray:
        spectrum = fftshift(fft2(image))
        image_size_x, image_size_y = image.shape[0], image.shape[1]
        center_x, center_y = image_size_x // 2, image_size_y // 2

        y, x = np.ogrid[:image_size_x, :image_size_y]
        mask_top_right = (x - (image_size_y - center_y)) ** 2 + (y - center_x) ** 2 <= radius**2
        spectrum[mask_top_right] = 0
        filtered_image = np.abs(ifft2(ifftshift(spectrum)))
        return filtered_image

    def make_subframe(
        self,
        frame: Union[str, np.ndarray],
        target_frame: Union[str, np.ndarray],
        filter: bool = True,
        fft_hpf: bool = True,
    ):
        # load if path
        if isinstance(frame, str):
            frame = cv2.imread(frame)
            if frame is None:
                frame =\db_read_fallback(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # assume already image array
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if isinstance(target_frame, str):
            target_frame = cv2.imread(target_frame)
            if target_frame is None:
                target_frame = _read_img(target_frame)
            if target_frame.ndim == 3:
                target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        else:
            if target_frame.ndim == 3:
                target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

        # resize to sub resolution
        frame = self.sub_resize.augment_image(frame)
        target_frame = self.sub_resize.augment_image(target_frame)

        # thresholding
        threshold_value = np.random.randint(int(0.1 * 255.0), int(0.3 * 255.0))

        if np.max(frame) <= threshold_value:
            th_frame = frame
        else:
            th_frame = np.where(frame < threshold_value, 0, frame).astype(np.uint8)

        if np.max(target_frame) <= threshold_value:
            th_target_frame = target_frame
        else:
            th_target_frame = np.where(target_frame < threshold_value, 0, target_frame).astype(np.uint8)

        # normalize to 0..255 safely
        def safe_norm_255(x: np.ndarray) -> np.ndarray:
            mn, mx = float(np.min(x)), float(np.max(x))
            if mx - mn < 1e-6:
                return x.astype(np.uint8)
            y = (x - mn) / (mx - mn) * 255.0
            return y.astype(np.uint8)

        th_frame = safe_norm_255(th_frame)
        th_target_frame = safe_norm_255(th_target_frame)

        # gaussian noise
        gaussian_noise = np.random.uniform(0.05, 0.15)
        gaussian_noise_aug = iaa.AdditiveGaussianNoise(scale=gaussian_noise * 255)
        th_frame = gaussian_noise_aug(images=th_frame)

        # filter
        th_gt_frame = th_frame
        if filter:
            sigmaColor = self.calculate_sigma_color(image=th_frame, k=0.01)
            th_gt_frame = cv2.GaussianBlur(th_frame, (5, 5), sigmaColor)

        # fft high-pass filter
        if fft_hpf:
            edge_th_frame = self.radius_filter(th_gt_frame)
            return th_frame, th_target_frame, edge_th_frame

        # if no fft_hpf, still return a dummy edge map for compatibility
        edge_th_frame = np.zeros_like(th_frame, dtype=np.float32)
        return th_frame, th_target_frame, edge_th_frame

    # -------------------------
    # normalize: CHW tensors
    # -------------------------
    @staticmethod
    def _to_chw_float_tensor(img_hwc: np.ndarray) -> torch.Tensor:
        # expects uint8 HWC
        if img_hwc.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape={img_hwc.shape}")
        x = img_hwc.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        return torch.from_numpy(x)

    @staticmethod
    def _to_1hw_float_tensor(img_hw: np.ndarray) -> torch.Tensor:
        # expects uint8/float HW
        x = img_hw.astype(np.float32) / 255.0
        if x.ndim != 2:
            # if somehow HWC single channel
            x = x[..., 0]
        x = np.expand_dims(x, axis=0)  # 1HW
        return torch.from_numpy(x)

    def _preprocess(self, img: np.ndarray, res: np.ndarray, sub: np.ndarray, sub_tar: np.ndarray, sub_edge: np.ndarray):
        # ignore aug.get_normalize() implementation (it permutes to HWC and can break models)
        a = self._to_chw_float_tensor(img)
        b = self._to_chw_float_tensor(res)
        c = self._to_1hw_float_tensor(sub)
        d = self._to_1hw_float_tensor(sub_tar)
        e = self._to_1hw_float_tensor(sub_edge)
        return a, b, c, d, e

    def __len__(self):
        return len(self.files_a)

    def __getitem__(self, idx):
        a_path = self.files_a[idx]
        b_path = self.files_b[idx]
        c_path = self.files_c[idx]  # sub frame path (+2)

        # load main images
        if self.preload and self._cache_a is not None and self._cache_b is not None:
            a_img = self._cache_a[idx]
            b_img = self._cache_b[idx]
        else:
            a_img = _read_img(a_path)
            b_img = _read_img(b_path)

        # make sub images (use +2 frame for "frame", current sharp for "target")
        sub_img, sub_target_img, sub_edge_img = self.make_subframe(
            frame=c_path,
            target_frame=b_path,
            filter=self.filter,
            fft_hpf=self.fft_hpf,
        )

        # albumentations expects numpy images
        a_img, b_img, sub_img, sub_target_img, sub_edge_img = self.transform_fn(
            a_img, b_img, sub_img, sub_target_img, sub_edge_img
        )

        if self.corrupt_fn is not None:
            a_img = self.corrupt_fn(a_img)

        a, b, c, d, e = self._preprocess(a_img, b_img, sub_img, sub_target_img, sub_edge_img)

        return {"a": a, "b": b, "c": c, "d": d, "e": e, "b_path": b_path}

    # -------------------------
    # factory from config
    # -------------------------
    @staticmethod
    def from_config(config: Dict):
        config = deepcopy(config)

        files_a = sorted(glob(config["files_a"], recursive=True))  # blur
        files_b = sorted(glob(config["files_b"], recursive=True))  # sharp

        # 1) 정확한 페어링: basename 기준으로 매칭 (치명 버그 수정)
        a_map = {os.path.basename(p): p for p in files_a}
        b_map = {os.path.basename(p): p for p in files_b}
        common = sorted(set(a_map.keys()) & set(b_map.keys()))

        paired_a = [a_map[name] for name in common]
        paired_b = [b_map[name] for name in common]

        # 2) sub frame (+2) 만들고 존재하는 것만 남김
        paired_c = [add_two_to_number(p) for p in paired_b]

        existing_a, existing_b, existing_c = [], [], []
        for ap, bp, cp in zip(paired_a, paired_b, paired_c):
            if os.path.exists(cp):
                existing_a.append(ap)
                existing_b.append(bp)
                existing_c.append(cp)

        # 3) subsample (train/val split 유지)
        verbose = config.get("verbose", True)
        data = subsample(
            data=zip(existing_a, existing_b),
            bounds=tuple(config.get("bounds", (0, 1))),
            hash_fn=hash_from_paths,
            verbose=verbose,
        )
        existing_a, existing_b = map(list, zip(*data))

        # existing_c도 existing_b와 같은 basename 순서라고 가정하지 말고 다시 매핑
        # (subsample 이후 인덱스가 바뀌었으므로, b 기준으로 다시 c를 계산)
        existing_c = [add_two_to_number(p) for p in existing_b]

        # 4) transforms
        transform_fn = aug.get_transforms(size=config["size"], scope=config["scope"], crop=config.get("crop", "random"))

        # normalize_fn은 사용하진 않지만 인터페이스 유지 (외부 코드가 기대할 수 있어 보관)
        normalize_fn = lambda a, b, c, d, e: (a, b, c, d, e)

        return PairedDataset(
            files_a=existing_a,
            files_b=existing_b,
            files_c=existing_c,
            transform_fn=transform_fn,
            normalize_fn=normalize_fn,
            corrupt_fn=None,
            preload=bool(config.get("preload", False)),
            preload_size=int(config.get("preload_size", 0) or 0),
            filter=bool(config.get("filter", True)),
            fft_hpf=bool(config.get("fft_hpf", True)),
            verbose=verbose,
        )

def db_read_fallback(path: str) -> np.ndarray:
    # small helper: keep behavior similar if cv2.imread fails
    img = _read_img(path)
    if img.ndim == 2:
        return img
    return img
