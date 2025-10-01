import os
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box


class AugConfig:
    resize_lim = (0.38, 0.55)
    final_dim = (256, 704)     # (H, W)
    rot_lim = (-5.0, 5.0)      # deg
    random_flip = True
    color_jitter = False
    cam_drop_prob = 0.0
    extrinsic_noise_std = 0.0


class GridConfig:
    # X x Y grid: 50m x 50m with 0.5m x 0.5m cells -> BEV grid 200 x 200
    xbound = (-50.0, 50.0, 0.5)
    ybound = (-50.0, 50.0, 0.5)
    zbound = (-10.0, 10.0, 20.0)

    def gen_dx_bx(self) -> Dict[str, np.ndarray]:
        def p(b):
            lo, hi, step = b
            n = int(round((hi - lo) / step))
            return step, lo, n

        dx, bx, nx = p(self.xbound)
        dy, by, ny = p(self.ybound)
        dz, bz, nz = p(self.zbound)

        dbins = np.arange(4.0, 45.0 + 1e-6, 1.0, dtype=np.float32)

        return {
            "dx": np.array([dx, dy, dz], np.float32),
            "bx": np.array([bx, by, bz], np.float32),
            "nx": np.array([nx, ny, nz], np.int64),
            "dbins": dbins,
        }


class NuscData(Dataset):
    CAMERAS = (
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    )

    def __init__(
        self,
        nusc: NuScenes,
        dataroot: str,
        version: str,
        split: str,                      # "train" | "val" | "test"
        grid_conf: GridConfig,
        aug_conf: AugConfig,
        cams: Optional[List[str]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.nusc = nusc
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.grid_conf = grid_conf
        self.aug_conf = aug_conf
        self.cams = cams if cams is not None else list(self.CAMERAS)
        self.rng = np.random.default_rng(seed)

        self.scenes = self.get_scenes(split)
        self.sample_tokens = self.preprocess(self.scenes)
        self.grid_data = self.grid_conf.gen_dx_bx()

        self.dx = self.grid_data["dx"]
        self.bx = self.grid_data["bx"]
        self.nx = self.grid_data["nx"]

        # Small caches for static per-camera params
        self._calib_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # ---------------------------- utilities ----------------------------

    @staticmethod
    def _project_to_so3(R: np.ndarray) -> np.ndarray:
        """Project a near-rotation to SO(3) via SVD (polar decomposition)."""
        U, _, Vt = np.linalg.svd(R.astype(np.float64))
        Rp = U @ Vt
        if np.linalg.det(Rp) < 0:
            U[:, -1] *= -1
            Rp = U @ Vt
        return Rp

    def _global2ego(self, sample: Dict[str, Any]) -> np.ndarray:
        """Return 4x4 T_global2ego for this sample (analytic inverse, float64)."""
        lidar_tok = sample["data"].get("LIDAR_TOP") or next(iter(sample["data"].values()))
        sd = self.nusc.get("sample_data", lidar_tok)
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])

        t_eg = np.array(ep["translation"], dtype=np.float64)  # ego->global
        R_eg = Quaternion(ep["rotation"]).rotation_matrix.astype(np.float64)

        R_ge = self._project_to_so3(R_eg.T)
        t_ge = -R_ge @ t_eg

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_ge
        T[:3,  3] = t_ge
        return T

    # ---------------------------- dataset plumbing ----------------------------

    def get_scenes(self, split: str) -> List[str]:
        """Return scene names for the given split."""
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'. Expected 'train','val','test'.")

        splits = create_splits_scenes()
        if "mini" in self.version:
            split_key = {"train": "mini_train", "val": "mini_val"}.get(split, split)
        else:
            split_key = split

        scene_names = splits[split_key]
        present = {s["name"] for s in self.nusc.scene}
        scene_names = [n for n in scene_names if n in present]
        if not scene_names:
            raise RuntimeError(f"No scenes found for split '{split_key}' in version '{self.version}'.")
        return scene_names

    def preprocess(self, scenes: List[str]) -> List[str]:
        """Collect sample tokens (all cameras present) in scene order."""
        name2scene = {s["name"]: s for s in self.nusc.scene}
        sample_tokens: List[str] = []

        for scene_name in scenes:
            scene = name2scene[scene_name]
            tok = scene["first_sample_token"]
            while tok:
                sample = self.nusc.get("sample", tok)
                if all(cam in sample["data"] for cam in self.cams):
                    sample_tokens.append(tok)
                tok = sample["next"]

        return sample_tokens

    # ---------------------------- IO & augmentation ----------------------------

    def get_camera_data(self, sample: Dict[str, Any], camera: str) -> Dict[str, Any]:
        """Return dict with image path, intrinsics K, and T_cam2ego (4x4)."""
        sd = self.nusc.get('sample_data', sample['data'][camera])
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

        K = np.array(cs['camera_intrinsic'], dtype=np.float32)
        R = Quaternion(cs['rotation']).rotation_matrix.astype(np.float32)
        t = np.array(cs['translation'], dtype=np.float32)

        T_cam2ego = np.eye(4, dtype=np.float32)
        T_cam2ego[:3, :3] = R
        T_cam2ego[:3,  3] = t

        return {
            'img_path': os.path.join(self.dataroot, sd['filename']),
            'K': K,
            'T_cam2ego': T_cam2ego,
            'timestamp': sd['timestamp'],
            'channel': camera,
        }

    def sample_augmentation(self, H0: int, W0: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """No-op aug except resize to final_dim. Returns (post_rot, post_tran, (H,W))."""
        H, W = self.aug_conf.final_dim
        post_rot = np.eye(3, dtype=np.float32)
        post_tran = np.zeros(3, dtype=np.float32)
        return post_rot, post_tran, (H, W)

    def prepare_images(self, cam_infos: List[Dict[str, Any]]):
        """Load/resize/normalize images; adjust intrinsics; collect extrinsics."""
        H, W = self.aug_conf.final_dim
        imgs, intrins, rots, trans, post_rots, post_trans = [], [], [], [], [], []

        for info in cam_infos:
            im = Image.open(info["img_path"]).convert("RGB")
            W0, H0 = im.size

            post_rot_np, post_tran_np, (H_out, W_out) = self.sample_augmentation(H0, W0)
            assert (H_out, W_out) == (H, W)

            im = im.resize((W, H), resample=Image.BILINEAR)
            im = np.asarray(im, dtype=np.float32) / 255.0
            im = torch.from_numpy(im).permute(2, 0, 1).contiguous()  # (3,H,W)

            sx, sy = W / float(W0), H / float(H0)
            S = np.array([[sx, 0, 0],
                          [0,  sy, 0],
                          [0,   0, 1]], dtype=np.float32)
            K_aug = (S @ info["K"]).astype(np.float32)

            R = info["T_cam2ego"][:3, :3].astype(np.float32)
            t = info["T_cam2ego"][:3,  3].astype(np.float32)

            imgs.append(im)
            intrins.append(torch.from_numpy(K_aug))
            rots.append(torch.from_numpy(R))
            trans.append(torch.from_numpy(t))
            post_rots.append(torch.from_numpy(post_rot_np))
            post_trans.append(torch.from_numpy(post_tran_np))

        imgs       = torch.stack(imgs, dim=0)        # (N_cam,3,H,W)
        intrins    = torch.stack(intrins, dim=0)     # (N_cam,3,3)
        rots       = torch.stack(rots, dim=0)        # (N_cam,3,3)
        trans      = torch.stack(trans, dim=0)       # (N_cam,3)
        post_rots  = torch.stack(post_rots, dim=0)   # (N_cam,3,3)
        post_trans = torch.stack(post_trans, dim=0)  # (N_cam,3)

        return imgs, rots, trans, intrins, post_rots, post_trans

    # ---------------------------- labels ----------------------------

    def make_targets(self, sample: Dict[str, Any], T_global2ego: np.ndarray) -> Dict[str, torch.Tensor]:
        """Return {'occ': (1, ny, nx)} vehicle occupancy in ego BEV."""
        dx, dy = self.dx[0], self.dx[1]
        bx, by = self.bx[0], self.bx[1]
        nx, ny = self.nx[0], self.nx[1]

        occ = np.zeros((ny, nx), dtype=np.uint8)

        R_ge = self._project_to_so3(T_global2ego[:3, :3])
        t_ge = T_global2ego[:3,  3]

        for ann_tok in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_tok)
            if not ann["category_name"].startswith("vehicle."):
                continue

            # global -> ego (rotate then translate)
            box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
            box.rotate(Quaternion(matrix=R_ge))
            box.translate(t_ge)

            # ground-plane corners -> grid indices
            corners = box.bottom_corners()[:2, :].T.astype(np.float32)    # (4,2) in meters
            ix = np.floor((corners[:, 0] - bx) / dx + 0.5).astype(np.int32)
            iy = np.floor((corners[:, 1] - by) / dy + 0.5).astype(np.int32)
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)

            poly = np.stack([ix, iy], axis=1).reshape(-1, 1, 2).astype(np.int32)
            if poly.shape[0] >= 3:
                cv2.fillPoly(occ, [poly], 1)

        return {"occ": torch.from_numpy(occ.astype(np.float32)).unsqueeze(0)}

    # ---------------------------- item ----------------------------

    def __getitem__(self, index: int) -> Dict[str, Any]:
        token = self.sample_tokens[index]
        sample = self.nusc.get("sample", token)

        cam_infos = [self.get_camera_data(sample, cam) for cam in self.cams]
        imgs, rots, trans, intrins, post_rots, post_trans = self.prepare_images(cam_infos)

        T_global2ego = self._global2ego(sample)
        targets = self.make_targets(sample, T_global2ego)

        return {
            "imgs": imgs,
            "rots": rots,
            "trans": trans,
            "intrins": intrins,
            "post_rots": post_rots,
            "post_trans": post_trans,
            "targets": targets,
            "grid": self.grid_data,
            "token": token,
        }