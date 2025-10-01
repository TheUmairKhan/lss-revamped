import os
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
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
    rot_lim = (-5.0, 5.0)  # deg
    random_flip = True
    color_jitter = False
    cam_drop_prob = 0.0
    extrinsic_noise_std = 0.0


class GridConfig:
    # X x Y grid: 50m x 50m with 0.5m x 0.5m cells
    # Resulting BEV grid 200 x 200
    xbound = (-50.0, 50.0, 0.5)
    ybound = (-50.0, 50.0, 0.5)
    zbound = (-10.0, 10.0, 20.0)

    def gen_dx_bx(self):
        def p(b):
            lo, hi, step = b
            n = int(round((hi - lo) / step))
            return step, lo, n
        dx, bx, nx = p(self.xbound)
        dy, by, ny = p(self.ybound)
        dz, bz, nz = p(self.zbound)
        # For now, create a simple depth range for depth estimation
        dmin, dmax, dstep = 2.0, 58.0, 0.5
        dbins = np.arange(dmin, dmax + 1e-6, dstep, dtype=np.float32)
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

        # Small caches for static per-camera params
        # keys like (log_token, cam) -> dict(K, T_cam2ego, distortion, etc.)
        self._calib_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get_scenes(self, split):
        """Get scene names for the given split"""
        # TODO: Use nuscenes utils to get train/val/test scenes
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'. Expected 'train','val','test'.")

        splits = create_splits_scenes()

        if "mini" in self.version:
            key_map = {"train": "mini_train", "val": "mini_val"}
            split_key = key_map.get(split, split)  # allow user to pass 'train'/'val'
        
        else:
            split_key = split
        
        scenes = splits[split_key]
        
        present = {s["name"] for s in self.nusc.scene}
        scenes = [n for n in scenes if n in present]
        if not scenes:
            raise RuntimeError(f"No scenes found for split '{split_key}' in version '{self.version}'.")
        return scenes

    def preprocess(self, scenes: List[str]) -> List[str]:
        """
        Traverse scenes (by name) to collect sample tokens in order.
        """
        # Build name -> scene dict once
        name2scene = {s["name"]: s for s in self.nusc.scene}

        sample_tokens: List[str] = []
        for scene_name in scenes:
            scene = name2scene[scene_name]  # scene is a dict, not a token
            tok = scene["first_sample_token"]
            while tok:
                sample = self.nusc.get("sample", tok)
                if all(cam in sample["data"] for cam in self.cams):
                    sample_tokens.append(tok)
                tok = sample["next"]
        print(f"Collected {len(sample_tokens)} sample tokens from {len(scenes)} scenes")
        return sample_tokens


    def get_camera_data(self, sample, camera):
        """Return dict with img_path, K (3x3), T_cam2ego (4x4), timestamp, etc."""
        # TODO: sample -> sample_data for this camera
        # from calibrated_sensor: intrinsics and cam extrinsics wrt ego
        # cache K and T_cam2ego per (log_token, camera)

        sd = self.nusc.get('sample_data', sample['data'][camera])
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ep = self.nusc.get('ego_pose', sd['ego_pose_token'])

        # Intrinsics (3x3)
        K = np.array(cs['camera_intrinsic'], dtype=np.float32)

        # Cam -> Ego (4x4)
        q = Quaternion(cs['rotation'])            # nuScenes stores (w,x,y,z)
        R = q.rotation_matrix.astype(np.float32)  # (3,3)
        t = np.array(cs['translation'], dtype=np.float32)  # (3,)
        T_cam2ego = np.eye(4, dtype=np.float32)
        T_cam2ego[:3, :3] = R
        T_cam2ego[:3,  3] = t
        
        return {
        'img_path': os.path.join(self.dataroot, sd['filename']),
        'K': K,
        'T_cam2ego': T_cam2ego,
        'timestamp': sd['timestamp'],
        'channel': camera
        }


    def sample_augmentation(self, H0, W0):
        """
        Start simple: no rotate/flip; just resize to final_dim.
        Return post_rot (3x3), post_tran (3,), and (H,W) after aug.
        """
        H, W = self.aug_conf.final_dim
        post_rot = np.eye(3, dtype=np.float32)
        post_tran = np.zeros(3, dtype=np.float32)
        return post_rot, post_tran, (H, W)


    def prepare_images(self, cam_infos):
        """
        Load each image, resize to final_dim, normalize to [0,1],
        adjust intrinsics for the resize, and collect extrinsics.
        Returns tensors ready for the model.
        """
        H, W = self.aug_conf.final_dim
        imgs, intrins, rots, trans, post_rots, post_trans = [], [], [], [], [], []

        for info in cam_infos:
            im = Image.open(info["img_path"]).convert("RGB")
            W0, H0 = im.size

            post_rot_np, post_tran_np, (H_out, W_out) = self.sample_augmentation(H0, W0)
            assert (H_out, W_out) == (H, W)

            im = im.resize((W, H), resample=Image.BILINEAR)
            im = np.asarray(im, dtype=np.float32) / 255.0      # (H,W,3)
            im = torch.from_numpy(im).permute(2, 0, 1).contiguous()  # (3,H,W)

            sx, sy = W / float(W0), H / float(H0)
            S = np.array([[sx, 0, 0],
                        [0,  sy, 0],
                        [0,   0, 1]], dtype=np.float32)
            K_aug = (S @ info["K"]).astype(np.float32)

            R = info["T_cam2ego"][:3, :3].astype(np.float32)
            t = info["T_cam2ego"][:3,  3].astype(np.float32)

            # collect
            imgs.append(im)
            intrins.append(torch.from_numpy(K_aug))
            rots.append(torch.from_numpy(R))
            trans.append(torch.from_numpy(t))
            post_rots.append(torch.from_numpy(post_rot_np))
            post_trans.append(torch.from_numpy(post_tran_np))
        
        # stack
        imgs       = torch.stack(imgs, dim=0)         # (N_cam,3,H,W)
        intrins    = torch.stack(intrins, dim=0)      # (N_cam,3,3)
        rots       = torch.stack(rots, dim=0)         # (N_cam,3,3)
        trans      = torch.stack(trans, dim=0)        # (N_cam,3)
        post_rots  = torch.stack(post_rots, dim=0)    # (N_cam,3,3)
        post_trans = torch.stack(post_trans, dim=0)   # (N_cam,3)

        return imgs, rots, trans, intrins, post_rots, post_trans


    def make_targets(self, sample, T_global2ego):
        """
        Return {'occ': (1, ny, nx)} vehicle occupancy in ego BEV.
        Assumes T_global2ego is already computed for this sample.
        """
        dx, dy = self.dx[0], self.dx[1]
        bx, by = self.bx[0], self.bx[1]
        nx, ny = self.nx[0], self.nx[1]

        occ = np.zeros((ny, nx), dtype=np.uint8)  # (rows=y, cols=x)

        R_ge = T_global2ego[:3, :3]
        t_ge = T_global2ego[:3,  3]

        for ann_tok in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_tok)
            if not ann["category_name"].startswith("vehicle."):
                continue

            # global box -> ego frame
            box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
            box.rotate(Quaternion(matrix=R_ge))
            box.translate(t_ge)

            # ground-plane corners (x,y in meters)
            corners = box.bottom_corners()[:2, :].T.astype(np.float32)  # (4,2)

            # meters -> grid indices (x->col, y->row)
            ix = np.floor((corners[:, 0] - bx) / dx + 0.5).astype(np.int32)
            iy = np.floor((corners[:, 1] - by) / dy + 0.5).astype(np.int32)
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)

            poly = np.stack([ix, iy], axis=1).reshape(-1, 1, 2).astype(np.int32)
            if poly.shape[0] >= 3:
                cv2.fillPoly(occ, [poly], 1)

        return {"occ": torch.from_numpy(occ.astype(np.float32)).unsqueeze(0)}


    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single training sample"""
        # 1. Get sample token and load sample data
        # 2. Load images from all cameras
        # 3. Load calibration parameters (intrinsics, extrinsics)
        # 4. Load BEV labels (segmentation maps)
        # 5. Apply data augmentation
        # 6. Return formatted batch

        token = self.sample_tokens[index]
        sample = self.nusc.get("sample", token)
        cam_infos = [self.get_camera_data(sample, cam) for cam in self.cams]
        imgs, rots, trans, intrins, post_rots, post_trans = self.prepare_images(cam_infos)
        
        lidar_tok = sample["data"].get("LIDAR_TOP") or next(iter(sample["data"].values()))
        sd = self.nusc.get("sample_data", lidar_tok)
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        T_ego2global = np.eye(4, dtype=np.float32)
        T_ego2global[:3, :3] = Quaternion(ep["rotation"]).rotation_matrix.astype(np.float32)
        T_ego2global[:3,  3] = np.array(ep["translation"], dtype=np.float32)
        T_global2ego = np.linalg.inv(T_ego2global).astype(np.float32)

        targets = self.make_targets(sample, T_global2ego)

        return {
            "imgs": imgs,              # (N_cam, 3, H, W)
            "rots": rots,              # (N_cam, 3, 3) cam->ego
            "trans": trans,            # (N_cam, 3)
            "intrins": intrins,        # (N_cam, 3, 3) adjusted for aug
            "post_rots": post_rots,    # (N_cam, 3, 3) image-space aug
            "post_trans": post_trans,  # (N_cam, 3)
            "targets": targets,        # {} for now
            "grid": self.grid_data,    # {"dx","bx","nx","dbins"} (np arrays)
            "token": token,
        }

    
if __name__ == "__main__":
    # Test with NuscData class initialization
    print("Testing NuscData class initialization:")
    grid_conf = GridConfig()
    aug_conf = AugConfig()
    dataroot = "/Users/umair/BEV/mini"
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    dataset = NuscData(
        nusc=nusc,
        dataroot=dataroot,
        version="v1.0-mini",
        split="train",
        grid_conf=GridConfig(),
        aug_conf=AugConfig()
    )
    
    print(f"Scenes: {len(dataset.scenes)}  -> first 3: {dataset.scenes[:3]}")
    print(f"Samples: {len(dataset.sample_tokens)}")
    print(f"\nTesting camera data for first sample:")

    sample_token = dataset.sample_tokens[0]
    sample = nusc.get("sample", sample_token)
    for camera in dataset.CAMERAS[:2]:  # Test just first 2 cameras   
            cam_data = dataset.get_camera_data(sample, camera)