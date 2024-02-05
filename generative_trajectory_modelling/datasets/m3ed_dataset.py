from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torch.utils.data import Dataset


class M3EDDataset(Dataset):

    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path = Path(dataset_path).absolute()
        self.dataset_name = self.dataset_path.stem

        self.data_file = self.dataset_path / f"{self.dataset_name}_data.h5"
        self.pose_gt_file = self.dataset_path / f"{self.dataset_name}_pose_gt.h5"

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> torch.Tensor:
        with h5py.File(self.data_file) as f:
            image = torch.as_tensor(f["ovc"]["rgb"]["data"][index].transpose(2, 0, 1))
            image_t = f["ovc"]["ts"][index]
            
        # NOTE(rahul): The code below does:
        # 1. Takes the image timestep and finds the two closest pose_gt timesteps (above and below)
        # 2. Using the image timestep, linearly interpolate the associated translation vectors
        # 3. Spherically interpolate the rotation vectors 
        with h5py.File(self.pose_gt_file) as f:
            pose_t_index = np.searchsorted(f["ts"], image_t)
            if (pose_t_index == 0 or pose_t_index == len(f["ts"])):
                pose = f["Cn_T_C0"][pose_t_index]
            else:
                time0, time1 = f["ts"][pose_t_index - 1], f["ts"][pose_t_index]
                T0, T1 = f["Cn_T_C0"][pose_t_index - 1], f["Cn_T_C0"][pose_t_index]
                R0R1 = R.from_matrix(np.stack((T0[:3, :3], T1[:3, :3])))
                t0, t1 = T0[:3, 3], T1[:3, 3]

                pose_R = Slerp((time0, time1), R0R1)(image_t)
                pose_t = ((image_t - time0) / (time1 - time0)) * (t1 - t0) + t0

                pose = np.eye(4)
                pose[:3, :3] = pose_R.as_matrix()
                pose[:3, 3] = pose_t
        
        # TODO(rahul): add test cases for image and pose
        return image, torch.as_tensor(pose)