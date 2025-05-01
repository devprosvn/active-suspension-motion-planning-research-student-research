import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class CarlaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Look for CSV in the provided directory
        csv_path = os.path.join(data_dir, 'driving_log.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"driving_log.csv not found in {data_dir}")

        self.df = pd.read_csv(csv_path)

        # 🔧 Strip leading "output/" if present in relative paths
        self.df['rgb_path'] = self.df['rgb_path'].apply(
            lambda x: x.replace("output/", "", 1) if x.startswith("output/") else x
        )
        self.df['depth_path'] = self.df['depth_path'].apply(
            lambda x: x.replace("output/", "", 1) if x.startswith("output/") else x
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # ✅ Use correct relative paths (e.g., rgb/xyz.jpg, depth/xyz.png)
        rgb_path = os.path.join(self.data_dir, self.df.iloc[idx]['rgb_path'])
        depth_path = os.path.join(self.data_dir, self.df.iloc[idx]['depth_path'])

        # Load images
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path)

        # Get controls and IMU data
        controls = torch.tensor([
            self.df.iloc[idx]['steering'],
            self.df.iloc[idx]['throttle'],
            self.df.iloc[idx]['brake']
        ], dtype=torch.float32)

        imu = torch.tensor([
            self.df.iloc[idx]['accel_x'],
            self.df.iloc[idx]['accel_y'],
            self.df.iloc[idx]['accel_z'],
            self.df.iloc[idx]['gyro_x'],
            self.df.iloc[idx]['gyro_y'],
            self.df.iloc[idx]['gyro_z']
        ], dtype=torch.float32)

        # Apply transforms if any
        if self.transform:
            rgb = self.transform(rgb)
            depth = self.transform(depth)

        # Convert to tensors and normalize
        rgb = torch.from_numpy(np.array(rgb)).float().permute(2, 0, 1) / 255.0
        depth = torch.from_numpy(np.array(depth)).float().unsqueeze(0) / 255.0

        return rgb, depth, imu, controls