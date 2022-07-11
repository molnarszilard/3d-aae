import torch.utils.data as data
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import Resize, Compose, ToTensor
import torch, time, os
import random
import cv2
import open3d as o3d

class DatasetLoaderPCD(data.Dataset):
    
    def __init__(self, root='./datasets/', seed=None, train=True, classes=None):
        np.random.seed(seed)
        self.root = Path(root)

        if train:
            self.input_paths = [root+'train/'+d for d in os.listdir(root+'train')]
            # Randomly choose 50k images without replacement
            # self.rgb_paths = np.random.choice(self.rgb_paths, 4000, False)
        else:
            self.input_paths = [root+'test/'+d for d in os.listdir(root+'test/')]
            # self.rgb_paths = np.random.choice(self.rgb_paths, 1000, False)
        
        self.length = len(self.input_paths)
            
    def __getitem__(self, index):
        path = self.input_paths[index]
        pcd = o3d.io.read_point_cloud(path)
        pcd = torch.from_numpy(np.moveaxis(np.array(pcd.points).astype(np.float32),-1,0))
        pcd = pcd - pcd.min()
        pcd_normalized = pcd/torch.abs(pcd).max()
        return pcd_normalized
    def __len__(self):
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = DatasetLoaderPCD()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
