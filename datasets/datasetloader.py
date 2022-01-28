import torch.utils.data as data
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import Resize, Compose, ToTensor
import torch, time, os
import random
import cv2
import open3d as o3d

class DatasetLoader(data.Dataset):
    
    def __init__(self, root='./datasets/', seed=None, train=True, classes=None):
        np.random.seed(seed)
        self.root = Path(root)

        if train:
            self.depth_input_paths = [root+'ModelNet10_pcd/train/'+d for d in os.listdir(root+'ModelNet10_pcd/train')]
            # Randomly choose 50k images without replacement
            # self.rgb_paths = np.random.choice(self.rgb_paths, 4000, False)
        else:
            self.depth_input_paths = [root+'ModelNet10_pcd/test/'+d for d in os.listdir(root+'ModelNet10_pcd/test/')]
            # self.rgb_paths = np.random.choice(self.rgb_paths, 1000, False)
        
        self.length = len(self.depth_input_paths)
            
    def __getitem__(self, index):
        path = self.depth_input_paths[index]
        pcd = o3d.io.read_point_cloud(path)
        pcd = torch.from_numpy(np.moveaxis(np.array(pcd.points).astype(np.float32),-1,0))
        gimgt=cv2.imread(path.replace('ModelNet10_pcd', 'ModelNet10_gim').replace('pcd','jpg'),cv2.IMREAD_UNCHANGED).astype(np.float32)
        # depth_input_mod = np.moveaxis(depth_input,-1,0)
        # gimgt= Compose([Resize((100,100)), ToTensor()])(gimgt)
        gimgt=torch.from_numpy(np.moveaxis(gimgt,-1,0))
        # gimgt=gimgt-gimgt.min()
        # gimgt=gimgt/gimgt.max()
        gimgt=gimgt/255
        gimgt_rgb=gimgt*0
        gimgt_rgb[0]=gimgt[1]
        gimgt_rgb[1]=gimgt[0]
        gimgt_rgb[2]=gimgt[2]
        pcd = pcd - pcd.min()
        pcd_normalized = pcd/torch.abs(pcd).max()-0.5
        return pcd_normalized, gimgt_rgb

    def __len__(self):
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = DatasetLoader()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
