'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import os
import h5py
import numpy as np
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid  # Center the point cloud
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m  # Scale the point cloud
    return pc

class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False, data_augmentation=True):
        self.root = root
        self.npoints = 1024  # Set number of points to 1024
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.split = split
        self.data_augmentation = data_augmentation
        self.data, self.labels = None, None

        # Load dataset based on the split (train/test)
        self.data, self.labels = self.load_data(split)

        # List of class indices
        self.classes = {i: i for i in range(40)}  # ModelNet40 has 40 classes

    def load_data(self, split):
        """Load ModelNet data from the extracted h5 files."""
        all_data = []
        all_label = []
        partition_file = 'train_files.txt' if split == 'train' else 'test_files.txt'
        partition_file_path = os.path.join(self.root, partition_file)

        # Read file paths from train_files.txt or test_files.txt
        with open(partition_file_path, 'r') as f:
            h5_files = [line.strip() for line in f.readlines()]

        # Load all .h5 files listed in the partition file
        for h5_file in h5_files:
            h5_file_path = os.path.join(self.root, h5_file)  # Construct full path to the .h5 file
            if not os.path.exists(h5_file_path):
                raise FileNotFoundError(f"H5 file not found: {h5_file_path}")
            
            with h5py.File(h5_file_path, 'r') as f:
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
            all_data.append(data)
            all_label.append(label)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    def __getitem__(self, index):
        point_set = self.data[index]
        cls = self.labels[index]

        # Randomly select points (npoints=1024)
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # Normalize the point set
        point_set = pc_normalize(point_set)

        # Apply data augmentation if enabled
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # Random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # Random jitter

        # Normalize XYZ if not using normals
        if not self.use_normals:
            point_set = point_set[:, 0:3]  # Use only XYZ

        point_set = torch.from_numpy(point_set.astype(np.float32))
        
        # Ensure cls is a 1D tensor (avoid multi-target issue)
        cls = torch.tensor(cls).squeeze().long()  # Making sure it's a scalar if it's a single label
        return point_set, cls

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
