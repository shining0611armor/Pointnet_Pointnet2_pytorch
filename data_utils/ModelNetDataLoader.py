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


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

# we  changed this class for h5  version of modelnet40_normal_resampled


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        # Load the H5 file directly
        if self.num_category == 10:
            self.h5_file = os.path.join(self.root, 'modelnet10_%s.h5' % split)
        else:
            self.h5_file = os.path.join(self.root, 'modelnet40_%s.h5' % split)

        # Open H5 dataset and read data/labels
        with h5py.File(self.h5_file, 'r') as f:
            self.data = f['data'][:]  # Point cloud data
            self.labels = f['label'][:]  # Corresponding labels

        # Subtract 1 from labels to shift the range from [1, 40] to [0, 39]
        self.labels = self.labels - 1

        print(f"The size of {split} data is {len(self.data)}")

        if self.process_data:
            # Optional: process and save data for faster future loading
            if self.uniform:
                self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
            else:
                self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

            if not os.path.exists(self.save_path):
                print(f'Processing data {self.save_path} (only running the first time)...')
                self.list_of_points = []
                self.list_of_labels = []
                for index in tqdm(range(len(self.data)), total=len(self.data)):
                    point_set = self.data[index]
                    label = self.labels[index]

                    # If uniform sampling is enabled
                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[:self.npoints, :]

                    self.list_of_points.append(point_set)
                    self.list_of_labels.append(label)

                # Save processed data
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print(f'Loading processed data from {self.save_path}...')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        if self.process_data:
            point_set = self.list_of_points[index]
            label = self.list_of_labels[index]
        else:
            point_set = self.data[index]
            label = self.labels[index]

            # Uniform sampling if enabled
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[:self.npoints, :]

        # Normalize the point set
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.use_normals:
            point_set = point_set[:, 0:3]  # Use only XYZ coordinates

        return point_set, label

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
