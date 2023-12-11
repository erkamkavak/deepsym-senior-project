import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
import h5py

from config import TRAIN_TEST_SPLIT

class DatasetLoader(Dataset):
    def __init__(self, dataset_path, is_train=True):
        self.dataset_path = dataset_path
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")

        self.hdf5_file = h5py.File(self.dataset_path, 'r')
        self.timestamps = self.hdf5_file["timestamps"]['timestamps']

        if is_train:
            self.timestamps = self.timestamps[:int(len(self.timestamps) * TRAIN_TEST_SPLIT)]
        else: 
            self.timestamps = self.timestamps[int(len(self.timestamps) * TRAIN_TEST_SPLIT):]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        timestamp = str(self.timestamps[idx])
        current_state = np.array(self.hdf5_file['current_state'][timestamp])
        action = np.array(self.hdf5_file['current_state'][timestamp])
        next_state = np.array(self.hdf5_file['next_state'][timestamp])

        current_state = self.transform(current_state)
        next_state = self.transform(next_state)

        return current_state, action, next_state
    
if __name__ == "__main__": 
    from config import DATASET_PATH
    dataset = DatasetLoader(DATASET_PATH)
    for data in dataset:
        cv2.imshow('image', data)
        cv2.waitKey(1)