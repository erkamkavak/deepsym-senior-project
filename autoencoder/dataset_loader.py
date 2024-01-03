import torch
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
            # transforms.Normalize(mean=[0.1147, 0.3680, 0.1231], std=[0.1498, 0.2630, 0.1801])
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def calc_mean_and_std(self): 
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        for batch in self:
            image = batch[0]
            c, h, w = image.shape
            nb_pixels = h * w
            sum_ = torch.sum(image, dim=[1, 2])
            sum_of_square = torch.sum(image ** 2,
                                    dim=[1, 2])
            fst_moment = (cnt * fst_moment + sum_) / (
                        cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (
                                cnt + nb_pixels)
            cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)        
        return mean,std
        
    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        timestamp = str(self.timestamps[idx])
        current_state = np.array(self.hdf5_file['current_state'][timestamp])
        action = np.array(self.hdf5_file['action'][timestamp])
        next_state = np.array(self.hdf5_file['next_state'][timestamp])

        # current_state = cv2.resize(current_state, (640, 640))
        # next_state = cv2.resize(next_state, (640, 640))

        current_state = self.transform(current_state)
        next_state = self.transform(next_state)

        return current_state, action, next_state
    
if __name__ == "__main__": 
    from config import DATASET_PATH

    dataset_name = "crafter-15000"
    dataset_path = f"{DATASET_PATH}/{dataset_name}.hdf5"
    dataset = DatasetLoader(dataset_path)
    mean, std = dataset.calc_mean_and_std()
    print(mean, std)