import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os

from config import TRAIN_TEST_SPLIT

class DatasetLoader(Dataset):
    def __init__(self, dataset_path, is_train=True):
        self.dataset_path = dataset_path
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")

        self.observations = []
        image_files = os.listdir(dataset_path)
        if len(image_files) == 0:
            raise ValueError(f"Dataset path {dataset_path} is empty")

        if is_train:
            image_files = image_files[:int(len(image_files) * TRAIN_TEST_SPLIT)]
        else: 
            image_files = image_files[int(len(image_files) * TRAIN_TEST_SPLIT):]

        for image in image_files:
            if not image.endswith('.png'):
                continue
            image_path = os.path.join(dataset_path, image)
            image = cv2.imread(image_path)
            self.observations.append(image)
        print(f"Dataset loaded: {len(self.observations)} images")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.transform(self.observations[idx])
    
if __name__ == "__main__": 
    from config import DATASET_PATH
    dataset = DatasetLoader(DATASET_PATH)
    for data in dataset:
        cv2.imshow('image', data)
        cv2.waitKey(1)