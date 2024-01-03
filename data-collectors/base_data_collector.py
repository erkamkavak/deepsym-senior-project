import numpy as np
import h5py
from tqdm import tqdm
import os
import gym

class BaseDataCollector: 
    def __init__(self, model_name) -> None:
        self.TOTAL_BATCH_COUNT = 15000
        self.DATASET_DIR = os.path.normpath(os.path.join(os.getcwd(), '../datasets/'))

        self.current_saved_image_count = 0
        self.p_bar = tqdm(range(self.TOTAL_BATCH_COUNT))

        self.save_h5f_file = h5py.File(f'{self.DATASET_DIR}/{model_name}.hdf5', 'w')
        self.timestamp_group = self.save_h5f_file.create_group("timestamps")
        self.current_state_group = self.save_h5f_file.create_group("current_state")
        self.action_group = self.save_h5f_file.create_group("action")
        self.next_state_group = self.save_h5f_file.create_group("next_state")

        self.current_batch_id = 0
        self.env = None

    def save_one_batch(self, current_state, action, next_state): 
        current_batch = str(self.current_batch_id)

        self.current_state_group.create_dataset(current_batch, data=np.asarray(current_state))
        self.action_group.create_dataset(current_batch, data=np.asarray(action))
        self.next_state_group.create_dataset(current_batch, data=np.asarray(next_state))

        self.p_bar.update(1)
        self.p_bar.refresh()
        self.current_batch_id += 1
    
    def collect_all(): 
        pass
    
    def collect_all_with_gym(self): 
        if self.env == None: 
            raise Exception("Gym environment is not defined")
        while self.current_batch_id < self.TOTAL_BATCH_COUNT:
            obs = self.env.reset()
            done = False

            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            current_state = obs

            interval = 0
            while not done and self.current_batch_id < self.TOTAL_BATCH_COUNT:
                action = self.env.action_space.sample()
                obs, reward, done, info = self.env.step(action)
                if interval < 5: 
                    interval += 1
                    current_state = obs
                    continue

                state_after_action = obs
                self.save_one_batch(current_state, action, state_after_action)
                current_state = state_after_action
                interval = 0
        self.close_and_save()

    def close_and_save(self): 
        self.timestamp_group.create_dataset("timestamps", data=np.arange(0, self.current_batch_id))        
        self.save_h5f_file.close()