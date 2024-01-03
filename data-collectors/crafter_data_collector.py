import gym
import crafter

from base_data_collector import BaseDataCollector

import matplotlib.pyplot as plt 
import numpy as np

def parse_semantic_info_to_masks(semantic_info, image): 
  unique_ids = np.unique(semantic_info)
  masks = [np.zeros(image.shape[:2], dtype=np.uint8) for _ in unique_ids]

  for i, unique_id in enumerate(unique_ids):
    masks[i][semantic_info == unique_id] = 1

  return masks

def apply_masks(image, masks): 
    segmented_images = [image * mask[:, :, np.newaxis] for mask in masks]
    return segmented_images


class CrafterDataCollector(BaseDataCollector): 
  def __init__(self, model_name) -> None:
    super().__init__(model_name)
    self.env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
    self.env = crafter.Recorder(
      self.env, f"{self.DATASET_DIR}/env-recording",
      save_stats=True,
      save_video=False,
      save_episode=True,
    )
  
  def collect_all(self): 
    # self.collect_all_with_gym()
    if self.env == None: 
        raise Exception("Gym environment is not defined")
    while self.current_batch_id < self.TOTAL_BATCH_COUNT:
        obs = self.env.reset()
        done = False

        action = self.env.action_space.sample()
        obs, reward, done, info = self.env.step(action)
        # semantic_info = info["local_semantic"]
        # masks = parse_semantic_info_to_masks(semantic_info, obs)
        # segmented_images = apply_masks(obs, masks)
        # current_state = np.stack(segmented_images, axis=0)
        current_state = obs

        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, len(segmented_images) + 1, 1)
        # plt.imshow(obs)
        # plt.title("Original Image")

        # for i, segmented_image in enumerate(segmented_images):
        #     plt.subplot(1, len(segmented_images) + 1, i + 2)
        #     plt.imshow(segmented_image)
        #     plt.title(f"Segment {i + 1}")

        # plt.show()

        interval = 0
        while not done and self.current_batch_id < self.TOTAL_BATCH_COUNT:
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)

            if interval < 5: 
               interval += 1
               current_state = obs
               continue
            # semantic_info_after_action = info["local_semantic"]
            # masks = parse_semantic_info_to_masks(semantic_info_after_action, obs)
            # segmented_images = apply_masks(obs, masks)
            # state_after_action = np.stack(segmented_images, axis=0)
            state_after_action = info["view_based_on_prev_pos"]

            self.save_one_batch(current_state, action, state_after_action)
            # current_state = state_after_action
            current_state = obs
            interval = 0
    self.close_and_save()