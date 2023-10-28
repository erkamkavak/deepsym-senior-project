import gym
import crafter
import numpy as np
import cv2
from tqdm import tqdm

TOTAL_SAVE_IMAGE_COUNT = 2000
LOG_DIR = './crafter-logs/'

current_saved_image_count = 0
p_bar = tqdm(range(TOTAL_SAVE_IMAGE_COUNT))

env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, f"{LOG_DIR}/env-recording",
  save_stats=True,
  save_video=False,
  save_episode=True,
)

observations = []
while current_saved_image_count < TOTAL_SAVE_IMAGE_COUNT:
  obs = env.reset()
  done = False

  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)

  while not done and current_saved_image_count < TOTAL_SAVE_IMAGE_COUNT:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    cv2.imwrite(f'{LOG_DIR}/images/{current_saved_image_count}.png', obs)
    p_bar.update(1)
    p_bar.refresh()
    current_saved_image_count += 1