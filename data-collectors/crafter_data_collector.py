import gym
import crafter

from base_data_collector import BaseDataCollector


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
    self.collect_all_with_gym()