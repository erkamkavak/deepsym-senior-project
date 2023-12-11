import gym
# import minerl
from base_data_collector import BaseDataCollector


class MinerlDataCollector(BaseDataCollector): 
  def __init__(self, model_name) -> None:
    super().__init__(model_name)
    self.env = gym.make('MineRLBasaltFindCave-v0')

  def collect_all(self): 
    self.collect_all_with_gym()