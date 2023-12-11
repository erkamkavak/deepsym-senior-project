import gym
import griddly
from base_data_collector import BaseDataCollector


class GriddlyDataCollector(BaseDataCollector): 
  def __init__(self, model_name) -> None:
    super().__init__(model_name)
    self.env = gym.make('GDY-Sokoban-v0')
  
  def collect_all(self): 
    self.collect_all_with_gym()