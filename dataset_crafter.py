import gym
import crafter
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import os

from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader

class AutoEncoderDataset(Dataset):
  def __init__(self, image_dir, transform=None):
    self.image_dir = image_dir
    self.image_filenames = os.listdir(image_dir)
    self.transform = transform

  def __len__(self):
      return len(self.image_filenames)
  
  def __getitem__(self, index):
      img_path = os.path.join(self.image_dir, self.image_filenames[index])
      image = Image.open(img_path).convert('RGB')

      if self.transform:
          image = self.transform(image)

      # For an autoencoder, the label is the same as the input
      return image, image
    
def crafter_record_runs(output_path, number_of_simulations):
  env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
  env = crafter.Recorder(
    env, './deepsym-senior-project/data/records',
    save_stats=True,
    save_video=False,
    save_episode=True,
  )


  for _ in range(number_of_simulations):
      obs = env.reset()
      done = False

      while not done:
          action = env.action_space.sample()
          obs, reward, done, info = env.step(action)
          if done:
              print(f"Record {_} is done. Restarting the simulation...")
              break
          
# Function to extract images from npz files
def extract_images_from_npz(path_to_npz, path_to_output_file, experiment_name, step_size):
    # Check if the directory exists
    """
    if not os.path.exists(os.path.join(path_to_output_file, experiment_name)):
        # If not, create the directory
        os.makedirs(os.path.join(path_to_output_file, experiment_name))
        print(f"Directory {experiment_name} created")
    else:
        raise FileExistsError(f"There is another experiment named {experiment_name}, Please choose another name.")
    """
    try:
      global counter
    except:
        raise NameError("A global Counter should be defined.")
    
    with np.load(path_to_npz) as data:
        images = data['image']

    for i in range(0, len(images), step_size):
        # Create a figure with specified size to match the image's size
        fig, ax = plt.subplots(figsize=(64/80, 64/80), dpi=80)  # Here, dpi=80 is just a typical screen resolution

        # Remove axes
        ax.axis('off')

        # Display the image data
        ax.imshow(images[i], interpolation='none')

        # Save the image in a lossless format (e.g., PNG)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #fig.savefig(os.path.join(path_to_output_file, experiment_name,f"{i}.png"), dpi=80, bbox_inches='tight', pad_inches=0, transparent=True) # Different experiments into different folders
        fig.savefig(os.path.join(path_to_output_file, "all_data",f"{counter}.png"), dpi=80, bbox_inches='tight', pad_inches=0, transparent=True) # Different experiments into same folder
        plt.close(fig)
        counter += 1

# Function to extract images from multiple crafter recordings
def extract_images_from_records_file(path_to_records, path_to_output_file, step_size):
    records = os.listdir(path_to_records)

    for record in records:
        extract_images_from_npz(os.path.join(path_to_records, record), path_to_output_file, record, step_size) 

if __name__ == "__main__":
  # Specify the transformations
  transform = transforms.Compose([
      transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
      # Add any other transformations if needed
  ])

  # Create datasets
  dataset_path = r"C:\\Users\\EmirKISA\\Desktop\\Projects\\Symbolic Learning\\deepsym-senior-project\\data\\images\\all_data"
  dataset = AutoEncoderDataset(image_dir=dataset_path, transform=transform)

  # Create dataloaders
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        
