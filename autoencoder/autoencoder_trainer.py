import random
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from config import *

matplotlib.use('Agg')

class AutoEncoderTrainer(): 
    def __init__(self, model, dataset : Dataset, batch_size=BATCH_SIZE, model_name="default-model"): 
        super(AutoEncoderTrainer, self).__init__()
        self.model = model
        self.model_name = model_name

        self.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.save_path = f"{SAVE_PATH}/{self.model_name}"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(f"{self.save_path}/logs")

    def save_output_ground_truth(self, output, ground_truth, filename):
        output = output.cpu().numpy()
        ground_truth = ground_truth.cpu().numpy()

        output = output * 0.5 + 0.5
        ground_truth = ground_truth * 0.5 + 0.5

        f, [ax1, ax2] = plt.subplots(1, 2, figsize=(32, 10))
        ax1.imshow(ground_truth[0].transpose(1, 2, 0))
        ax2.imshow(output[0].transpose(1, 2, 0))
        plt.savefig(f"{self.save_path}/logs/{filename}")
        plt.close()

    def validate_one_image(self, epoch):
        for data in self.data_loader:
            input = data.cuda()
            output = self.model(input)
            loss = self.model.loss_function(input, output)
            self.save_output_ground_truth(output, input, f"epoch_{epoch}.png")
            return loss

    def train_one_epoch(self):
        total_loss = 0
        for data in self.data_loader:
            input = data.cuda()
            output = self.model(input)
            loss = self.model.loss_function(input, output)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        avg_loss = total_loss / len(self.data_loader)
        return avg_loss

    def train(self):
        self.model.cuda()
        self.model.train()

        best_loss = 1e9
        for epoch in range(NUM_EPOCHS):
            avg_loss = self.train_one_epoch()
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}/{NUM_EPOCHS},  Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), f"{self.save_path}/best.pt")

        torch.save(self.model.state_dict(), f"{self.save_path}/last.pt")
        
    def eval(self): 
        self.model.load_state_dict(torch.load(f"{self.save_path}/best.pt"))
        self.model.cuda()
        self.model.eval()

        total_loss = 0
        curr_iter = 0
        with torch.no_grad():
            for data in self.data_loader:
                input = data.cuda()
                output = self.model(input)
                loss = self.model.loss_function(input, output)
                total_loss += loss.item()

                if curr_iter % 10 == 0:
                    self.save_output_ground_truth(output, input, f"output_{curr_iter}.png")
                curr_iter += 1
        
        avg_loss = total_loss / len(self.data_loader)
        print(f"Average test loss: {avg_loss:.4f}")