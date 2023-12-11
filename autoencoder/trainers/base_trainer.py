import random
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import abc

from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, LEARNING_RATE, SAVE_PATH, NUM_EPOCHS

matplotlib.use('Agg')

class BaseTrainer(): 
    __metaclass__ = abc.ABCMeta

    def __init__(self, model, dataset : Dataset, batch_size=BATCH_SIZE, model_name="default-model"): 
        super(BaseTrainer, self).__init__()
        self.model = model
        self.model_name = model_name

        self.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        # self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.save_path = f"{SAVE_PATH}/{self.model_name}"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(f"{self.save_path}/logs")
            os.makedirs(f"{self.save_path}/logs/val")
            os.makedirs(f"{self.save_path}/logs/test")

    def save_output_ground_truth(self, output, ground_truth, filename, val=False):
        output = output.detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy()

        output = output * 0.5 + 0.5
        ground_truth = ground_truth * 0.5 + 0.5

        f, [ax1, ax2] = plt.subplots(1, 2, figsize=(32, 10))
        ax1.imshow(ground_truth[0].transpose(1, 2, 0))
        ax2.imshow(output[0].transpose(1, 2, 0))
        folder = "val" if val else "test"
        plt.savefig(f"{self.save_path}/logs/{folder}/{filename}")
        plt.close()

    @abc.abstractmethod
    def pass_batch_from_model(self, batch): 
        # input = data.cuda()
        # output = self.model(input)
        # loss = self.model.loss_function(input, output)
        return

    @abc.abstractmethod
    def save_batch_logs(self, batch, save_path, curr_iter, val=False): 
        # if type(output) == tuple:
        #     self.model.save_other_outputs(output, f"{self.save_path}/logs/test/", f"output_{curr_iter}")
        #     output = output[0]
        # self.save_output_ground_truth(output, input, f"output_{curr_iter}.png")
        pass

    def validate_one_image(self, epoch):
        for batch in self.data_loader:
            loss = self.pass_batch_from_model(batch)

            self.save_batch_logs(batch, save_path="{self.save_path}/logs/val/", curr_iter=epoch, val=True)
            return loss

    def train_one_epoch(self):
        total_loss = 0
        for batch in self.data_loader:
            loss = self.pass_batch_from_model(batch)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # self.lr_scheduler.step()
        
        avg_loss = total_loss / len(self.data_loader)
        return avg_loss

    def train(self):
        self.model.cuda()
        self.model.train()

        best_loss = 1e9
        for epoch in range(NUM_EPOCHS):
            avg_loss = self.train_one_epoch()
            if epoch % 10 == 0:
                self.validate_one_image(epoch)
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
            for batch in self.data_loader:
                loss = self.pass_batch_from_model(batch)
                total_loss += loss.item()

                if curr_iter % 10 == 0:
                    self.save_batch_logs()
                curr_iter += 1
        
        avg_loss = total_loss / len(self.data_loader)
        print(f"Average test loss: {avg_loss:.4f}")