import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss

class Trainer:
    def __init__(
        self, 
        train_dataloader, 
        valid_dataloader, 
        model, 
        optimizer, 
        loss_fn, 
        val_loss_fn, 
        scheduler, 
        device="cuda:0", 
        plot_results=True):
        """
        Constructor for Trainer class
        """
        self.train = train_dataloader
        self.valid = valid_dataloader
        self.optim = optim
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn
        self.scheduler = scheduler
        self.device = device
        self.plot_results = plot_results
    
    def train_one_cycle(self):
        """
        Runs one epoch of training, backpropagation and optimization
        """
        model.train()
        train_prog_bar = tqdm(self.train, total=len(self.train))

        all_train_labels = []
        all_train_preds = []
        
        running_loss = 0
        
        for xtrain, ytrain in train_prog_bar:
            xtrain = xtrain.to(device).float()
            ytrain = ytrain.to(device).float()
            
            with autocast():
                # Get predictions
                z = model(xtrain)

                # Training
                train_loss = self.loss_fn(z, ytrain)
                scaler.scale(train_loss).backward()
                
                scaler.step(self.optim)
                scaler.update()
                self.optim.zero_grad()

                # For averaging and reporting later
                running_loss += train_loss

                # Convert the predictions and corresponding labels to right form
                train_predictions = torch.argmax(z, 1).detach().cpu().numpy()
                train_labels = ytrain.detach().cpu().numpy()

                # Append current predictions and current labels to a list
                all_train_labels += [train_predictions]
                all_train_preds += [train_labels]

            # Show the current loss to the progress bar
            train_pbar_desc = f'loss: {train_loss.item():.4f}'
            train_prog_bar.set_description(desc=train_pbar_desc)
        
        # Now average the running loss over all batches and return
        train_running_loss = running_loss / len(self.train)
        print(f"Final Training Loss: {train_running_loss:.4f}")
        
        # Free up memory
        del all_train_labels, all_train_preds, train_predictions, train_labels, xtrain, ytrain, z
        
        return train_running_loss

    def valid_one_cycle(self):
        """
        Runs one epoch of prediction
        """        
        model.eval()
        
        valid_prog_bar = tqdm(self.valid, total=len(self.valid))
        
        with torch.no_grad():
            all_valid_labels = []
            all_valid_preds = []
            
            running_loss = 0
            
            for xval, yval in valid_prog_bar:
                xval = xval.to(device).float()
                yval = yval.to(device).float()
                
                val_z = model(xval)
                
                val_loss = self.val_loss_fn(val_z, yval)
                
                running_loss += val_loss.item()
                
                val_pred = torch.argmax(val_z, 1).detach().cpu().numpy()
                val_label = yval.detach().cpu().numpy()
                
                all_valid_labels += [val_label]
                all_valid_preds += [val_pred]
            
                # Show the current loss
                valid_pbar_desc = f"loss: {val_loss.item():.4f}"
                valid_prog_bar.set_description(desc=valid_pbar_desc)
            
            # Get the final loss
            final_loss_val = running_loss / len(self.valid)
            
            # Get Validation Accuracy
            all_valid_labels = np.concatenate(all_valid_labels)
            all_valid_preds = np.concatenate(all_valid_preds)
            
            print(f"Final Validation Loss: {final_loss_val:.4f}")
            
            # Free up memory
            del all_valid_labels, all_valid_preds, val_label, val_pred, xval, yval, val_z
            
        return (final_loss_val, model)