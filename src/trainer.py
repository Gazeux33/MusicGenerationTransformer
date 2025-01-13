import torch
from tqdm import tqdm
import json

import os

from src.model import MusicModel


class MusicTrainer:
    def __init__(self,
                 model:MusicModel,
                 train_loader,
                 optimizer,
                 loss_fn,
                 device,
                 note_vocab_size,
                duration_vocab_size,
                 save_freq=10,
                 checkpoint_dir=None,
                 remove_previous_models=True
                 ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.note_vocab_size = note_vocab_size
        self.duration_vocab_size = duration_vocab_size
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.remove_previous_models = remove_previous_models
        self.epoch = 0

        # Load the checkpoint
        if checkpoint_dir is not None:
            self._load_checkpoint()

    def train(self,max_epochs):
        while self.epoch < max_epochs:

            # create a progress bar
            bar = tqdm(self.train_loader,total=len(self.train_loader))
            bar.set_description(f"Epoch {self.epoch}")


            for x_note, x_duration, y_note, y_duration in bar:  # [256, 50],[256, 50],[256, 50],[256, 50]

                # Move the data to the device
                x_note, x_duration, y_note, y_duration = self._move_to_device(x_note, x_duration, y_note, y_duration)

                # Process the batch
                loss_note, loss_duration, total_loss_batch = self._process_batch(x_note, x_duration, y_note, y_duration)

                # Update the progress bar
                bar.set_postfix(note=loss_note.item(), duration=loss_duration.item(), total=total_loss_batch.item())

            # Save the model
            if self.epoch % self.save_freq == 0:
                self._save_checkpoint(f"checkpoint_{self.epoch}.pt")

            # increment the epoch
            self.epoch += 1

    def _process_batch(self,x_note,x_duration,y_note,y_duration):
        # clear gradients
        self.optimizer.zero_grad()

        # Prédiction
        out_note, out_duration, att_score = self.model(x_note,x_duration)  # [256, 50, 59],[256, 50, 24],[256, 50, 50]

        # Calculate Loss
        out_note = out_note.view(-1, self.note_vocab_size)  # [256*50, 59]
        out_duration = out_duration.view(-1, self.duration_vocab_size)  # [256*50, 24]
        y_note = y_note.view(-1)  # [256*50]
        y_duration = y_duration.view(-1)  # [256*50]
        loss_note = self.loss_fn(out_note, y_note)
        loss_duration = self.loss_fn(out_duration, y_duration)
        total_loss_batch = loss_note + loss_duration

        # Backpropagation
        total_loss_batch.backward()
        self.optimizer.step()
        return loss_note, loss_duration, total_loss_batch


    def _load_checkpoint(self):
        # check if the checkpoint directory exists
        if os.path.exists(os.path.join(self.checkpoint_dir,"train.json")):

            # load the checkpoint
            with open(os.path.join(self.checkpoint_dir,"train.json"),"r") as f:
                train_state = json.load(f)
            path = str(os.path.join(self.checkpoint_dir,train_state["model_path"]))

            # check if the model exists
            if os.path.exists(path):
                # load state
                self.model.load_state_dict(torch.load(path,weights_only=True))
                self.epoch = train_state["epoch"]
                print("Checkpoint loaded from ",train_state["model_path"])




    def _save_checkpoint(self,checkpoint_path):
        # Create the checkpoint directory if it does not exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Prepare the state
        train_state = {
            "model_path":checkpoint_path,
            "epoch":self.epoch,
        }

        # Save the model and data
        torch.save(self.model.state_dict(),str(os.path.join(self.checkpoint_dir,checkpoint_path)))
        with open(os.path.join(self.checkpoint_dir,"train.json"),"w+") as f:
            json.dump(train_state,f,indent=2)

        # Remove previous models
        if self.remove_previous_models:
            if os.path.exists(os.path.join(self.checkpoint_dir, f"checkpoint_{self.epoch - 1}.pt")):
                os.remove(os.path.join(self.checkpoint_dir, f"checkpoint_{self.epoch - 1}.pt"))

    def _move_to_device(self,*data):
        return (x.to(self.device) for x in data)

