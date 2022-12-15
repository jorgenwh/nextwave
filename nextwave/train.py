import os
import random

import numpy as np
import torch
from tqdm import tqdm

from .util import AverageMeter

DEFAULT_LOSS_FUNCTION = torch.nn.MSELoss
DEFAULT_OPTIMIZER = torch.optim.SGD

def build_random_batch(training_examples, batch_size):
    indices = np.random.randint(len(training_examples), size=batch_size)
    inputs, targets = list(zip(*[training_examples[i] for i in indices]))
    return inputs, targets

def train_network_on_audio(
        model, training_examples, 
        loss_function=None, optimizer=None, 
        iters_per_epoch=10000, epochs=10, learning_rate=0.01, batch_size=4, cuda=False, 
        output_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
    model = model.to(device)

    if loss_function is None:
        loss_function = DEFAULT_LOSS_FUNCTION()
    if optimizer is None:
        optimizer = DEFAULT_OPTIMIZER(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"EPOCH: {epoch+1}")
        epoch_loss = AverageMeter()

        bar = tqdm(
                range(iters_per_epoch),
                desc="Training", 
                bar_format="{l_bar}{bar}| Update: {n_fmt}/{total_fmt} - {unit} - Elapsed: {elapsed}")
        for i in bar:
            model.zero_grad()

            inputs, targets = build_random_batch(training_examples, batch_size)

            inputs = torch.tensor(inputs).reshape(batch_size, -1, 1).to(device)
            targets = torch.tensor(targets).reshape(batch_size, -1, 1).to(device)

            y = model(inputs)
            
            loss = loss_function(y, targets)

            epoch_loss.update(loss.item())
            bar.unit = f"Loss: {epoch_loss}"

            loss.backward()
            optimizer.step()

        if output_dir:
            torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_{epoch+1}"))

