import logging
import os

import torch
import torch.nn as nn
#from modules import UNet
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(ddpm_model, dataloader, optimizer, loss_fn, num_epochs, device, run_name):
    """_summary_

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
        optimizer (_type_): _description_
        loss_fn (_type_): _description_
        num_epochs (_type_): _description_
        device (_type_): _description_
    """
    ddpm_model.to(device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    len_dataloader = len(dataloader)

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for idx, (hr_images, lr_images) in enumerate(pbar):
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)
            t = ddpm_model.sample_timesteps(hr_images.shape[0]).to(device)
            x_t, noise = ddpm_model.noise_images(hr_images, t)
            predicted_noise = ddpm_model(x_t, t)
            loss = loss_fn(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("LOSS FUNCTION", loss.item(), global_step=epoch * len_dataloader + idx)

        sampled_images = ddpm_model.sample(ddpm_model, n=hr_images.shape[0])
        #save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        #torch.save(ddpm_model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)