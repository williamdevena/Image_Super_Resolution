import logging
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from modules import UNet
from torchvision import transforms
from tqdm import tqdm

from models import esrgan
from src import super_resolution_dataset
from utils import costants


def train(num_epochs, device, run_name):
    generator = esrgan.Generator(upsample_algo="nearest")
    optimizer = optim.Adam(generator.parameters())
    loss_fn = nn.L1Loss()


    data_transforms = transforms.Compose([
    transforms.RandomCrop(200, 200),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.ToTensor()
    ]
)


    train_dataset = super_resolution_dataset.SuperResolutionDataset(
        hr_path=costants.ORIGINAL_DS_TRAIN,
        lr_path=costants.LR_TRAIN,
        transform=data_transforms
    )
    val_dataset = super_resolution_dataset.SuperResolutionDataset(
        hr_path=costants.ORIGINAL_DS_VAL,
        lr_path=costants.LR_VAL,
        transform=data_transforms
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    train_only_generator(
        model=generator,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        device=device,
        run_name=run_name,
    )




def train_only_generator(model,
                         optimizer,
                         loss_fn,
                         train_dataloader,
                         val_dataloader,
                         num_epochs,
                         device,
                         run_name):
    """
    Pre-trains a "PSNR-based" Generator for the first stage of
    the training.

#     Args:
#         model (_type_): _description_
#         train_dataloader (_type_): _description_
#         optimizer (_type_): _description_
#         loss_fn (_type_): _description_
#         num_epochs (_type_): _description_
#         device (_type_): _description_
#     """
    model.train()
    model.to(device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    len_dataloader = len(train_dataloader)

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        tot_loss = 0

        for idx, (hr_images, lr_images) in enumerate(pbar):
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)
            fake_hr_images = model(lr_images)

            loss = loss_fn(hr_images, fake_hr_images)
            tot_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        tot_loss /= len_dataloader
        logger.add_scalar("LOSS FUNCTION", tot_loss, global_step=epoch)

        if epoch%10==9:
            validation(model=model, loss_fn=loss_fn, val_dataloader=val_dataloader,
                       device=device, logger=logger, epoch=epoch)

        #torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))



def validation(model,
               loss_fn,
               val_dataloader,
               device,
               logger,
               epoch):
    """
    One iteration of validation.

    Args:
        model (_type_): _description_
        loss_fn (_type_): _description_
        val_dataloader (_type_): _description_
        device (_type_): _description_
        logger (_type_): _description_
    """

    len_dataloader = len(val_dataloader)
    pbar = tqdm(val_dataloader)
    tot_loss = 0
    for idx, (hr_images, lr_images) in enumerate(pbar):
        hr_images = hr_images.to(device)
        lr_images = lr_images.to(device)

        fake_hr_images = model(lr_images)

        loss = loss_fn(hr_images, fake_hr_images)
        tot_loss += loss.item()

        pbar.set_postfix(MSE=loss.item())

    tot_loss /= len(val_dataloader)
    logger.add_scalar("LOSS FUNCTION", tot_loss, global_step=epoch)











# def train_ddpm(ddpm_model, dataloader, optimizer, loss_fn, num_epochs, device, run_name):
#     """_summary_

#     Args:
#         model (_type_): _description_
#         dataloader (_type_): _description_
#         optimizer (_type_): _description_
#         loss_fn (_type_): _description_
#         num_epochs (_type_): _description_
#         device (_type_): _description_
#     """
#     ddpm_model.to(device)
#     logger = SummaryWriter(os.path.join("runs", run_name))
#     len_dataloader = len(dataloader)

#     for epoch in range(num_epochs):
#         logging.info(f"Starting epoch {epoch}:")
#         pbar = tqdm(dataloader)
#         for idx, (hr_images, lr_images) in enumerate(pbar):
#             hr_images = hr_images.to(device)
#             lr_images = lr_images.to(device)
#             t = ddpm_model.sample_timesteps(hr_images.shape[0]).to(device)
#             x_t, noise = ddpm_model.noise_images(hr_images, t)
#             predicted_noise = ddpm_model(x_t, t)
#             loss = loss_fn(noise, predicted_noise)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             pbar.set_postfix(MSE=loss.item())
#             logger.add_scalar("LOSS FUNCTION", loss.item(), global_step=epoch * len_dataloader + idx)

#         sampled_images = ddpm_model.sample(ddpm_model, n=hr_images.shape[0])
#         #save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
#         #torch.save(ddpm_model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))


# def launch():
#     import argparse
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.run_name = "DDPM_Uncondtional"
#     args.epochs = 500
#     args.batch_size = 12
#     args.image_size = 64
#     args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
#     args.device = "cuda"
#     args.lr = 3e-4
#     train(args)