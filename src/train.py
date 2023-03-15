import logging
import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from modules import UNet
from torchvision import transforms
from tqdm import tqdm

#from models import esrgan
#from src import super_resolution_dataset
#from utils import costants


def train(start_epoch, num_epochs, device, run_name, weights_folder, batch_size, pretrained, weights=None):
    generator = Generator(upsample_algo="nearest", num_blocks=10)

    if pretrained:
        generator.load_state_dict(torch.load(weights))

    optimizer = optim.Adam(generator.parameters())
    loss_fn = nn.L1Loss()

    HR = 128
    LR = HR//4
    #print(LR)


    transform_both = A.Compose([
        A.RandomCrop(HR, HR)
        ]
    )

    transform_hr = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
        ]
    )

    transform_lr = A.Compose([
        A.Resize(width=LR, height=LR, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
        ]
    )


    train_dataset = SuperResolutionDataset(
        hr_path=ORIGINAL_DS_TRAIN,
        lr_path=LR_TRAIN,
        transform_both=transform_both,
        transform_hr=transform_hr,
        transform_lr=transform_lr
    )
    val_dataset = SuperResolutionDataset(
        hr_path=ORIGINAL_DS_VAL,
        lr_path=LR_VAL,
        transform_both=transform_both,
        transform_hr=transform_hr,
        transform_lr=transform_lr
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    train_only_generator(
        model=generator,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        device=device,
        run_name=run_name,
        weights_folder=weights_folder
    )




def train_only_generator(model,
                         optimizer,
                         loss_fn,
                         train_dataloader,
                         val_dataloader,
                         start_epoch,
                         num_epochs,
                         device,
                         run_name,
                         weights_folder):
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
        pbar.set_description("Training")
        train_tot_loss = 0

        for idx, (hr_images, lr_images) in enumerate(pbar):
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)
            fake_hr_images = model(lr_images)

            #print(hr_images.shape, lr_images.shape, fake_hr_images.shape)

            loss = loss_fn(hr_images, fake_hr_images)
            train_tot_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(L1=loss.item())

        train_tot_loss /= len_dataloader
        pbar.set_postfix(TRAIN_L1=train_tot_loss)
        #logger.add_scalar("LOSS FUNCTIONS", tot_loss, global_step=epoch)

        if epoch%2==0:
            val_tot_loss = validation(model=model, loss_fn=loss_fn, val_dataloader=val_dataloader,
                       device=device, epoch=epoch)

            logger.add_scalars("LOSS FUNCTIONS", {
                'train_loss': train_tot_loss,
                'val_loss': val_tot_loss,
            }, start_epoch+epoch)
            torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{start_epoch+epoch}.pt"))
        else:
            logger.add_scalars("LOSS FUNCTIONS", {
                'train_loss': train_tot_loss,
            }, start_epoch+epoch)

    torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{start_epoch+num_epochs}.pt"))



def validation(model,
               loss_fn,
               val_dataloader,
               device,
               #logger,
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
    pbar.set_description("Validation")
    for idx, (hr_images, lr_images) in enumerate(pbar):
        hr_images = hr_images.to(device)
        lr_images = lr_images.to(device)

        fake_hr_images = model(lr_images)

        loss = loss_fn(hr_images, fake_hr_images)
        tot_loss += loss.item()

        pbar.set_postfix(L1=loss.item())

    tot_loss /= len(val_dataloader)
    #logger.add_scalar("LOSS FUNCTIONS", tot_loss, global_step=epoch)

    return tot_loss












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