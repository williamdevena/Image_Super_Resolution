import logging
import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import esrgan
from src import super_resolution_dataset
from utils import costants


def train(start_epoch, num_epochs, device, run_name,
          weights_folder, batch_size, lr, pretrained_gen, num_blocks_gen,
          pretrained_disc=False, weights_gen=None, weights_disc=None):


    generator = esrgan.Generator(upsample_algo="nearest", num_blocks=num_blocks_gen)
    #discriminator = Discriminator()

    if pretrained_gen:
        generator.load_state_dict(torch.load(weights_gen))

    #if pretrained_disc:
     #   discriminator.load_state_dict(torch.load(weights_disc))

    optimizer = optim.Adam(generator.parameters(), lr=lr)
    #optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    loss_fn = nn.L1Loss()


    #loss_fn = nn.BCEWithLogitsLoss()

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


    train_dataset = super_resolution_dataset.SuperResolutionDataset(
        hr_path=costants.ORIGINAL_DS_TRAIN,
        lr_path=costants.LR_TRAIN,
        transform_both=transform_both,
        transform_hr=transform_hr,
        transform_lr=transform_lr
    )
    val_dataset = super_resolution_dataset.SuperResolutionDataset(
        hr_path=costants.ORIGINAL_DS_VAL,
        lr_path=costants.LR_VAL,
        transform_both=transform_both,
        transform_hr=transform_hr,
        transform_lr=transform_lr
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    train_generator(
        model=generator,
       optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        device=device,
        run_name=run_name,
        weights_folder=weights_folder,
        num_blocks_gen=num_blocks_gen
    )

    #train_discriminator(
    #        generator=generator,
    #        discriminator=discriminator,
    #        optimizer=optimizer,
    #        loss_fn=loss_fn,
    #        train_dataloader=train_dataloader,
    #        start_epoch=start_epoch,
    #        num_epochs=num_epochs,
    ##        device=device,
     #       run_name=run_name,
     #       weights_folder=weights_folder
     #   )




def train_discriminator(generator,
                        discriminator,
                        optimizer,
                        loss_fn,
                        train_dataloader,
                        start_epoch,
                        num_epochs,
                        device,
                        run_name,
                        weights_folder):
    generator.eval()
    discriminator.train()
    generator.to(device)
    discriminator.to(device)
    # model.to(device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    len_dataloader = len(train_dataloader)

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Training epoch {start_epoch+epoch}")
        tot_loss = 0

        for idx, (hr_images, lr_images) in enumerate(pbar):
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)
            fake_hr_images = generator(lr_images)
            #fake_hr_images = fake_hr_images.to(device)
            #print("\ngen fatto")
            out_fake = discriminator(fake_hr_images)
            out_real = discriminator(hr_images)

            #print(out_real)
            loss_real = loss_fn(
                out_real, torch.ones_like(out_real)
            )
            #print(loss_real)

            #print(out_fake)
            loss_fake = loss_fn(
                out_fake, torch.zeros_like(out_fake)
            )
            #print(loss_fake)
            loss = loss_fake + loss_real
            #print(hr_images.shape, lr_images.shape, fake_hr_images.shape)
            tot_loss += loss.item()


            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(BCE=loss.item())

        tot_loss /= len_dataloader
        pbar.set_postfix(BCE=tot_loss)

        logger.add_scalars("DISCRIMINATOR LOSS FUNCTION", {
                'train_loss': tot_loss,
            }, start_epoch+epoch)

        if epoch%10==9:
            torch.save(discriminator.state_dict(), os.path.join(weights_folder, f"discriminator_{start_epoch+epoch}.pt"))

    torch.save(discriminator.state_dict(), os.path.join(weights_folder, f"discriminator_{start_epoch+num_epochs}.pt"))







def train_generator(model,
                    optimizer,
                    loss_fn,
                    train_dataloader,
                    val_dataloader,
                    start_epoch,
                    num_epochs,
                    device,
                    run_name,
                    weights_folder,
                    num_blocks_gen):
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
        pbar.set_description(f"Training epoch {start_epoch+epoch}")
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

        if epoch%5==0:
            val_tot_loss = validation(model=model, loss_fn=loss_fn, val_dataloader=val_dataloader,
                       device=device, epoch=epoch)

            logger.add_scalars(f"LOSS FUNCTIONS {num_blocks_gen} BLOCKS", {
                'train_loss': train_tot_loss,
                'val_loss': val_tot_loss,
            }, start_epoch+epoch)
            torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{start_epoch+epoch}.pt"))
        else:
            logger.add_scalars(f"LOSS FUNCTIONS {num_blocks_gen} BLOCKS", {
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
