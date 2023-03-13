
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models import esrgan
from src import baselines, train
from src.super_resolution_dataset import SuperResolutionDataset
from utils import costants, logging_utilities, preprocessing


def main():
    """
    Executes the all the stages of the project
    """

    ## INITIAL CONFIGURATIONS
    # if not os.path.exists("./project_log"):
    #     os.mkdir("./project_log")
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(message)s",
    #     handlers=[
    #         logging.FileHandler("project_log/assignment.log"),
    #         logging.StreamHandler()
    #     ]
    # )
    # logging.info((('-'*70)+'\n')*5)
    # logging_utilities.print_name_stage_project("IMAGE SUPER RESOLUTION")
    # # datetime object containing current date and time
    # now = datetime.now()
    # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    # logging.info(f"DATE AND TIME OF EXCUTION: {dt_string}")




    # ## BASELINES TESTING
    # #baselines.baselines()



    # ## TEST PYTROCH DS
    # transform = transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    # ds = SuperResolutionDataset(
    #     hr_path=costants.ORIGINAL_DS_TRAIN,
    #     lr_path=costants.TRACK2_TRAIN,
    #     transform=transform
    # )
    # #print(ds.list_couples_hr_lr)
    # hr_image, lr_image = ds[567]

    # print(hr_image.shape, lr_image.shape)
    # plt.imshow(lr_image.permute(1,2,0))
    # plt.show()




    ## CREATE DOWNSAMPLED DS

    # preprocessing.create_downsampled_ds(
    #     original_ds_path=costants.ORIGINAL_DS_VAL,
    #     new_dataset_path="../Data/X8/DIV2K_val_LR",
    #     downsample_dimensions=(255, 175)
    # )


    ## RENAME HR2 IMAGES
    # preprocessing.rename_images_x8()



    ## PSNR-BASED GENERATOR TRAINING

    train.train(num_epochs=2,
                device="cpu",
                run_name="TEST_GENERATOR")




if __name__ == "__main__":
    main()
