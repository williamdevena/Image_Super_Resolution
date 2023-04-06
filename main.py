
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models import esrgan
from src import super_resolution_dataset, test
#from src.super_resolution_dataset import SuperResolutionDataset
from utils import costants, logging_utilities, preprocessing


def main():
    """
    Executes the all the stages of the project
    """

    ## INITIAL CONFIGURATIONS
    # if not os.path.exists("./project_log"):
    #     os.mkdir("./project_log")
    # if not os.path.exists("./output_images"):
    #     os.mkdir("./output_images")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("project_log/assignment.log"),
            logging.StreamHandler()
        ]
    )
    logging.info((('-'*70)+'\n')*5)
    logging_utilities.print_name_stage_project("IMAGE SUPER RESOLUTION")
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"DATE AND TIME OF EXCUTION: {dt_string}")



    ## CHECKING OVERALL QUALITY OF DATASET
    #preprocessing.analyze_quality_of_images(costants.ORIGINAL_DS_TRAIN)



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

    # train.train(start_epoch=100,
    #     num_epochs=100,
    #     device="cuda",
    #     run_name="PSNR_GENERATOR_3",
    #     weights_folder="./models_weights/gen_15blocks",
    #     batch_size=64,
    #     lr=0.0005,
    #     num_blocks_gen=15,
    #     pretrained_gen=True,
    #     #pretrained_disc=False,
    #     weights_gen="./models_weights/gen_15blocks/model_100.pt"
    # )



    ### TESTING
    test_dataloader = test.build_test_dataloader()
    model = esrgan.Generator(upsample_algo="nearest", num_blocks=2)
    device="cpu"
    model.load_state_dict(torch.load("./models_weights/gan.pt", map_location=torch.device(device=device)))
    run_name="GAN_TRAINING_2"
    image_folder = "./output_images/gan"

    tot_lpips, tot_ssim, tot_psnr = test.testing_generator(model=model, test_dataloader=test_dataloader, device=device, image_folder=image_folder)
    print(tot_lpips, tot_ssim, tot_psnr)




if __name__ == "__main__":
    main()
