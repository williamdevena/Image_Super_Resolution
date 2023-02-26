"""
This file contains the baselines used
"""
import os

import cv2
import skimage
import torch

from src import costants, metrics


def baselines():
    """
    Runs the baselines and produces metrics.
    """
    # test_image_path = os.path.join(
    #     costants.DATASET_PATH, "Track_2", "DIV2K_test_LR_unknown", "X4", "0700x4.png")
    # img = cv2.imread(test_image_path)
    # size = costants.ORIGINAL_SIZE

    # #upscaled_img = bilinear_interpolation_upscaling(img, size)
    # upscaled_img = nearest_neighbour_upscaling(img, size)

    # cv2.imwrite("./test_upscale.png", upscaled_img)

    # original_img_path = os.path.join(
    #     costants.DATASET_PATH, "HR", "DIV2K_test_HR", "0700.png")
    # original_img = cv2.imread(original_img_path)

    # ssim_score = skimage.metrics.structural_similarity(original_img, upscaled_img, channel_axis=2)
    # # ssim_score = skimage.metrics.structural_similarity(original_img, original_img, channel_axis=2)

    # # lpips_distance = metrics.calculate_lpips_distance(torch.tensor(original_img).view(3, 1728, 2040), torch.tensor(upscaled_img).view(3, 1728, 2040))
    # lpips_distance = metrics.calculate_lpips_distance(torch.tensor(original_img).view(3, 1728, 2040), torch.tensor(original_img).view(3, 1728, 2040))


    # print(ssim_score, lpips_distance)

    test_ds_path = os.path.join(
         costants.DATASET_PATH, "Track_2", "DIV2K_test_LR_unknown", "X4")
    original_ds_path = costants.ORIGINAL_DS_TEST

    test_bilinear_interpolation_upscaling(
        test_ds_path=test_ds_path,
        original_ds_path=original_ds_path,
        path_upscaled_folder="./test_upsampling")



def test_bilinear_interpolation_upscaling(test_ds_path, original_ds_path, path_upscaled_folder):


    files = os.listdir(test_ds_path)

    for img_name in files:
        ## LOW RESOLUTION IMAGE
        low_res_img_path = os.path.join(test_ds_path, img_name)
        low_res_img = cv2.imread(low_res_img_path)

        ## ORIGINAL HIGH RESOLUTION IMAGE
        original_img_name = "".join([img_name.split("x")[0], ".png"])
        original_img_path = os.path.join(original_ds_path, original_img_name)
        # print(original_img_path, low_res_img_path)
        original_img = cv2.imread(original_img_path)
        original_shape = original_img.shape

        ## UPSCALED HIGH RESOLUTION IMAGE
        size = (original_shape[1], original_shape[0])
        upscaled_img = bilinear_interpolation_upscaling(low_res_img, size)

        ## METRICS
        ssim_score = skimage.metrics.structural_similarity(original_img, upscaled_img, channel_axis=2)
        lpips_distance = metrics.calculate_lpips_distance(torch.tensor(original_img).view(3, original_shape[0], original_shape[1]), torch.tensor(upscaled_img).view(3, original_shape[0], original_shape[1]))

        cv2.imwrite(os.path.join(path_upscaled_folder, original_img_name), upscaled_img)

        print(ssim_score, lpips_distance)





def bilinear_interpolation_upscaling(img, size):
    """
    Upscales an image using bilinear interpolation.

    Args:
        img (_type_): _description_
    """

    upscaled_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    return upscaled_img


def nearest_neighbour_upscaling(img, size):
    """
    Upscales an image using nearest neighbour.

    Args:
        img (_type_): _description_
    """
    upscaled_img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    return upscaled_img
