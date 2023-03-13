import os

import cv2

from utils import costants


def create_downsampled_ds(original_ds_path, new_dataset_path, downsample_dimensions):
    """
    Creates a downsampled version of a given dataset.


    """
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)

    for image_name in os.listdir(original_ds_path):
        if ".png" in image_name:
            print(image_name)
            original_image = cv2.imread(os.path.join(original_ds_path, image_name))
            downsampled_image = cv2.resize(original_image, downsample_dimensions)
            cv2.imwrite(os.path.join(new_dataset_path, image_name), downsampled_image)



def rename_images_hr2():
    ds_path = costants.ORIGINAL_DS_TEST
    for image_name in os.listdir(ds_path):
        if ".png" in image_name:
            num_image = image_name.split("x")[0]
            new_name = num_image + ".png"
            print(os.path.join(ds_path, image_name),
                os.path.join(ds_path, image_name))
            os.rename(
                src=os.path.join(ds_path, image_name),
                dst=os.path.join(ds_path, new_name)
            )



def rename_images_x8():
    ds_path = costants.LR_VAL
    for image_name in os.listdir(ds_path):
        if ".png" in image_name:
            num_image = image_name.split(".")[0]
            new_name = num_image + "x4(x8).png"
            print(os.path.join(ds_path, image_name),
                os.path.join(ds_path, image_name))
            os.rename(
                src=os.path.join(ds_path, image_name),
                dst=os.path.join(ds_path, new_name)
            )





def main():
    pass

if __name__=="__main__":
    main()