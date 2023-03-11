import os

import cv2


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


def main():
    pass

if __name__=="__main__":
    main()
