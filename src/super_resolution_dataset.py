import os

import cv2
from PIL import Image
from torch.utils.data import Dataset

from utils import costants


class SuperResolutionDataset(Dataset):
    """
    Custom PyTorch dataset used to train models in the
    Image Super Resolution  framework.
    """

    def __init__(self, hr_path, lr_path, transform=None):
        self.transform = transform
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.list_couples_hr_lr = self.build_list_couples_hr_lr()


    def build_list_couples_hr_lr(self):
        """
        Builds a list of tuples that contain the path of a HR image and
        the path of the corresponding LR image.

        Returns:
            list_couples_hr_lr (List): contains tuples that contain
            the path of a HR image and the path of the corresponding LR image, in
            the following form (hr_path, lr_path).
        """
        hr_images = os.listdir(self.hr_path)
        list_couples_hr_lr = []

        for hr_image_name in hr_images:
            if ".png" in hr_image_name:
                id_num = hr_image_name.split(".")[0]
                lr_image_name = id_num +"x4(x8).png"
                hr_image_path = os.path.join(self.hr_path, hr_image_name)
                lr_image_path = os.path.join(self.lr_path, lr_image_name)
                list_couples_hr_lr.append((hr_image_path, lr_image_path))

        list_couples_hr_lr.sort(key=lambda x: int(x[0].split("/")[-1].split(".")[0]))

        return list_couples_hr_lr


    def __len__(self):
        return len(self.list_couples_hr_lr)


    def __getitem__(self, idx):
        hr_image_name, lr_image_name = self.list_couples_hr_lr[idx]
        # hr_image = cv2.imread(hr_image_name)
        # lr_image = cv2.imread(lr_image_name)
        hr_image = Image.open(hr_image_name)
        lr_image = Image.open(lr_image_name)
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return hr_image, lr_image


if __name__=="__main__":
    ds = SuperResolutionDataset(
        hr_path=costants.ORIGINAL_DS_TRAIN,
        lr_path=costants.TRACK2_TRAIN
    )

    print(ds.list_couples_hr_lr)
