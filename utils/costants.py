"""
This module contains costants used across all the repository
"""

import os

## PATHS

PROJECT_PATH = os.path.abspath(".")
DATASET_PATH = os.path.join(PROJECT_PATH, "../Data")

## ORIGINAL HIGH RES
ORIGINAL_DS = os.path.join(DATASET_PATH, "HR")
ORIGINAL_DS_TRAIN = os.path.join(ORIGINAL_DS, "DIV2K_train_HR")
ORIGINAL_DS_TEST = os.path.join(ORIGINAL_DS, "DIV2K_test_HR")
ORIGINAL_DS_VAL = os.path.join(ORIGINAL_DS, "DIV2K_valid_HR")


## TRACK 1 (BICUBIC)
TRACK1 = os.path.join(DATASET_PATH, "Track_1")
TRACK1_TRAIN = os.path.join(TRACK1, "DIV2K_train_LR_bicubic", "X4")
TRACK1_TEST = os.path.join(TRACK1, "DIV2K_test_LR_bicubic", "X4")
TRACK1_VAL = os.path.join(TRACK1, "DIV2K_val_LR_bicubic", "X4")



## TRACK 2
TRACK2 = os.path.join(DATASET_PATH, "Track_2")
TRACK2_TRAIN = os.path.join(TRACK2, "DIV2K_train_LR_unknown", "X4")
TRACK2_TEST = os.path.join(TRACK2, "DIV2K_test_LR_unknown", "X4")
TRACK2_VAL = os.path.join(TRACK2, "DIV2K_val_LR_unknown", "X4")


