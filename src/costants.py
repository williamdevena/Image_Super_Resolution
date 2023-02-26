"""
This module contains costants used across all the repository
"""

import os

## PATHS

PROJECT_PATH = os.path.abspath(".")
DATASET_PATH = os.path.join(PROJECT_PATH, "../Data")
ORIGINAL_DS = os.path.join(DATASET_PATH, "HR")
ORIGINAL_DS_TRAIN = os.path.join(ORIGINAL_DS, "DIV2K_train_HR")
ORIGINAL_DS_TEST = os.path.join(ORIGINAL_DS, "DIV2K_test_HR")
ORIGINAL_DS_VAL = os.path.join(ORIGINAL_DS, "DIV2K_valid_HR")



