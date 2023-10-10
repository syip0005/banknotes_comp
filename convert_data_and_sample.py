## IMPORTS ##########################################################

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

plt.rcParams["savefig.bbox"] = 'tight'

## Global Parameters ####################################################

TRAINING_PCT = 0.8

## Main Loop ############################################################

if __name__ == "__main__":

    # Read CSV
    df_train_labels = pd.read_csv('./Dataset/train_labels.csv')

    # Manual tt-split / no stratification
    df_train_labels['dset'] = np.random.choice(['train', 'val'], size = len(df_train_labels), p = [TRAINING_PCT, 1 - TRAINING_PCT])

    # Move images into appropriate folders
    for dset_type in ['train', 'val']:

        for cat in df_train_labels['category'].unique(): # Get all unique categories

            Path("./Dataset_Categorised/train/" + str(cat)).mkdir(parents=True, exist_ok=True)
            Path("./Dataset_Categorised/val/" + str(cat)).mkdir(parents=True, exist_ok=True)

            for file_path in df_train_labels[(df_train_labels['category'] == cat) & (df_train_labels['dset'] == dset_type)].iterrows(): # Loop through and move file path for file in category
                shutil.copyfile('./Dataset/train/' + file_path[1].filename, './Dataset_Categorised/' + str(dset_type) + '/' + str(cat) + '/' + file_path[1].filename)