import argparse
import os
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file

parser = argparse.ArgumentParser()
parser.add_argument('--trusted', default=None, help='path to .csv file with extracted feature vectors from trusted images')
parser.add_argument('--suspected', default=None, help='path to .csv file extracted feature vectors from suspected images')
parser.add_argument('--dataset', default=None, help='"FERET", "FRGC" or "FRLL" as the datasets have different dividers depending on the dataset')
parser.add_argument('--label',default=None, help='0(bonafide) or 1(morphed) depending on whether the suspected images are morphed or not')
parser.add_argument('--savefile',default=None, help='Name of libsvm file to save data to')
opt = parser.parse_args()
print(opt)

# open readers
trusted_data= pd.read_csv(opt.trusted, delimiter=',', encoding='utf-8', header=None)
suspected_data= pd.read_csv(opt.suspected, delimiter=',', encoding='utf-8', header=None)



difference=[]
for i in range (0,trusted_data.shape[0]):
    trustedImageName=trusted_data.iloc[i][0]
    # Create DMAD pairs depending on dataset
    if opt.dataset=="FERET":
        if trustedImageName.split("_")[2].startswith('f'):
            identity=trustedImageName.split("_")[0]
            for j in range(0,suspected_data.shape[0]):
                if suspected_data.iloc[j][0].split("_")[0]==identity:difference.append((suspected_data.iloc[j][1:].values-trusted_data.iloc[i][1:].values).tolist())
    if opt.dataset=="FRGC":
        identity=trustedImageName.split("d")[0]
        for j in range(0,suspected_data.shape[0]):
            if suspected_data.iloc[j][0].split("d")[0]==identity:difference.append((suspected_data.iloc[j][1:].values-trusted_data.iloc[i][1:].values).tolist())
    if opt.dataset=="FRLL":
        identity=trustedImageName.split("_")[0]
        for j in range(0,suspected_data.shape[0]):
            if suspected_data.iloc[j][0].split("_")[0]==identity:difference.append((suspected_data.iloc[j][1:].values-trusted_data.iloc[i][1:].values).tolist())
# Add labels
labels=[int(opt.label)]*len(difference)
# Save file
dump_svmlight_file(difference, labels, opt.savefile)
