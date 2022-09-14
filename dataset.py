import os
import torch
from torchvision import transforms, datasets
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import csv
import random

kirc1 = ['B0-5691', 'B0-5693', 'B0-4707', 'DV-5566', 'A3-3358', 'B0-5088', 'CJ-4900', 'BP-4999', 'A3-3336', 'DV-5565', 'CJ-6030', 'A3-3329', 'B2-5635', 'B0-5698', 'DV-5574', 'BP-4353', 'BP-4986', 'CJ-4904', 'BP-4327', 'BP-4775', 'BP-4160', 'CJ-4891', 'CJ-6027', 'BP-4807', 'B0-5812', 'BP-4342', 'DV-5568', 'BP-5178', 'CJ-4642', 'DV-5575', 'BP-4994', 'DV-5569', 'BP-5168', 'BP-5006', 'A3-3307', 'CZ-4856', 'BP-4158', 'AK-3425', 'BP-5199', 'B0-5694']
kirc2 = ['A3-3325', 'BP-5181', 'BP-4344', 'BP-4995', 'B0-5695', 'AK-3450', 'BP-5009', 'BP-4799', 'CZ-4865', 'BP-5191']

kirp1 = ['BQ-7046', 'AT-A5NU', 'UZ-A9Q0', 'BQ-7049', 'Y8-A898', 'BQ-7055', 'A4-7288', 'SX-A71W', 'BQ-7058', 'MH-A857', 'DW-5561', 'BQ-5886', 'BQ-5888', 'Q2-A5QZ', '2Z-A9JE', '2Z-A9J8', 'GL-7966', 'SX-A7SP', 'A4-8518', 'SX-A7SL', 'MH-A561', 'A4-7734', 'DW-7834', 'IA-A83W', '2Z-A9JO', 'BQ-5887', '5P-A9K6', 'DW-7841', 'DW-7840', '5P-A9K3', 'DW-7842', 'DW-7838', 'EV-5902', 'B9-5155', 'DW-7839', 'G7-6790', 'P4-A5EB', 'DW-7837', 'A4-7584', 'BQ-5884']
kirp2 = ['B9-A44B', 'UZ-A9PV', 'DW-5560', '2Z-A9J2', '5P-A9KF', 'BQ-5879', 'BQ-5876', '5P-A9K9', 'KV-A6GE', 'B1-A47N']

kich1 = ['UW-A72S', 'UW-A72I', 'UW-A72H', 'KN-8424', 'KN-8432', 'UW-A72N', 'KN-8419', 'KN-8418', 'KO-8406', 'KL-8339', 'UW-A72O', 'UW-A72L', 'KL-8325', 'KL-8335', 'KO-8405', 'KN-8436', 'KL-8329', 'KM-8443', 'KO-8408', 'KO-8411', 'KL-8330', 'KO-8407', 'UW-A72P', 'UW-A72T', 'NP-A5H7', 'NP-A5GZ', 'UW-A72J', 'KN-8430', 'KN-8437', 'KL-8342', 'KO-8414', 'KL-8344', 'KO-8421', 'KL-8334', 'KO-8409', 'KL-8346', 'KO-8410', 'KN-8426', 'KL-8327', 'KM-8442']
kich2 = ['NP-A5GY', 'KL-8328', 'KO-8415', 'KN-8431', 'KL-8340', 'KN-8435', 'KN-8423', 'KL-8337', 'KL-8331', 'KL-8332']

# split slides (patients) for training and testing set
# (here we consider 5-fold cross-validation and dataset is divided to 5 groups)
def slide_split(train, test):
    # ex. train = '1234', test = '5'
    data_map = {}
    data_map['data1'] = [kirc1, kirp1, kich1]
    data_map['data2'] = [kirc2, kirp2, kich2]

    train_kirc = data_map[f'data{train}'][0]
    train_kirp = data_map[f'data{train}'][1]
    train_kich = data_map[f'data{train}'][2]

    test_kirc = data_map[f'data{test}'][0]
    test_kirp = data_map[f'data{test}'][1]
    test_kich = data_map[f'data{test}'][2]

    train_dataset_num = train_kirc + train_kirp + train_kich
    test_dataset_num = test_kirc + test_kirp + test_kich

    return train_dataset_num, test_dataset_num
    # return train_basiloma, train_bowen, train_squamous, test_basiloma, test_bowen, test_squamous

# def make_dataset(csv_path, dataset):
def make_dataset(csv_path, mag, dataset_num):
    ############ read csv file for label ############
    with open(csv_path,"r") as csvfile:
        reader = csv.DictReader(csvfile)
        stockPxData = []
        for row in reader:
            stockPxData.append(row)

    ####### make data #########
    dataset = []
    if mag == '10X':
        for slideID in dataset_num:
            for instance in stockPxData:
                if instance['img_name'] == slideID:
                    label = [float(instance['label1']), float(instance['label2']), float(instance['label3']), float(instance['label4']), float(instance['label5']), float(instance['label6'])]
                    dataset.append([slideID, label])
    elif mag == '20X':  
        for slideID in dataset_num:
            for instance in stockPxData:
                if instance['img_name'] == slideID:
                    label = [float(instance['label1']), float(instance['label2']), float(instance['label3']), float(instance['label4']), float(instance['label5']), float(instance['label6'])]
            dataset.append([slideID, label])
    elif mag == '40X':
        for slideID in dataset_num:
            for instance in stockPxData:
                if instance['img_name'] == slideID:
                    label = [float(instance['label1']), float(instance['label2']), float(instance['label3']), float(instance['label4']), float(instance['label5']), float(instance['label6'])]
            dataset.append([slideID, label])
    else:
        print("The mag message is arror !")
    csvfile.close()
    random.shuffle(dataset)
    return dataset

# def make_test_dataset(csv_path, basiloma, bowen, squamous):
def make_test_dataset(csv_path, test_dataset_num):
    ############ read csv file for label ############
    with open(csv_path,"r") as csvfile:
        reader = csv.DictReader(csvfile)
        stockPxData = []
        for row in reader:
            stockPxData.append(row)

    ####### make data #########
    dataset = []
    for slideID in test_dataset_num:
        for instance in stockPxData:
            if instance['img_name'] == slideID:
                label = int(instance['label'])
        dataset.append([slideID, label])
    csvfile.close()
    random.shuffle(dataset)
    return dataset