# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import csv
import os
import random
import utils
import model
import dataset as ds
from tqdm import tqdm
from utils import calculate_acuracy_mode_one, calculate_acuracy_mode_two, calculate_acuracy_mode_three
import time

def test(model, device, test_data, mag):
    model.eval() # eval mode
    total_acc = 0.0
    # test_precision = 0.0
    # test_recall = 0.0
    # shuffle bag
    
    kirc = ['A3-3325', 'BP-5181', 'BP-4344', 'BP-4995', 'B0-5695', 'AK-3450', 'BP-5009', 'BP-4799', 'CZ-4865', 'BP-5191']
    kirp = ['B9-A44B', 'UZ-A9PV', 'DW-5560', '2Z-A9J2', '5P-A9KF', 'BQ-5879', 'BQ-5876', '5P-A9K9', 'KV-A6GE', 'B1-A47N']
    kich = ['NP-A5GY', 'KL-8328', 'KO-8415', 'KN-8431', 'KL-8340', 'KN-8435', 'KN-8423', 'KL-8337', 'KL-8331', 'KL-8332']

    random.shuffle(test_data)
    f_kirc = open(f'./{mag}_ensemble_kirc.csv','w',encoding='utf-8')
    f_kirp = open(f'./{mag}_ensemble_kirp.csv','w',encoding='utf-8')
    f_kich = open(f'./{mag}_ensemble_kich.csv','w',encoding='utf-8')
    kirc_write = csv.writer(f_kirc)
    kirp_write = csv.writer(f_kirp)
    kich_write = csv.writer(f_kich)
    for data in tqdm(test_data):
        # loading data
        img_id, input_tensor, class_label = utils.data_load(data)
        # to device
        input_tensor = input_tensor.to(device)
        class_label = class_label.to(device)
        class_prob, class_hat = model(input_tensor)
        acc = calculate_acuracy_mode_three(class_prob, class_label)
        total_acc += acc
        write_content = class_prob[0].detach().cpu().numpy()
        write_content1 = np.round(write_content,3)
        if img_id in kirc:
            kirc_write.writerow(write_content1)
        elif img_id in kirp:
            kirp_write.writerow(write_content1)
        else:
            kich_write.writerow(write_content1)
        # csv_write.writerow(write_content1)

        # csv_write.writerow(class_sigmoid.detach().cpu().numpy())
        # precision, recall = calculate_acuracy_mode_one(class_prob, class_label)
        # test_precision += precision
        # test_recall += recall
    test_acc = total_acc / len(test_data)
    return test_acc

if __name__ == "__main__":
    ################## experimental setup ##########################
    train_slide = '1'
    test_slide = '2'
    mag = '40X' # ('20X' or '40X')
    device = 'cuda:2'
    csv_path = './dataset/'+ str(mag) + '_label.csv'
    output_path = './output/'+ str(mag)
    label_num_dict = {'10X':6,'20X':6,'40X':6}
    LABEL_NUM_CLASSES = label_num_dict[mag]
    ################################################################
    # split slides for training and validation
    train_dataset_num, test_dataset_num = ds.slide_split(train_slide, test_slide)
    # provide class and labels for training data
    test_dataset = ds.make_dataset(csv_path, mag, test_dataset_num)

    torch.backends.cudnn.benchmark=True #cudnn benchmark mode

    feature_extractor = model.feature_extractor()
    # class_predictor = model.class_predictor_noatten(LABEL_NUM_CLASSES)
    class_predictor = model.class_predictor(LABEL_NUM_CLASSES)

    model = model.EM_MIL(feature_extractor, class_predictor)
    model = model.to(device)
    model.load_state_dict(torch.load(output_path + '/' + str(mag) + '_best_model.pkl', map_location = torch.device('cpu')))

    # start testing
    test_start = time.time()
    test_data = []
    for data in test_dataset:
        slideID = data[0]
        class_label = data[1]
        bag_list = utils.build_bag(slideID, class_label, mag, 25, 40)
        test_data = test_data + bag_list
    test_acc = test(model, device, test_data, mag)
    test_end = time.time()
    print('Test ACC: {:.4f}'.format(test_acc))
    print('Test use time: {:.2f} minutes'.format((test_end - test_start)/60))