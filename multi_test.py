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
import math
from tqdm import tqdm
from utils import cal_metrics

def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

def test(pred_model, device, test_data, LABEL_NUM_CLASSES, label_thrd, bag_vote_thrd):
    sum_label = np.zeros(LABEL_NUM_CLASSES)
    for data in test_data:
        # loading data
        img_id, input_tensor, class_label = utils.data_load(data)
        # to device
        input_tensor = input_tensor.to(device)
        class_prob, class_hat = pred_model(input_tensor)
        class_prob = class_prob.squeeze()
        pred_result = class_prob > label_thrd
        pred_result = pred_result.cpu().numpy()
        sum_label += pred_result
    num_thrd = int(len(test_data) * bag_vote_thrd)
    final_pred_label = sum_label > num_thrd
    final_pred_label = final_pred_label.astype(int).tolist()
    return final_pred_label

if __name__ == "__main__":
    ################## experimental setup ##########################
    train_slide = '1'
    test_slide = '2'
    mag = ['10X','20X','40X']
    device = 'cuda:0'
    label_path = './dataset/final_label.csv'
    output_path = './output/'
    save_name = '_best_model.pkl'
    label_num_dict = {'10X':6,'20X':6,'40X':6}
    label_thrd = {'10X':0.5,'20X':0.5,'40X':0.5}
    bag_vote_thrd = 0.5
    FINAL_CLASS = 3

    #################### expert knowledge ##########################
    kirc_expert = {'10X':[[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], 
    '20X':[[1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]], 
    '40X':[[1, 0, 0, 0, 0, 0]]}
    kirp_expert = {'10X':[[0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]], 
    '20X':[[0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]], 
    '40X':[[0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]}
    kich_expert = {'10X':[[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], 
    '20X':[[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], 
    '40X':[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]}

    disease_name = ["kirc", "kirp", "kich"]

    ###################### data and model ##########################
    # split slides for training and validation
    train_dataset_num, test_dataset_num = ds.slide_split(train_slide, test_slide)
    # provide class and labels for training data
    test_dataset = ds.make_test_dataset(label_path, test_dataset_num)

    torch.backends.cudnn.benchmark=True #cudnn benchmark mode
    ########## start test ###########
    
    x10_result = {}
    x20_result = {}
    x40_result = {}
    for mag_num in range(len(mag)):
        mag_name = mag[mag_num]
        LABEL_NUM_CLASSES = label_num_dict[mag_name]
        feature_extractor = model.feature_extractor()
        class_predictor = model.class_predictor(LABEL_NUM_CLASSES)
        # class_predictor = model.class_predictor_noatten(LABEL_NUM_CLASSES)
        pred_model = model.EM_MIL(feature_extractor, class_predictor)
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        pred_model.load_state_dict(torch.load(output_path + str(mag_name) + '/' + str(mag_name) + str(save_name), map_location = torch.device('cpu')))
        pred_model = pred_model.to(device)
        pred_model.eval() # eval mode
        # start testing
        print("Start to test {} image.".format(mag_name))
        for data in tqdm(test_dataset):
            slideID = data[0]
            class_label = data[1]
            bag_list = utils.build_bag(slideID, class_label, mag_name, 25, 40)
            final_pred_label = test(pred_model, device, bag_list, LABEL_NUM_CLASSES, label_thrd[mag_name], bag_vote_thrd)
            # print(slideID, final_pred_label)
            if mag_name == '10X':
                x10_result[slideID] = final_pred_label
            elif mag_name == '20X':
                x20_result[slideID] = final_pred_label
            elif mag_name == '40X':
                x40_result[slideID] = final_pred_label
            else:
                print("The thing about scale name is error ! ")
    
    ########## will delate soon ##########
    '''
    test_dataset = [['114661', 0], ['1510866', 2], ['149841', 1]] 
    x10_result = {'114661':[1, 1, 0, 0, 0], '1510866':[0, 0, 0, 1, 1], '149841':[0, 0, 1, 1, 0]}
    x20_result = {'114661':[1, 1, 0, 0, 0], '1510866':[0, 0, 0, 1, 1], '149841':[0, 0, 1, 1, 0]}
    x40_result = {'114661':[1, 1, 0, 0, 0], '1510866':[0, 0, 0, 1, 1], '149841':[0, 1, 1, 0, 0]}
    '''
    ########### metric distance ###########
    
    true_positive = 0
    sum_confusion_matrix = np.zeros((FINAL_CLASS, FINAL_CLASS))
    for data in test_dataset:
        image_name = data[0]
        final_label = data[1]
        predict_x = int(''.join(str(i) for i in x10_result[image_name]),2)
        predict_y = int(''.join(str(i) for i in x20_result[image_name]),2)
        predict_z = int(''.join(str(i) for i in x40_result[image_name]),2)
        predict_xyz = [predict_x, predict_y, predict_z]
        kirc_distance = []
        kirp_distance = []
        kich_distance = []
        for sub_x_label in kirc_expert['10X']:
            label_x = int(''.join(str(i) for i in sub_x_label),2)
            for sub_y_label in kirc_expert['20X']:
                label_y = int(''.join(str(i) for i in sub_y_label),2)
                for sub_z_label in kirc_expert['40X']:
                    label_z = int(''.join(str(i) for i in sub_z_label),2)
                    label_xyz = [label_x, label_y, label_z]
                    distance = eucliDist(predict_xyz, label_xyz)
                    kirc_distance.append(distance)
        for sub_x_label in kirp_expert['10X']:
            label_x = int(''.join(str(i) for i in sub_x_label),2)
            for sub_y_label in kirp_expert['20X']:
                label_y = int(''.join(str(i) for i in sub_y_label),2)
                for sub_z_label in kirp_expert['40X']:
                    label_z = int(''.join(str(i) for i in sub_z_label),2)
                    label_xyz = [label_x, label_y, label_z]
                    distance = eucliDist(predict_xyz, label_xyz)
                    kirp_distance.append(distance)
        for sub_x_label in kich_expert['10X']:
            label_x = int(''.join(str(i) for i in sub_x_label),2)
            for sub_y_label in kich_expert['20X']:
                label_y = int(''.join(str(i) for i in sub_y_label),2)
                for sub_z_label in kich_expert['40X']:
                    label_z = int(''.join(str(i) for i in sub_z_label),2)
                    label_xyz = [label_x, label_y, label_z]
                    distance = eucliDist(predict_xyz, label_xyz)
                    kich_distance.append(distance)

        final_distance = [min(kirc_distance), min(kirp_distance), min(kich_distance)]
        final_predict_label = final_distance.index(min(final_distance))
        ### bowen conclusion ###
        # if x10_result[image_name][-1] == 1:
        #     final_predict_label = 2
        ########################
        final_predict_label_name = disease_name[final_predict_label]
        final_label_name = disease_name[final_label]
        print("Image name is : {}".format(image_name))
        print("10X feature is : {}".format(x10_result[image_name]))
        print("20X feature is : {}".format(x20_result[image_name]))
        print("40X feature is : {}".format(x40_result[image_name]))
        print("Predicted disease is : {}, Label is {}\n".format(final_predict_label_name, final_label_name))
        if final_predict_label == final_label:
            true_positive += 1
        sum_confusion_matrix[final_label][final_predict_label] += 1
    
    ############ final predicted result ############
    acc = true_positive / len(test_dataset)
    print("The final test accuracy is {:.2f} %".format(acc*100))
    test_prec_recall_speci = cal_metrics(sum_confusion_matrix)
    print('Test confusion_matrix: {}'.format(sum_confusion_matrix))
    print('Test test_prec_recall_speci: {}'.format(test_prec_recall_speci))
        
