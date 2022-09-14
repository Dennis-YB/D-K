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
import time
from utils import calculate_acuracy_mode_one, calculate_acuracy_mode_two, calculate_acuracy_mode_three

def train(model, device, loss_fn, optimizer, train_data):
    model.train() # train mode
    total_loss = 0.0
    # running_precision = 0.0
    # running_recall = 0.0
    total_acc = 0.0
    # shuffle bag
    random.shuffle(train_data)
    for i, data in enumerate(train_data):
        # loading data
        img_id, input_tensor, class_label = utils.data_load(data)
        # to device
        input_tensor = input_tensor.to(device)
        class_label = class_label.to(device)
        optimizer.zero_grad() # initialize gradient
        class_prob, class_hat = model(input_tensor)
        # print(img_id, class_prob)
        class_prob = class_prob.to(torch.float32)
        class_label = class_label.to(torch.float32)
        # calculate loss
        loss = loss_fn(class_prob, class_label)
        total_loss += loss.item()
        loss.backward() # backpropagation
        optimizer.step() # renew parameters
        current_learning_rate = optimizer.param_groups[0]['lr']

        # precision, recall = calculate_acuracy_mode_one(class_prob, class_label)
        acc = calculate_acuracy_mode_three(class_prob, class_label)
        total_acc += acc
        # running_precision += precision
        # running_recall += recall

        if (i+1)%100 == 0:
            print('{}/{} train iters loss: {:.4f}, lr: {}'.format((i+1), len(train_data), (total_loss / (i+1)), current_learning_rate))

    epoch_loss = total_loss / len(train_data)
    epoch_acc = total_acc / len(train_data)
    # epoch_precision = running_precision / len(train_data)
    # epoch_recall = running_recall / len(train_data)
    return epoch_loss, epoch_acc, current_learning_rate

def test(model, device, test_data):
    model.eval() # eval mode
    total_acc = 0.0
    # test_precision = 0.0
    # test_recall = 0.0
    # shuffle bag
    random.shuffle(test_data)
    for data in test_data:
        # loading data
        img_id, input_tensor, class_label = utils.data_load(data)
        # to device
        input_tensor = input_tensor.to(device)
        class_label = class_label.to(device)
        class_prob, class_hat = model(input_tensor)
        acc = calculate_acuracy_mode_three(class_prob, class_label)
        total_acc += acc
        # precision, recall = calculate_acuracy_mode_one(class_prob, class_label)
        # test_precision += precision
        # test_recall += recall
    test_acc = total_acc / len(test_data)
    return test_acc
    # test_final_precision = test_precision / len(test_data)
    # test_final_recall = test_recall / len(test_data)
    # return test_final_precision, test_final_recall

if __name__ == "__main__":
    ################## experimental setup ###########################
    train_slide = '1'
    test_slide = '2'
    mag = '40X' # ('10X', '20X' or '40X')
    label_num_dict = {'10X':6,'20X':6,'40X':6}
    # w_dict = {'10X':[1,1,5,5,5,5,5,5],'20X':[1,1,5,5,5,5,5,5],'40X':[1,1,5,5,5,5,5,5]}
    # w_dict = {'10X':[1,1,1,1,1,1,1,1],'20X':[1,1,1,1,1,1,1,1],'40X':[1,1,1,1,1,1,1,1]}
    EPOCHS = 10
    device = 'cuda:3'
    csv_path = './dataset/'+ str(mag) + '_label.csv'
    output_path = './output/'+ str(mag) # noatten_lossweight
    save_name = '_best_model.pkl'
    LR = 0.0001
    LR_decay_step = 20
    LABEL_NUM_CLASSES = label_num_dict[mag]
    ################################################################
    # split slides for training and validation
    train_dataset_num, test_dataset_num = ds.slide_split(train_slide, test_slide)
    # provide class and labels for training data
    train_dataset = ds.make_dataset(csv_path, mag, train_dataset_num)
    test_dataset = ds.make_dataset(csv_path, mag, test_dataset_num)
    torch.backends.cudnn.benchmark=True #cudnn benchmark mode

    ################# model ###############
    feature_extractor = model.feature_extractor()
    class_predictor = model.class_predictor(LABEL_NUM_CLASSES)
    # class_predictor = model.class_predictor_noatten(LABEL_NUM_CLASSES) # no attention
    model = model.EM_MIL(feature_extractor, class_predictor)
    # model.load_state_dict(torch.load(output_path + '/' + str(mag) + '_recall_best_model.pkl')) # re-train

    model = model.to(device)

    # use BCELoss function
    # w = torch.tensor(w_dict[mag], dtype=torch.float)
    # w = w.to(device)
    loss_fn = nn.BCELoss()
    # use SGD momentum for optimizer and step reduce
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0)
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_decay_step, gamma=0.1)

    ######### start training ##########
    # best_precision_score = 0.0
    # best_recall_score = 0.0
    best_acc = 0.0
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for epoch in range(EPOCHS):
        train_start = time.time()
        # generate bags
        train_data = []
        for data in train_dataset:
            slideID = data[0]
            class_label = data[1]
            bag_list = utils.build_bag(slideID, class_label, mag, 25, 40)
            train_data = train_data + bag_list
        epoch_loss, epoch_acc, current_learning_rate = train(model, device, loss_fn, optimizer, train_data)
        train_end = time.time()
        scheduler.step() #learning rate
        print('{}/{} Train Loss: {:.4f}'.format(epoch+1, EPOCHS, epoch_loss))
        print('{}/{} Train ACC: {:.4f}'.format(epoch+1, EPOCHS, epoch_acc))
        # print('{}/{} Train Precision: {:.4f}'.format(epoch+1, EPOCHS, epoch_precision))
        # print('{}/{} Train Recall: {:.4f}'.format(epoch+1, EPOCHS, epoch_recall))
        print('{}/{} Train Learning_rate: {}'.format(epoch+1, EPOCHS, current_learning_rate))
        print('{}/{} Train use time: {:.2f} minutes'.format(epoch+1, EPOCHS, (train_end - train_start)/60))
        
        ###### test #########
        test_start = time.time()
        test_data = []
        for data in test_dataset:
            slideID = data[0]
            class_label = data[1]
            bag_list = utils.build_bag(slideID, class_label, mag, 25, 40)
            test_data = test_data + bag_list
        test_acc = test(model, device, test_data)
        test_end = time.time()
        print('{}/{} Test ACC: {:.4f}'.format(epoch+1, EPOCHS, test_acc))
        # print('{}/{} Test Precision: {:.4f}'.format(epoch+1, EPOCHS, test_final_precision))
        # print('{}/{} Test Recall: {:.4f}'.format(epoch+1, EPOCHS, test_final_recall))
        print('{}/{} Test use time: {:.2f} minutes'.format(epoch+1, EPOCHS, (test_end - test_start)/60))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), output_path + '/' + str(mag) + str(save_name))
            # torch.save(model.state_dict(), output_path + '/' + str(mag)+ '_' + str(epoch+1) + '_best_model.pkl')
        # if test_final_precision > best_precision_score:
        #     best_precision_score = test_final_precision
        #     torch.save(model.state_dict(), output_path + '/' + str(mag) + '_precision_best_model.pkl')
        #     # torch.save(model.state_dict(), output_path + '/' + str(mag) + '_gcn_precision_best_model.pkl')
        # if test_final_recall > best_recall_score:
        #     best_recall_score = test_final_recall
        #     torch.save(model.state_dict(), output_path + '/' + str(mag) + '_recall_best_model.pkl')
        #     # torch.save(model.state_dict(), output_path + '/' + str(mag) + '_gcn_recall_best_model.pkl')
        '''
        else:
            torch.save(model.state_dict(), output_path + '/' + str(mag) + '_' + str(epoch+1) + '_epoch_model.pkl')
        '''