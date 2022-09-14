import cv2
import torch
import os
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from multiprocessing import Pool

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# make lists for patch ID of all slides
# data_dir = './dataset/patch/10X' # or /20X and /40X
# patch_map = {} # key:slideID
# for slideID in os.listdir(data_dir): # make patch ID lists for all slides
#     patch_map[slideID] = []
#     for patch in os.listdir(f'{data_dir}/{slideID}'):
#         patch_i = patch.split('_')[1].split('.')[0] # get patch ID from image filename
#         patch_map[slideID].append(patch_i)

'''
bag-generating function for single-scale
class_label: {0, 1, 0, 1, 1}
mag: magnification ('10X' or '20X' or '40X')
inst_num: The number of patches for a bag (basically 100)
max_bag: The maximum number of bags for a slide (basically 50)
'''
def build_bag(slideID, class_label, mag, inst_num, max_bag):
    patch_dir = f'./dataset/patch/{mag}/{slideID}' # directory for a selected magnification
    patch_i_list = os.listdir(patch_dir) # list of patch ID for a slide
    patch_num = len(patch_i_list) # the number of patches for a slide
    bag_num = int(patch_num/inst_num) # the possible number of bags
    if bag_num < max_bag:
        max_bag = bag_num
    random.shuffle(patch_i_list) # shuffle patch ID
    bag_list = []
    for i in range(max_bag):
        bag = []
        start = i*inst_num
        end = start + inst_num
        for j in range(start, end):
            patch_i = patch_i_list[j]
            patch_path = f'{patch_dir}/{patch_i}'
            bag.append(patch_path)
        bag_list.append([slideID, bag, class_label])
    return bag_list

'''
data loading function for training bag
data = [patch_list, class_label]
'''
def process_img(patch_path):
    img = cv2.imread(patch_path)
    img = img.transpose((2, 1, 0))
    return img

def data_load(data):
    img_id = data[0]
    ###### normalization ######
    '''
    tensor_list = []
    # read image -> add to list as tensor
    for patch_path in data[1]:
        img = Image.open(patch_path)
        input_img = data_transform(img)
        tensor_list.append(input_img)
    '''
    ############################
    # tensor_list = [torch.Tensor(cv2.imread(patch_path).transpose((2, 1, 0))) for patch_path in data[1]]
    # generate bag tensor by stacking all tensors in a bag (size = [# instance,3,224,224])
    # input_tensor = torch.stack(tensor_list, dim=0)
    pool = Pool(6) # suggest num_workers for multiprocess
    tensor_list = pool.map(process_img, data[1])
    pool.close()
    pool.join()
    input_tensor = torch.Tensor(np.array(tensor_list))
    # generate class label
    class_label = torch.tensor([data[2]], dtype=torch.int64)
    return img_id, input_tensor, class_label

######### calculate precision and acc ############
def calculate_acuracy_mode_one(model_pred, labels):
    # Model_pred was processed by sigmoid.
    # If it is bigger than th, it is true.
    accuracy_th = 0.3
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    if pred_one_num == 0 or target_one_num == 0:
        return 0, 0
    # precision
    precision = true_predict_num / pred_one_num
    # recall
    recall = true_predict_num / target_one_num
 
    return precision.item(), recall.item()

def calculate_acuracy_mode_two(model_pred, labels):
    # rank 'top' is the true
    precision = 0
    recall = 0
    top = 2
    model_pred = model_pred.cpu()
    labels = labels.cpu()
    # The prediction results are arranged in descending order according to the probability value, 
    # and the top results with the highest probability are taken as the prediction results of the model
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0,pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # precision
        precision += true_predict_num / top
        # recall
        recall += true_predict_num / target_one_num
    return precision.item(), recall.item()

def calculate_acuracy_mode_three(model_pred, labels):
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.int()
    pred_true = pred_result.eq(labels)
    pred_true = pred_true.float()
    acc = torch.sum(pred_true)/len(labels[0])
    return acc.item()

def cal_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(confusion_matrix)
        # 对角线上是正确预测的
        TP = confusion_matrix[i, i]
        # 列加和减去正确预测是该类的假阳
        FP = np.sum(confusion_matrix[:, i]) - TP
        # 行加和减去正确预测是该类的假阴
        FN = np.sum(confusion_matrix[i, :]) - TP
        # 全部减去前面三个就是真阴
        TN = ALL - TP - FP - FN
        metrics_result.append([TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)])
    return metrics_result