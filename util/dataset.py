import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None):
    assert split in [0, 1, 2, 999]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    image_label_list = []  
    list_read = open(data_list,encoding='UTF-8-sig').readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split()
        image_name = os.path.join(data_root, line_split[0])
        temp = line_split[0].replace('Images/', 'GT/')
        label_name = os.path.join(data_root, temp)
        label_name = label_name.replace('jpg', 'png')
        label_class = line_split[1]
        item = (image_name, label_name, label_class)

        image_label_list.append(item)

        sub_class_file_list[int(label_class)].append(item)
                    
    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list


class SemData(Dataset):
    def __init__(self, split=2, shot=1, data_root=None, data_list=None, nom_list=None, transform=None, mode='train',
                 use_coco=False, use_split_coco=False):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root
        self.class_list = list(range(1, 13))
        if self.split == 2:
            self.sub_list = list(range(1, 9))
            self.sub_val_list = list(range(9, 13))
        elif self.split == 1:
            self.sub_list = list(range(1, 5)) + list(range(9, 13))
            self.sub_val_list = list(range(5, 9))
        elif self.split == 0:
            self.sub_list = list(range(5, 13))
            self.sub_val_list = list(range(1, 5))

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        if self.mode == 'train':
            self.data_list, self.sub_class_file_list= make_dataset(split, data_root, data_list, self.sub_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        elif self.mode == 'val':
            self.data_list, self.sub_class_file_list= make_dataset(split, data_root, data_list, self.sub_val_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path, label_class = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label[label != 255] = 0
        label[label == 255] = 1
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))          

        assert len(label_class) > 0
        class_chosen = label_class
        file_class_chosen = self.sub_class_file_list[int(class_chosen)]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path

            while ((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1, num_file)-1
                support_image_path, support_label_path,_ = file_class_chosen[support_idx]

            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        label1=[]
        support_label_list = []
        subcls_list = []
        for k in range(self.shot):  
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(int(class_chosen)))
            else:
                subcls_list.append(self.sub_val_list.index(int(class_chosen)))
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            support_label[support_label != 255] = 0
            support_label[support_label == 255] = 1

            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot

        raw_label = label.copy()
        if self.transform is not None:
            image, label = self.transform(image, label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)
        if self.mode == 'train':
            return image, label, s_x, s_y, subcls_list
        else:
            return image, label, s_x, s_y, subcls_list, raw_label