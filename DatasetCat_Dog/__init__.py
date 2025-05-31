import os
import pickle
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

class Dataset_folder:
    def __init__(self, root, train, transform = None):
        '''
        root: đường dẫn thư mục gốc
        train: xác định muốn lấy bộ train hay test
        transform: đưa ảnh về có kích thước hay chuẩn hóa như nào? có thể không dùng transform
        '''
        self.transform:str = transform
        self.root:str = root
        labels = [0,1]
        self.path_images = []
        self.labels = []
        if train:
            paths = os.path.join(root,"train")
            path_folders = os.listdir(paths)
            for label, path_folder in zip(labels,path_folders):
                files = os.path.join(paths,path_folder)
                path_files = os.listdir(files)
                for path_file in path_files:
                    self.path_images.append(os.path.join(files,path_file))
                    self.labels.append(label)
        else:
            paths = os.path.join(root,"test")
            path_folders = os.listdir(paths)
            for label,path_folder in zip(labels,path_folders):
                files = os.path.join(paths,path_folder)
                path_files = os.listdir(files)
                for path_file in path_files:
                    self.path_images.append(os.path.join(files,path_file))
                    self.labels.append(label)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,item):
        img_path = self.path_images[item]
        label = self.labels[item]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


                


            
            
            
            
