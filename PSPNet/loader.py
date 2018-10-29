# coding: utf-8

import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class Data_Preprocess(Dataset):

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='',
                 transform=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        
        self.noseq=[]
        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'

       
        # Initialize the original DAVIS splits for training the parent network
        with open(os.path.join(db_root_dir, fname + '.txt')) as f:
            seqs = f.readlines()
            # list!!!!!!!!!!!
            #print type(seqs)
            img_list = []
            labels = []

            for seq in seqs:
                #catch all file in the dir
                images = np.sort(os.listdir(os.path.join(db_root_dir,seq.strip(),'JPEGImages')))
                images_path = list(map(lambda x: os.path.join(seq.strip(),'JPEGImages',x), images))
               
                img_list.extend(images_path)
                lab=list(map(lambda x: x.replace('jpg', 'png'), images))
                lab_path = list(map(lambda x: os.path.join(seq.strip(),'SegmentationClass', x), lab))
                labels.extend(lab_path)
       
        assert (len(labels) == len(img_list))
        self.img_list = img_list
        self.labels = labels
        print('Done initializing ' + fname + ' Dataset')
    

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img1, lbl1 = self.make_img_gt_pair(idx) 
        img = np.array(img1, dtype=np.float32)            
        lbl = np.expand_dims(lbl1, axis=2)
        sample = {'image': img, 'gt': lbl}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
            
    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))[...,::-1]
        label=cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 2)
        gt=np.array(label, dtype=np.float32)
        if label is None:
            gt = np.zeros(img.shape[:-1], dtype=np.float32)
            

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)      
        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])

