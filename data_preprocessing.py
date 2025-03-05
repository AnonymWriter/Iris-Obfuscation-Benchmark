import os
import tqdm
import json
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.v2 as transforms
    
class OpenEDS2019Dataset(torch.utils.data.Dataset):
    """
    Self-defined dataset, used for iris style transfer.
    """
    
    def __init__(
        self, 
        c_images: list[torch.Tensor], 
        c_labels: list[int],
        c_masks_gt: list[torch.Tensor],
        device: str = 'cuda:0',
        ) -> None:
        """
        Arguments:
            c_images (list[torch.Tensor]): image tensors. 
            c_labels (list[int]): class labels.
            c_masks_gt (list[torch.Tensor]): image segmentation ground truth labels.
            device (str): CPU or GPU.
        """
        
        assert(len(c_images) == len(c_labels) == len(c_masks_gt))
        
        # original samples
        self.c_images = torch.stack(c_images).to(device)
        self.c_labels = torch.as_tensor(c_labels).long().to(device)
        self.c_masks_gt = torch.stack(c_masks_gt).to(device)
        
        # attacker samples
        self.s_images = []
        self.s_labels = []
                        
        print('processing data...')
        for c_label in tqdm.tqdm(self.c_labels, total = len(self.c_labels)):
            # randomly sample a attacker image, which is an eye image of another user
            s_idx = sample_other(c_label, c_labels)
            self.s_labels.append(self.c_labels[s_idx])
            s_img = self.c_images[s_idx].to(device)
            self.s_images.append(s_img)
            
    def __len__(self) -> int:
        """
        Returns:
            (int): size of dataset.
        """
        return len(self.c_labels)

    def __getitem__(self, idx: int
                    ) -> tuple[
                        torch.Tensor,              # original image tensor
                        int,                       # original image class label
                        torch.Tensor,              # original image ground truth segmentation label
                        torch.Tensor,              # attacker image tensor
                        int]:                      # attacker image class label
        """
        Arguments:
            idx (int): index.

        Returns:
        (torch.Tensor): original image tensor
        (int): original image class label
        (torch.Tensor): original image ground truth segmentation label
        (torch.Tensor): attacker image tensor
        (int): attacker image class label
        """
        
        return self.c_images[idx],      \
               self.c_labels[idx],      \
               self.c_masks_gt[idx],    \
               self.s_images[idx],      \
               self.s_labels[idx]

def sample_other(label: int, labels: list[int]) -> int:
    """
    Given a class label, sample a random sample of another class.

    Arguments:
        label (int): class label.
        labels (list[int]): sample label list.

    Returns:
        idx (int): index of sample.
    """
    idx = random.randrange(len(labels))
    while labels[idx] == label:
        idx = random.randrange(len(labels))
    return idx

def load_data_openeds2019(
    test_split_ratio: float = 0.2, 
    image_paths: list[str] = ['../data/openeds2019/Semantic_Segmentation_Dataset/train/images/', 
                            '../data/openeds2019/Semantic_Segmentation_Dataset/validation/images/',
                            '../data/openeds2019/Semantic_Segmentation_Dataset/test/images/'],
    json_paths:  list[str] = ['../data/openeds2019/OpenEDS_train_userID_mapping_to_images.json', 
                            '../data/openeds2019/OpenEDS_validation_userID_mapping_to_images.json',
                            '../data/openeds2019/OpenEDS_test_userID_mapping_to_images.json'],
    seg_paths:   list[str] = ['../data/openeds2019/Semantic_Segmentation_Dataset/train/labels/', 
                            '../data/openeds2019/Semantic_Segmentation_Dataset/validation/labels/',
                            '../data/openeds2019/Semantic_Segmentation_Dataset/test/labels/'],
    ) -> tuple[
        list[torch.Tensor], # train images tensors 
        list[int],          # train image class labels
        list[torch.Tensor], # train ground truth segmentation labels
        list[torch.Tensor], # test images tensors
        list[int],          # test image class labels
        list[torch.Tensor], # test ground truth segmentation labels
        int                 # number of classes
    ]:
    """
    Load OpenEDS2019 dataset.

    Arguments:
        test_split_ratio (float): train-test-split ratio.
        image_paths (list[str]): image folder paths.
        json_paths (list[str]): user-image mapping json file paths.
        seg_paths (list[str]): grount truth segmentation folder paths.

    Returns:
        test_x (list[torch.Tensor]): test images tensors.
        test_y (list[int]): test image class labels.
        test_m (list[torch.Tensor]): test ground truth segmentation labels
        class_count (int): number of classes.
    """
    
    test_x, test_y, test_m = [], [], []
    class_count = 0
    
    # PIL to tensor
    t = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale = True)])

    for i_folder, j_path, m_folder in zip(image_paths, json_paths, seg_paths):
        with open(j_path, 'r') as file:
            mappings= json.load(file)
        
        # create image-class and image-split dictionaries
        img_class_dict = {}
        img_train_dict = {}
        for m in mappings:
            # id = m['id']
            imgs = m['semantic_segmenation_images']
            if len(imgs) <= 2: # skip users with too few samples
                continue
            
            train_imgs, test_imgs = torch.utils.data.random_split(imgs, [1 - test_split_ratio, test_split_ratio])
            for i in range(len(imgs)):
                img_class_dict[imgs[i]] = class_count
                img_train_dict[imgs[i]] = i in train_imgs.indices
            class_count += 1

        # load images and determine their classes
        img_paths = os.listdir(i_folder)
        for i_path in img_paths:
            if i_path not in img_class_dict: # skipped users
                continue
            
            # load eye image and get class label (each user is a class)
            p = i_folder + i_path
            img = Image.open(p).convert('L')
            img = t(img)
            img_class = img_class_dict[i_path] 
            img_train = img_train_dict[i_path] # whether this image is in training set or test set

            # load ground truth segmentation label
            m_path = i_path[:-4] + '.npy' # file name from .jpg to .npy
            img_mask = torch.from_numpy(np.load(m_folder + m_path))
            
            # if img_train:
            #     train_x.append(img)
            #     train_y.append(img_class)
            #     train_m.append(img_mask)
            if not img_train: # else:
                test_x.append(img)
                test_y.append(img_class)
                test_m.append(img_mask)
    
    return test_x, test_y, test_m, class_count # train_x, train_y, train_m, 

def load_data_openeds2020(
    data_path: str = '../data/openeds2020/openEDS2020-GazePrediction/',
    postfix: str = 'test/',
    device: str = 'cpu',
    ) -> tuple[
        torch.Tensor, # images
        torch.Tensor, # gaze vectors
        ]:
    """
    Load OpenEDS2020 dataset (gaze estimation part).

    Arguments:
        data_path (str): dataset folder path.
        postfix (str): train, validation, or test.
        device (str): CPU or GPU.

    Returns:
        images (torch.Tensor): images.
        labels (torch.Tensor): gaze vectors.
    """
    
    # transform
    t = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale = True)])
    
    images = []
    labels = []

    sequence_names = sorted(os.listdir(data_path + postfix + 'sequences/'))
    for sequence_name in sequence_names:
        # get sorted image names
        img_names = sorted(os.listdir(data_path + postfix + 'sequences/' + sequence_name))

        # read label
        label = pd.read_csv(data_path + postfix + 'labels/' + sequence_name + '.txt', header = None)
        label = label.iloc[:, 1:] # drop index column
        label = torch.tensor(label.values, dtype = torch.float32)

        # number of images should be equal to label length for train and valid sets, and 5 frames less for test set
        assert(len(img_names) == len(label) or len(img_names) == len(label) - 5)
        labels.append(label[:len(img_names)])
        
        for img_name in img_names:
            img = Image.open(data_path + postfix + 'sequences/' + sequence_name + '/' + img_name).convert('L')
            img = t(img)                
            images.append(img)

    images = torch.stack(images).to(device)
    labels = torch.cat(labels).to(device)
    
    return images, labels