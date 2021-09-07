import torch
import torchvision 
import numpy as np 
import os
from torch.utils.data import Dataset
import cv2 as cv 
from tqdm import tqdm

#Label encoding, a slight difference in test and train due to the original dataset


"""****************Handling the dataset************************************** """

def get_train_data(path, labels):

    """
    Input: Path of the training set
    Output: 2 lists, one containing the images and one containig the labels
    """

    x = []
    y = []
    for pic in os.listdir(path):
        if not pic.startswith('.'):
            if pic in labels.keys():
                label = labels[str(pic)]
            else:
                label =  29
            count = 0

            for image_filename in tqdm(os.listdir(path + str(pic))):
                img_file = cv.imread(path + str(pic) + '/' + image_filename)
                if img_file is not None:
                    img_file = cv.resize(img_file,(200,200))
                    img_arr = np.asarray(img_file)
                    if(count < 100):
                      x.append(img_arr)
                      y.append(label)
                      count = count+1
    x = np.asarray(x)
    y = np.asarray(y)
    return x,y



def get_test_data(path, labels_test):

    """
    Input: Path of the training set
    Output: 2 lists, one containing the images and one containig the labels
    """

    x=[]
    y=[]
    for pic in os.listdir(path):
        print(pic)
        if not pic.startswith('.'):
            if pic[0] in labels_test.keys():
                label = labels_test[pic[0]]
                print(label)
            else:
                label =  29
            img_file = cv.imread(path + str(pic))
            if img_file is not None:
                  img_file = cv.resize(img_file,(200,200))
                  img_arr = np.asarray(img_file)
                  x.append(img_arr)
                  y.append(label)
    x = np.asarray(x)
    y = np.asarray(y)
    return x,y

def get_data(path, listData, labels):

    """
    Input: Path of the training set
    Output: 2 lists, one containing the images and one containig the labels
    """

    x = []
    y = []
    for directory in listData:
        for image_ in os.listdir(os.path.join(path,directory)):
            if not image_.startswith('.'):
                image = cv.imread('./artisan_dataset/'+directory+'/'+image_)
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                x.append(image)
                y.append(labels[directory])

    for image_ in os.listdir('./artisan_dataset/resized_img/'):
        if not image_.startswith('.'):
            image = cv.imread('./artisan_dataset/resized_img/'+image_)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            x.append(image)
            key = image_.split('_')
            label = labels[key[0]]
            y.append(label)

    return x,y



#Dataset class, a standard in the pytorch framework ->combined with dataloader will allow us to iterate
class MyDataset(Dataset):

    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)