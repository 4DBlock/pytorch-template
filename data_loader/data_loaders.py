from torchvision import datasets, transforms
from base import BaseDataLoader


from torch.utils.data import Dataset
import cv2
import torch
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread("C:/Users/yoon/Desktop/pytorch-template/data/open/"+img_path) #수정 예정
        
        if self.transforms is not None:
            image = self.transforms(image)

        if self.label_list is not None:
            label = torch.FloatTensor(self.label_list[index])
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)

class FourDBlockDataLoader(BaseDataLoader):
    def __init__(self,data_dir, batch_size, shuffle=True, validation_split=0.2, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])# albumentation 추가
        self.data_dir = data_dir
        self.train_df = pd.read_csv(data_dir+"train.csv") # target(y), index, image_path
        self.img_path_list = self.train_df["img_path"]
        self.label_list = [list(self.train_df.iloc[i][2:]) for i in range(len(self.train_df))] # 수정
        

        self.dataset = CustomDataset(img_path_list=self.img_path_list, label_list=self.label_list,transforms=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


