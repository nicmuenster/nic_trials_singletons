import os
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms as tv
from PIL import Image
from .sampler import build_mining_list_32, build_mining_list_32_debug
import pytorch_lightning as pl
import pandas as pd
import numpy as np

class miningDataset(data.Dataset):
    def __init__(self, root, csv_many, csv_singles, train, bucket_size=8, singleton_percentage=1.0,
                 singleton_percentage_end=0.0, keep_singletons=True, input_size=256):
        self.root = root
        self.train = train
        self.mining_list = build_mining_list_32(csv_many, bucket_size, csv_singles=csv_singles,
                                                singleton_percentage=singleton_percentage,
                                                singleton_percentage_end=singleton_percentage_end,
                                                keep_singletons=keep_singletons)
        self.num_samples = len(self.mining_list)
        if self.train:
                transform_list = [
                            tv.transforms.ColorJitter(brightness=0.4,saturation=0.4,contrast=0.4,hue=0.1),
                            tv.transforms.Resize([input_size, input_size], interpolation=2),
          #          '''tv.transforms.RandomApply([tv.transforms.RandomAffine(degrees=20,
          #                                          translate=(0.2, 0.2),
          #                                          scale=(0.8, 1.2),
          #                                          resample=Image.BICUBIC,
          #                                          shear=0.1,
          #                                          fillcolor=0)], 0.8),
          #
          #                  tv.transforms.RandomResizedCrop(227, scale=(0.8, 1.0), ratio=(0.95, 1.05), interpolation=2),'''
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]

        else:
                transform_list = [
                            tv.transforms.Resize([input_size, input_size], interpolation=2),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])]

        self.transform = tv.transforms.Compose(transform_list)

    def __getitem__(self, idx):
        # extract name and id
        name, patient_id = self.mining_list[idx]
        img = self.transform(Image.open(os.path.join(self.root, name)).convert('RGB'))

        return img, patient_id

    def __len__(self):
        return self.num_samples

class miningDataset_debug(data.Dataset):
    def __init__(self, root, csv_many, csv_singles, train, bucket_size=8,singleton_percentage=1.0,
                 singleton_percentage_end=0.0, keep_singletons=True, input_size=256):
        self.root = root
        self.train = train
        self.remain,self.garb_size, self.mining_list = build_mining_list_32_debug(csv_many, bucket_size,
                                                                                  csv_singles=csv_singles,
                                                                                  singleton_percentage=singleton_percentage,
                                                                                  singleton_percentage_end=singleton_percentage_end,
                                                                                  keep_singletons=keep_singletons)
        self.num_samples = len(self.mining_list)
        if self.train:
                transform_list = [
                            tv.transforms.Resize([input_size, input_size], interpolation=2),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]

        else:
                transform_list = [
                            tv.transforms.Resize([input_size, input_size], interpolation=2),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])]

        self.transform = tv.transforms.Compose(transform_list)

    def __getitem__(self, idx):
        # extract name and id
        name, patient_id = self.mining_list[idx]
        img = self.transform(Image.open(os.path.join(self.root, name)).convert('RGB'))

        return img, patient_id

    def __len__(self):
        return self.num_samples

class basicDataset(data.Dataset):
    def __init__(self, root, csv_many, csv_singles, train, input_size=256):
        self.root = root
        self.train = train
        patient_data_frame = pd.read_csv(csv_many)
        single_data_frame = pd.read_csv(csv_singles)

        name_array = patient_data_frame['Image Name'].to_numpy()
        id_array = patient_data_frame['Label'].to_numpy()
        single_name_array = patient_data_frame['Image Name'].to_numpy()
        single_id_array = patient_data_frame['Label'].to_numpy()
        self.id_array = np.array(list(single_id_array) + list(id_array))
        self.name_array = np.array(list(single_name_array) + list(name_array))
        self.num_samples = len(self.name_array)
        if self.train:
                transform_list = [
                            tv.transforms.ColorJitter(brightness=0.4,saturation=0.4,contrast=0.4,hue=0.1),
                            tv.transforms.Resize([input_size, input_size], interpolation=2),
          #          '''tv.transforms.RandomApply([tv.transforms.RandomAffine(degrees=20,
          #                                          translate=(0.2, 0.2),
          #                                          scale=(0.8, 1.2),
          #                                          resample=Image.BICUBIC,
          #                                          shear=0.1,
          #                                          fillcolor=0)], 0.8),
          #
          #                  tv.transforms.RandomResizedCrop(227, scale=(0.8, 1.0), ratio=(0.95, 1.05), interpolation=2),'''
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]

        else:
                transform_list = [
                            tv.transforms.Resize([input_size, input_size], interpolation=2),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])]

        self.transform = tv.transforms.Compose(transform_list)

    def __getitem__(self, idx):
        # extract name and id
        name = self.name_array[idx]
        patient_id = self.id_array[idx]
        img = self.transform(Image.open(os.path.join(self.root, name)).convert('RGB'))

        return img, patient_id

    def __len__(self):
        return self.num_samples


class MiningDataModule(pl.LightningDataModule):
    def __init__(self,
                 image_root,
                 csv_train,
                 csv_val,
                 csv_test,
                 csv_train_singles,
                 csv_val_singles,
                 csv_test_singles,
                 batch_size=32,
                 num_workers=0,
                 singleton_percentage=1.0,
                 singleton_percentage_end=0.0,
                 keep_singletons=True,
                 input_size=256
                 ):
        super().__init__()
        self.root_images = image_root
        self.path_train = csv_train
        self.path_val = csv_val
        self.path_test = csv_test
        self.single_csv_train = csv_train_singles
        self.single_csv_val = csv_val_singles
        self.single_csv_test = csv_test_singles
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.singleton_percentage = singleton_percentage
        self.singleton_percentage_end = singleton_percentage_end
        self.keep_singletons = keep_singletons
        self.input_size = input_size

    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_set = miningDataset(self.root_images,
                                           self.path_train,
                                           csv_singles=self.single_csv_train,
                                           train=True,
                                           singleton_percentage=self.singleton_percentage,
                                           singleton_percentage_end=self.singleton_percentage_end,
                                           keep_singletons=self.keep_singletons,
                                           input_size=self.input_size
                                           )
            self.val_set = basicDataset(self.root_images,
                                         self.path_val,
                                         self.single_csv_val,
                                         train=False,
                                         input_size=self.input_size)
        if stage == 'test':
            self.test_set = basicDataset(self.root_images,
                                          self.path_test,
                                          self.single_csv_test,
                                          train=False,
                                          input_size=self.input_size)

    # return the dataloader for each split
    def train_dataloader(self):
        # newly created so new pairs are build
        self.train_set = miningDataset(self.root_images,
                                       self.path_train,
                                       csv_singles=self.single_csv_train,
                                       train=True,
                                       singleton_percentage=self.singleton_percentage,
                                       singleton_percentage_end=self.singleton_percentage_end,
                                       keep_singletons=self.keep_singletons,
                                       input_size=self.input_size)
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return test_loader