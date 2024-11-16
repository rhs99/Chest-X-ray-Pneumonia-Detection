import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from .utils.transforms import load_image_and_transform


DIR = 'xray_299'
WORKERS = 2


def get_data(stage):
    img_names, labels = [], []
    for i, cls in enumerate(('normal', 'bacteria', 'virus')):
        ddir = os.path.join(DIR, stage, cls)
        dlist = [os.path.join(ddir, file) for file in os.listdir(ddir)]
        img_names += dlist
        labels += [i] * len(dlist)
    return img_names, labels


# def custom_collate(batch):
#     x = [item[0] for item in batch]
#     y = [item[1] for item in batch]
#     y = torch.FloatTensor(y)
#     return [x, y]


class CustomDataset(Dataset):
    def __init__(self, img_names, labels):
        super().__init__()
        self.img_names = img_names
        self.y = F.one_hot(torch.tensor(labels)).float()

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return load_image_and_transform(self.img_names[index]), self.y[index]


class SimpleDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_data = CustomDataset(*get_data('train'))
        self.val_data = CustomDataset(*get_data('val'))
        self.test_data = CustomDataset(*get_data('test'))

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=WORKERS,
            # collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=WORKERS,
            # collate_fn=custom_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=WORKERS,
            # collate_fn=custom_collate
        )
