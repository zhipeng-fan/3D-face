import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CACDDataset
from model import BFMReconstruction

BATCH_SIZE=8

train_set = CACDDataset("./data/CACD2000_train.hdf5")
val_set = CACDDataset("./data/CACD2000_val.hdf5")

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4)

model = BFMReconstruction()

