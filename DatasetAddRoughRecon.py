import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
import soft_renderer as sr

from loss import BFMFaceLossUVMap
from dataset import CACDDataset
from model import BaseModel

import h5py

BATCH_SIZE = 64
MODEL_LOAD_PATH="./model_result_full/epoch_19_loss_0.5796_Img_loss_0.0098_LMK_loss0.5698_Recog_loss0.0023.pth"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

val_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

inv_normalize = transforms.Compose([
                    transforms.Normalize(
                                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                std=[1/0.229, 1/0.224, 1/0.255])
    ])
                    

# -------------------------- Dataset loading -----------------------------
train_set = CACDDataset("./data/CACD2000_train.hdf5", val_transform, inv_normalize)
val_set = CACDDataset("./data/CACD2000_val.hdf5", val_transform, inv_normalize)

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

base_model = BaseModel(IF_PRETRAINED=True)
base_model.to(device)
base_model.load_state_dict(torch.load(MODEL_LOAD_PATH)['model'])
base_model.eval()

# ------------------------- Loss loading --------------------------------
camera_distance = 2.732
elevation = 0
azimuth = 0

renderer = sr.SoftRenderer(image_size=250, sigma_val=1e-4, aggr_func_rgb='hard', 
                            camera_mode='look_at', viewing_angle=30, fill_back=False,
                            perspective=True, light_intensity_ambient=1.0, light_intensity_directionals=0)

renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
# face_loss = BFMFaceLoss(renderer, LMK_LOSS_WEIGHT, RECOG_LOSS_WEIGHT, device)
face_loss = BFMFaceLossUVMap(0, 0, 224, renderer, 0, 0, device).to(device)

def prepare_dataset(name, dataloader, size):
    with h5py.File("./data/CACD2000_{}_bfm_recon.hdf5".format(name), 'w') as f:
        image_dset = f.create_dataset("bfm_recon", shape=(size, 250, 250 , 4), dtype='uint8')
        params_dset = f.create_dataset("bfm_param", shape=(size, 258), dtype="float32")
        idx = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                in_img, _, _ = data
                in_img = in_img.to(device)
                recon_params = base_model(in_img)
                recon_img = face_loss.reconst_img(recon_params)
                recon_img = recon_img.permute(0,2,3,1).cpu().numpy()*255.
                recon_params =recon_params.cpu().numpy()
                for i in range(recon_params.shape[0]):
                    params_dset[idx] = recon_params[i]
                    image_dset[idx] = recon_img[i]
                    idx += 1

print ("Prepare validation set:")
prepare_dataset("val", val_dataloader, len(val_set))
print ("Prepare train set:")
prepare_dataset("train", train_dataloader, len(train_set))