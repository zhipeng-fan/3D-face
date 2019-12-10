import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from skimage import io
import soft_renderer as sr

from dataset import CACDDataset
from model import BaseModel
from loss import BFMFaceLoss

# -------------------------- Hyperparameter ------------------------------
BATCH_SIZE=64
NUM_EPOCH=20
LR=1e-4
VERBOSE_STEP=50
VIS_BATCH_IDX=5
LMK_LOSS_WEIGHT=20

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

val_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

inv_normalize = transforms.Compose([
                    transforms.Normalize(
                                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                std=[1/0.229, 1/0.224, 1/0.255])
    ])
                    

# -------------------------- Dataset loading -----------------------------
train_set = CACDDataset("./data/CACD2000_val.hdf5", train_transform)
val_set = CACDDataset("./data/CACD2000_val.hdf5", val_transform)

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

# -------------------------- Model loading ------------------------------
model = BaseModel()
model.to(device)

# -------------------------- Optimizer loading --------------------------
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_schduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5)

# ------------------------- Loss loading --------------------------------
camera_distance = 2.732
elevation = 0
azimuth = 180

renderer = sr.SoftRenderer(image_size=224, sigma_val=1e-4, aggr_func_rgb='hard', 
                            camera_mode='look_at', viewing_angle=30,
                            perspective=True, light_intensity_directionals=0)

renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
face_loss = BFMFaceLoss(renderer, LMK_LOSS_WEIGHT, device)
# ------------------------- plot visualization --------------------------
def visualize_batch(gt_imgs, recon_imgs):
    gt_imgs = gt_imgs.cpu()
    recon_imgs = recon_imgs.cpu()
    bs = gt_imgs.shape[0]
    num_cols = 4
    num_rows = int(bs/num_cols)

    canvas = np.zeros((num_rows*224, num_cols*224*2, 3))
    img_idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            gt_img = inv_normalize(gt_imgs[img_idx]).permute(1,2,0).numpy()
            recon_img = inv_normalize(recon_imgs[img_idx,:3,:,:]).permute(1,2,0).numpy()
            canvas[i*224:(i+1)*224, j*224*2:(j+1)*224*2-224, :3] = gt_img
            canvas[i*224:(i+1)*224, j*224*2+224:(j+1)*224*2, :4] = recon_img
            img_idx += 1
    return (np.clip(canvas,0,1)*255).astype(np.uint8)


# ------------------------- train ---------------------------------------
def train(model, epoch):
    model.train()
    running_loss = []
    running_img_loss = []
    running_lmk_loss = []
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        img, lmk = data
        img = img.to(device); lmk = lmk.to(device)

        optimizer.zero_grad()
        recon_params = model(img)
        loss,img_loss,lmk_loss,_ = face_loss(recon_params, img, lmk)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        running_img_loss.append(img_loss.item())
        running_lmk_loss.append(lmk_loss.item())
      
        if i % VERBOSE_STEP == 0 and i!=0:
            print ("Epoch: {:02}/{:02} Progress: {:06}/{:06} Loss: {:.6f} Img Loss: {:.6f} LMK Loss: {:.6f}".format(epoch+1, 
                                                                                                                    NUM_EPOCH, 
                                                                                                                    i, 
                                                                                                                    len(train_dataloader), 
                                                                                                                    np.mean(running_loss),
                                                                                                                    np.mean(running_img_loss),
                                                                                                                    np.mean(running_lmk_loss)))
            running_loss = []

    return model

# ------------------------- eval ---------------------------------------
def eval(model, epoch):
    model.eval()
    all_loss_list = []
    img_loss_list = []
    lmk_loss_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            img, lmk = data
            img = img.to(device); lmk = lmk.to(device)
            recon_params = model(img)
            all_loss,img_loss,lmk_loss,recon_img=face_loss(recon_params, img, lmk)
            all_loss_list.append(all_loss.item())
            img_loss_list.append(img_loss.item())
            lmk_loss_list.append(lmk_loss.item())
            if i == VIS_BATCH_IDX:
                visualize_image = visualize_batch(img, recon_img)

    print ("-"*50, " Test Results ", "-"*50)
    all_loss = np.mean(all_loss_list)
    img_loss = np.mean(img_loss_list)
    lmk_loss = np.mean(lmk_loss_list)
    print ("Epoch {:02}/{:02} image loss: {:.6f} landmark loss {:.6f}".format(epoch+1, NUM_EPOCH, img_loss, lmk_loss))
    print ("-"*116)
    return all_loss, img_loss, lmk_loss, visualize_image

for epoch in range(NUM_EPOCH):
    model = train(model, epoch)
    all_loss, img_loss, lmk_loss, visualize_image = eval(model, epoch)
    lr_schduler.step(all_loss)
    io.imsave("./result/Epoch:{:02}_AllLoss:{:.6f}_ImgLoss:{:.6f}_LMKLoss:{:.6f}.png".format(epoch, all_loss, img_loss, lmk_loss), visualize_image)
    model2save = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(model2save, "./model_result/epoch_{:02}_loss_{:.4f}_lmk_loss_{:.4f}_img_loss{:.4f}.pth".format(epoch+1, img_loss+LMK_LOSS_WEIGHT*lmk_loss, img_loss, lmk_loss, ))

