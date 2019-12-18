# import BFM
# import os
# import trimesh
# import numpy as np
# import utils
# import torch
# from skimage import io

# # if not os.path.isfile("./BFM/BFM_model_front.mat"):
# #     utils.transferBFM09()

# # bfm_face_model = BFM.BFM()
# # print (bfm_face_model.point_buf[-1,-1])
# # mesh = trimesh.Trimesh(vertices=bfm_face_model.meanshape.reshape(-1,3), 
# #                        faces=bfm_face_model.tri.reshape(-1,3)-1, 
# #                        vertex_colors=bfm_face_model.meantex.reshape(-1,3))

# # mesh.show()

# # def morph_face(bfm_model, id_param=None, ex_param=None, tex_param=None):
# #     if id_param is None:
# #         id_param = np.random.randn(80, 1)
# #     if ex_param is None:
# #         ex_param = np.random.randn(64, 1)
# #     if tex_param is None:
# #         tex_param = np.random.randn(80,1)

# #     vertices = bfm_face_model.meanshape+np.dot(bfm_face_model.idBase, id_param)#+np.dot(bfm_face_model.exBase, ex_param)
# #     colors = np.clip(bfm_face_model.meantex+np.matmul(bfm_face_model.texBase, tex_param), 0, 255)
# #     mesh = trimesh.Trimesh(vertices=vertices.reshape(-1,3), 
# #                             faces=bfm_face_model.tri.reshape(-1,3)-1, 
# #                             vertex_colors=colors.reshape(-1,3))

# #     mesh.show()


# # morph_face(bfm_face_model)

# # print (bfm_face_model.meanshape.shape)
# # print (bfm_face_model.texBase.shape)
# # print (bfm_face_model.exBase.shape)
# # print (bfm_face_model.meantex.shape)
# # print (bfm_face_model.idBase.shape)

# # print (bfm_face_model.tri.shape)
import torch
import soft_renderer as sr
# from loss import BFMFaceLoss
# from BFM import BFM_torch

camera_distance = 2.732
elevation = 0
azimuth = 180

# renderer = sr.SoftRenderer(image_size=512, sigma_val=1e-4, aggr_func_rgb='hard', 
#                             camera_mode='look_at', viewing_angle=30,
#                             perspective=True, light_intensity_directionals=0)

# renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)

# # bs = 10

# face_model = BFM.BFM_torch()

# bs=2

# camera_distance = 2.732
# elevation = 0
# azimuth = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = sr.SoftRenderer(image_size=250, sigma_val=1e-4, aggr_func_rgb='hard', 
                            camera_mode='look_at', viewing_angle=30, fill_back=False,
                            perspective=True, light_intensity_ambient=1.0, light_intensity_directionals=0)

renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
# face_loss = BFMFaceLoss(renderer, 20, device)

# d3d_param =torch.Tensor([[-1.82443619e+00,  4.34013456e-02, -7.67325342e-01,
#         -1.35242629e+00, -5.11713028e-01, -4.10691090e-02,
#          5.26137531e-01, -9.85630751e-01, -4.09944296e-01,
#         -4.43508774e-01, -1.52504385e+00, -4.70615327e-01,
#         -2.29804009e-01,  9.98994768e-01,  2.24300519e-01,
#         -2.83951283e-01, -9.16861773e-01, -2.70456612e-01,
#         -2.22548082e-01,  1.06795633e+00, -2.08728766e+00,
#          3.68633419e-01, -5.50950646e-01, -6.04444742e-01,
#          2.57286459e-01, -6.51607811e-01, -7.06202447e-01,
#          3.79041910e-01, -7.81520903e-01, -4.73769784e-01,
#         -1.54784039e-01,  6.21756539e-02, -5.07417321e-01,
#          4.91097048e-02,  5.00670433e-01, -4.66783971e-01,
#         -5.61647415e-01, -3.00246716e-01, -4.55036104e-01,
#          1.14828372e+00,  1.13247824e+00,  8.20224881e-01,
#          1.92450976e+00, -1.72113746e-01, -1.23266548e-01,
#         -8.74411702e-01,  1.41375437e-01,  1.53811380e-01,
#         -1.32311314e-01, -9.97675210e-02, -8.89653206e-01,
#          3.78128171e-01, -1.76426813e-01,  1.62149787e-01,
#          4.02987987e-01, -3.16322356e-01,  8.53126124e-03,
#         -4.90177959e-01, -5.21494150e-01,  6.98217273e-01,
#          8.14209938e-01,  9.58727479e-01,  3.61837178e-01,
#         -5.86323917e-01, -2.57825464e-01, -2.98531622e-01,
#          1.14213288e+00, -1.02846697e-01, -3.00282717e-01,
#          9.96255130e-03,  4.00781482e-01, -5.20598412e-01,
#          9.73515287e-02,  1.17015220e-01, -9.93605435e-01,
#          5.15048146e-01,  7.34730959e-02,  5.63720614e-03,
#          5.86912930e-01,  8.54108751e-01, -7.23754048e-01,
#          3.12979845e-03, -5.48629463e-01, -1.56913877e-01,
#          5.21444082e-02,  1.42312676e-01, -1.32376716e-01,
#          1.45427942e-01, -5.05696237e-02, -4.82063830e-01,
#         -1.46952188e+00, -4.62830961e-01,  3.45179528e-01,
#         -3.92011493e-01, -2.19142154e-01, -6.41663194e-01,
#         -1.44832045e-01, -2.53186285e-01, -6.07453100e-02,
#         -1.91909984e-01, -7.46585846e-01,  5.85532188e-02,
#         -4.30736542e-01,  3.17119807e-03, -5.59396446e-01,
#          3.07311445e-01, -1.48433417e-01, -1.42528474e-01,
#          1.69070691e-01, -7.90720284e-02,  9.48285460e-02,
#         -9.85537320e-02,  2.21763462e-01,  5.57182245e-02,
#         -9.89370644e-02, -5.39290905e-02, -4.49412584e-01,
#         -2.61910141e-01,  8.02617148e-02,  1.56347752e-01,
#         -1.22646332e-01,  2.02193737e-01, -1.44258857e-01,
#          1.66003436e-01,  5.24997786e-02, -4.28470522e-02,
#         -4.04243320e-02,  5.81844747e-02, -8.99919495e-03,
#         -7.01051429e-02,  2.89188530e-02,  4.93676886e-02,
#         -5.13751656e-02, -1.53318215e-02,  1.32013215e-02,
#          2.66979411e-02,  2.23326720e-02,  3.74747440e-04,
#         -1.25281010e-02,  4.10938524e-02,  5.87595515e-02,
#         -1.67628806e-02,  1.70406252e-02, -2.22928319e-02,
#         -1.82746267e+00, -1.17891467e+00,  2.28146935e+00,
#          1.88160253e+00, -1.31716955e+00, -2.23194003e+00,
#         -1.95896581e-01,  2.21401668e+00, -7.82479048e-01,
#         -1.03034484e+00,  3.28301954e+00, -7.81207085e-01,
#         -1.40463448e+00, -1.07121080e-01,  1.61312604e+00,
#          8.12099427e-02,  2.86755276e+00,  4.43824857e-01,
#          1.69761932e+00,  1.72108591e+00, -9.33292866e-01,
#          1.07835662e+00, -4.61664051e-02,  2.35164523e+00,
#          8.35312486e-01,  1.02654922e+00, -3.46728027e-01,
#         -3.10354638e+00, -4.96300817e-01, -3.01983476e-01,
#         -8.71175468e-01, -7.47978032e-01, -1.68082869e+00,
#         -1.54380023e+00, -3.01975846e+00, -1.91685915e+00,
#          1.43252921e+00, -6.07571721e-01, -8.28522384e-01,
#         -3.63177299e-01, -2.79242635e-01,  1.64517808e+00,
#         -2.35608172e+00, -1.26207149e+00,  2.87478304e+00,
#         -8.42347145e-01,  4.84829485e-01,  2.31910229e+00,
#         -1.55147254e+00,  1.50957203e+00, -8.79077315e-01,
#          2.01758432e+00, -3.11831206e-01,  6.00779057e+00,
#         -4.33622390e-01, -1.90958530e-02, -7.62203410e-02,
#         -3.16076934e-01,  7.04967141e-01, -1.07426405e+00,
#          1.06825829e+00, -9.65462029e-01, -1.58212751e-01,
#          4.48210192e+00, -1.86127436e+00, -1.05268824e+00,
#         -5.75243056e-01,  1.27161050e+00,  1.20546222e+00,
#          2.51234174e+00,  2.46903110e+00, -9.50164855e-01,
#          1.23225284e+00,  2.12471075e-02, -5.32995522e-01,
#         -1.35419798e+00,  2.03368336e-01, -2.04560876e+00,
#          2.64470339e-01,  2.47866607e+00, -1.95762709e-01,
#         -4.01266105e-03,  1.22496918e-01, -6.92362487e-02,
#         -7.72316903e-02,  2.52706230e-01,  1.38839364e-01,
#          7.79333990e-03,  6.49487451e-02,  3.40749137e-03,
#         -2.53637061e-02, -3.92647386e-02, -4.91852164e-02,
#         -6.14390671e-02,  2.82604098e-01,  1.54526502e-01,
#          8.72542709e-03,  6.60056770e-02,  1.31886518e-02,
#          9.18350648e-04, -4.47486714e-02, -4.63662855e-02,
#         -4.76260968e-02,  3.04713935e-01,  1.82856739e-01,
#          2.00223960e-02,  8.37084204e-02,  4.49283235e-02,
#          3.31540629e-02, -4.02633175e-02, -3.65127367e-03,
#          7.16769276e-03,  3.77793729e-01]])

# params = torch.zeros(bs, 258)
# params[:,:-1]= d3d_param.expand(bs,-1)
# img = torch.randn(bs, 3, 224, 224)
# lmk = torch.randn(bs, 68, 2)

# # loss,img_loss,lmk_loss,recon_img = face_loss(params.cuda(), img.cuda(), lmk.cuda())

# # import matplotlib.pyplot as plt

# # for i in range(bs):
# #     plt.imshow(recon_img[i].permute(1,2,0).cpu().numpy())
# #     plt.show()

# shape = face_model.get_shape(torch.randn(bs,80), torch.zeros(bs,64))
# albedo = face_model.get_texture(torch.randn(bs,80))

# shape = shape.reshape(bs,-1,3)
# shape[:,:,-1] *= -1
# albedo = albedo.reshape(bs,-1,3)/255.

# face_id = face_model.tri-1
# face_id = face_id[None,:,:].expand(bs,-1,-1)

# mesh = trimesh.Trimesh(vertices=shape[0].numpy(), 
#                        faces=1,
#                        vertex_colors=np.clip(albedo[0].numpy(),0,1))
# mesh.show()

# # print (shape[:, face_id.long(), :].shape)

# # v1 = torch.gather(shape, dim=1, index=face_id[:,:,:1].expand(-1,-1,3).long())

# # bs_id = 5
# # face_idx = 10
# # print (shape[bs_id, face_id[bs_id,face_idx,0].long(),:])
# # print (v1[bs_id,face_idx,:])

# # print (face_model.keypoints.shape)


# # rot_param = torch.randn(bs,3)
# # rot_mat = face_model.compute_rotation_matrix(rot_param)
# # print (shape.shape)
# # # shape = torch.bmm(shape, rot_mat)



# mesh, image = renderer(shape.cuda(), torch.flip(face_model.tri.reshape(-1,3)-1, dims=[1])[None,:,:].expand(bs,-1,-1).cuda(), albedo.cuda(), texture_type="vertex")
# # import pdb; pdb.set_trace()

# lmk = mesh.vertices[:, face_model.keypoints.long(), :].cpu().numpy()
# print (lmk.shape)
# lmk[:,:,:2] = (lmk[:,:,:2]+1)*250/2
# lmk[:,:,1] = 250-lmk[:,:,1]


# # io.imsave("./test_ori.png",image[0].permute(1,2,0).cpu().numpy())


# # shape += torch.Tensor([0.5,0,0])

# # mesh = trimesh.Trimesh(vertices=shape[0].numpy(), 
#                     #    faces=face_model.tri.reshape(-1,3)-1, 
#                     #    vertex_colors=np.clip(albedo[0].numpy(),0,1))
# # mesh.show()

# # image = renderer(shape[0].cuda(), (face_model.tri.reshape(-1,3)-1).cuda(), (albedo[0]).cuda(), texture_type="vertex")
# # io.imsave("./test_shift.png",image[0].permute(1,2,0).cpu().numpy())

# # image = renderer(shape.reshape(-1,3).cuda(), (face_model.tri.reshape(-1,3)-1).cuda(), (albedo.reshape(-1,3)/255.).cuda(), texture_type="vertex")

# # print (image[0].permute(1,2,0).cpu().numpy().shape)

# # io.imsave("./test.png",image[0].permute(1,2,0).cpu().numpy())

# from torchvision import transforms

# train_transform = transforms.Compose([
#                     transforms.ToPILImage(),
#                     transforms.RandomResizedCrop(224),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ])

# val_transform = transforms.Compose([
#                     transforms.ToPILImage(),
#                     transforms.CenterCrop(224),
#                     transforms.ToTensor(),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

# inv_normalize = transforms.Compose([
#                     transforms.Normalize(
#                                 mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
#                                 std=[1/0.229, 1/0.224, 1/0.255])
#     ])

# # from dataset import CACDDataset
# import matplotlib.pyplot as plt
# # TrainSet = CACDDataset('./data/CACD2000_test.hdf5', train_transform, inv_normalize)
# # img,gt_img,lmk = TrainSet[1]
# plt.figure()
# plt.imshow(image[0].permute(1,2,0).cpu().numpy())
# plt.scatter(lmk[0,:20,0], lmk[0,:20,1])
# plt.show()


# from torchvision import transforms

# train_transform = transforms.Compose([
#                     transforms.ToPILImage(),
#                     transforms.Resize(224),
#                     transforms.ToTensor(),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ])

# val_transform = transforms.Compose([
#                     transforms.ToPILImage(),
#                     transforms.Resize(224),
#                     transforms.ToTensor(),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

# inv_normalize = transforms.Compose([
#                     transforms.Normalize(
#                                 mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
#                                 std=[1/0.229, 1/0.224, 1/0.255])
#     ])

# from dataset import CACDDataset
# import matplotlib.pyplot as plt
# TrainSet = CACDDataset('./data/CACD2000_test.hdf5', train_transform, inv_normalize)
# img,gt_img,lmk = TrainSet[1]
# plt.figure()
# plt.imshow(gt_img.permute(1,2,0).cpu().numpy())
# plt.scatter(lmk[:20,0]/250.*224., lmk[:20,1]/250.*224.)
# plt.show()

"""
from skimage import io
import scipy.io as sio
import numpy as np
from uv_map import load_uv_map
import trimesh
import BFM

texture_map = io.imread("./BFM/uv_texture_map.jpg")
print (texture_map.shape)

uv_idx, uv_weight = load_uv_map(img_size=256)
# print (uv_idx)
uv_map = torch.Tensor(texture_map[None,:,:,:]).permute(0,3,1,2)
print (uv_map.shape)
sampled_value = torch.stack([uv_map[:,:,uv_idx[:,1,0], uv_idx[:,0,0]],
                             uv_map[:,:,uv_idx[:,1,1], uv_idx[:,0,1]],
                             uv_map[:,:,uv_idx[:,1,2], uv_idx[:,0,2]],
                             uv_map[:,:,uv_idx[:,1,3], uv_idx[:,0,3]],
                            ], dim=-1)
uv_weight = torch.tensor(uv_weight[None, None, :, :])
offset = torch.sum(sampled_value*uv_weight, dim=-1).permute(0,2,1)/255.

print (offset.shape)

bfm_face_model = BFM.BFM()
# print (bfm_face_model.meantex.reshape(-1,3))

# print (bfm_face_model.point_buf[-1,-1])
mesh = trimesh.Trimesh(vertices=bfm_face_model.meanshape.reshape(-1,3), 
                       faces=bfm_face_model.tri.reshape(-1,3)-1, 
                       vertex_colors=offset[0])

mesh.show()
"""
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CACDDataset

import matplotlib.pyplot as plt

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
val_set = CACDDataset("./data/CACD2000_val.hdf5", val_transform, inv_normalize, "./data/CACD2000_val_bfm_recon.hdf5")
val_dataloader = DataLoader(val_set, batch_size=1, num_workers=1, shuffle=False)


for in_img, gt_img, lm, recon_img, recon_params in val_dataloader:
    fig = plt.figure(figsize=(300, 300))
    fig.add_subplot(1,2,1)
    plt.imshow(recon_img[0].permute(1,2,0))
    fig.add_subplot(1,2,2)
    plt.imshow(gt_img[0].permute(1,2,0))
    plt.show()