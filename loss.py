import soft_renderer as sr
from torch import nn
import torch
import torch.nn.functional as f
import math
import numpy as np
from tqdm import tqdm
import os

from BFM import BFM_torch
from facenet_pytorch import InceptionResnetV1
from uv_map import load_uv_map

class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, device, average=True):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        self.device = device

        if os.path.isfile("./BFM/BFM_laplacian.pth"):
            mat = torch.load("./BFM/BFM_laplacian.pth")
            i = mat['i']
            v = mat['v']
            # import pdb; pdb.set_trace()
            self.laplacian = torch.sparse.FloatTensor(i.transpose(1,0),v,torch.Size([self.nv, self.nv])).to(self.device)
            self.laplacian.requires_grad = False

            laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

            laplacian[faces[:, 0], faces[:, 1]] = -1
            laplacian[faces[:, 1], faces[:, 0]] = -1
            laplacian[faces[:, 1], faces[:, 2]] = -1
            laplacian[faces[:, 2], faces[:, 1]] = -1
            laplacian[faces[:, 2], faces[:, 0]] = -1
            laplacian[faces[:, 0], faces[:, 2]] = -1

            r, c = np.diag_indices(laplacian.shape[0])
            laplacian[r, c] = -laplacian.sum(1)

            for i in range(self.nv):
                laplacian[i, :] /= laplacian[i, i]
            assert (self.laplacian.to_dense().cpu().numpy()-laplacian<1e-4).all()
            
        else:
            laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

            laplacian[faces[:, 0], faces[:, 1]] = -1
            laplacian[faces[:, 1], faces[:, 0]] = -1
            laplacian[faces[:, 1], faces[:, 2]] = -1
            laplacian[faces[:, 2], faces[:, 1]] = -1
            laplacian[faces[:, 2], faces[:, 0]] = -1
            laplacian[faces[:, 0], faces[:, 2]] = -1

            r, c = np.diag_indices(laplacian.shape[0])
            laplacian[r, c] = -laplacian.sum(1)

            for i in range(self.nv):
                laplacian[i, :] /= laplacian[i, i]

            x, y = [], []
            value = []
            for i in tqdm(range(self.nv)):
                for j in range(self.nv):
                    if laplacian[i,j] != 0:
                        y.append(i)
                        x.append(j)
                        value.append(laplacian[i,j])
            
            i = torch.stack([torch.LongTensor(y), torch.LongTensor(x)], dim=1)
            v = torch.FloatTensor(value)
            torch.save({"i": i, "v":v, "size":self.nv}, "./BFM/BFM_laplacian.pth")
            self.laplacian = torch.sparse.FloatTensor(i,v,torch.Size([self.nv, self.nv])).to(self.device)
            assert all(self.laplacian.to_dense().cpu().numpy()-laplacian<1e-4)
            self.laplacian.requires_grad=False


    def forward(self, x):
        batch_size = x.size(0)
        result = torch.zeros_like(x, device=self.device)
        for i in range(batch_size):
            result[i] = torch.sparse.mm(self.laplacian, x[i])

        dims = tuple(range(result.ndimension())[1:])
        result = result.pow(2).sum(dims)
        if self.average:
            return result.sum() / batch_size
        else:
            return result

class BFMFaceLoss(nn.Module):
    """Decode from the learned parameters to the 3D face model"""
    def __init__(self, renderer, lmk_loss_w, recog_loss_w, device):
        super(BFMFaceLoss, self).__init__()
        self.BFM_model = BFM_torch().to(device)
        self.renderer = renderer

        self.mse_criterion = nn.MSELoss()
        self.sl1_criterion = nn.SmoothL1Loss()
        self.lmk_loss_w = lmk_loss_w
        self.device = device

        self.a0 = torch.tensor(math.pi).to(self.device)
        self.a1 = torch.tensor(2*math.pi/math.sqrt(3.0)).to(self.device)
        self.a2 = torch.tensor(2*math.pi/math.sqrt(8.0)).to(self.device)
        self.c0 = torch.tensor(1/math.sqrt(4*math.pi)).to(self.device)
        self.c1 = torch.tensor(math.sqrt(3.0)/math.sqrt(4*math.pi)).to(self.device)
        self.c2 = torch.tensor(3*math.sqrt(5.0)/math.sqrt(12*math.pi)).to(self.device)

        self.reverse_z = torch.eye(3).to(self.device)[None,:,:]
        self.face_net = InceptionResnetV1(pretrained='vggface2').eval()
        for param in self.face_net.parameters():
            param.requires_grad=False
        self.face_net.to(device)
        self.recog_loss_w = recog_loss_w    

    def split(self, params):
        id_coef = params[:,:80]
        ex_coef = params[:,80:144]
        tex_coef = params[:,144:224]
        angles = params[:,224:227]
        gamma  = params[:,227:254]
        translation = params[:,254:257]
        scale = params[:,257:]
        return id_coef, ex_coef, tex_coef, angles, gamma, translation, scale

    def compute_norm(self, vertices):
        """
        Compute the norm of the vertices
        Input:
            vertices[bs, 35709, 3]
        """
        bs = vertices.shape[0]
        face_id = torch.flip(self.BFM_model.tri.reshape(-1,3)-1, dims=[1])
        point_id = self.BFM_model.point_buf-1
        # [bs, 70789, 3]
        face_id = face_id[None,:,:].expand(bs,-1,-1)
        # [bs, 35709, 8]
        point_id = point_id[None,:,:].expand(bs,-1,-1)
        # [bs, 70789, 3] Gather the vertex location
        v1 = torch.gather(vertices, dim=1,index=face_id[:,:,:1].expand(-1,-1,3).long())
        v2 = torch.gather(vertices, dim=1,index=face_id[:,:,1:2].expand(-1,-1,3).long())
        v3 = torch.gather(vertices, dim=1,index=face_id[:,:,2:].expand(-1,-1,3).long())
        # Compute the edge
        e1 = v1-v2
        e2 = v2-v3
        # Normal [bs, 70789, 3]
        norm = torch.cross(e1, e2)
        # Normal appended with zero vector [bs, 70790, 3]
        norm = torch.cat([norm, torch.zeros(bs, 1, 3).to(self.device)], dim=1)
        # [bs, 35709*8, 3]
        point_id = point_id.reshape(bs,-1)[:,:,None].expand(-1,-1,3)
        # [bs, 35709*8, 3]
        v_norm = torch.gather(norm, dim=1, index=point_id.long())
        v_norm = v_norm.reshape(bs, 35709, 8, 3)
        # [bs, 35709, 3]
        v_norm = f.normalize(torch.sum(v_norm, dim=2), dim=-1)
        return v_norm


    def lighting(self, norm, albedo, gamma):
        """
        Add lighting to the albedo surface
        gamma: [bs, 27]
        norm: [bs, num_vertex, 3]
        albedo: [bs, num_vertex, 3]
        """
        assert norm.shape[0] == albedo.shape[0]
        assert norm.shape[0] == gamma.shape[0]
        bs = gamma.shape[0]
        num_vertex = norm.shape[1]

        init_light = torch.zeros(9).to(self.device)
        init_light[0] = 0.8
        gamma = gamma.reshape(bs,3,9)+init_light

        Y0 = self.a0*self.c0*torch.ones(bs, num_vertex, 1, device=self.device)
        Y1 = -self.a1*self.c1*norm[:,:,1:2]
        Y2 = self.a1*self.c1*norm[:,:,2:3]
        Y3 = -self.a1*self.c1*norm[:,:,0:1]
        Y4 = self.a2*self.c2*norm[:,:,0:1]*norm[:,:,1:2]
        Y5 = -self.a2*self.c2*norm[:,:,1:2]*norm[:,:,2:3]
        Y6 = self.a2*self.c2*0.5/math.sqrt(3.0)*(3*norm[:,:,2:3]**2-1)
        Y7 = -self.a2*self.c2*norm[:,:,0:1]*norm[:,:,2:3]
        Y8 = self.a2*self.c2*0.5*(norm[:,:,0:1]**2-norm[:,:,1:2]**2)
        # [bs, num_vertice, 9]
        Y = torch.cat([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],dim=2)

        light_color = torch.bmm(Y, gamma.permute(0,2,1))
        vertex_color = light_color*albedo
        return vertex_color

    def reconst_img(self, params):
        bs = params.shape[0]
        id_coef, ex_coef, tex_coef, angles, gamma, tranlation, scale = self.split(params)

        face_shape = self.BFM_model.get_shape(id_coef, ex_coef)
        face_albedo = self.BFM_model.get_texture(tex_coef) 
        face_shape[:,:,-1] *= -1
        # Recenter the face mesh
        face_albedo = face_albedo.reshape(bs, -1, 3)/255.

        # face model scaling, rotation and translation
        rotation_matrix = self.BFM_model.compute_rotation_matrix(angles)
        face_shape = torch.bmm(face_shape, rotation_matrix)
        # Compute the normal
        normal = self.compute_norm(face_shape)
        
        face_shape = (1+scale[:,:,None])*face_shape
        face_shape = face_shape+tranlation[:,None,:]

        face_albedo = self.lighting(normal, face_albedo, gamma)

        tri = torch.flip(self.BFM_model.tri.reshape(-1,3)-1, dims=[-1])
        face_triangles = tri[None,:,:].expand(bs,-1,-1)

        recon_mesh, recon_img = self.renderer(face_shape,
                                              face_triangles,
                                              face_albedo,
                                              texture_type="vertex")
        return recon_img

    def forward(self, params, gt_img, gt_lmk):
        bs = params.shape[0]
        id_coef, ex_coef, tex_coef, angles, gamma, tranlation, scale = self.split(params)

        face_shape = self.BFM_model.get_shape(id_coef, ex_coef)
        face_albedo = self.BFM_model.get_texture(tex_coef) 
        face_shape[:,:,-1] *= -1
        # Recenter the face mesh
        face_albedo = face_albedo.reshape(bs, -1, 3)/255.

        # face model scaling, rotation and translation
        rotation_matrix = self.BFM_model.compute_rotation_matrix(angles)
        face_shape = torch.bmm(face_shape, rotation_matrix)
        # Compute the normal
        normal = self.compute_norm(face_shape)
        
        face_shape = (1+scale[:,:,None])*face_shape
        face_shape = face_shape+tranlation[:,None,:]

        face_albedo = self.lighting(normal, face_albedo, gamma)

        tri = torch.flip(self.BFM_model.tri.reshape(-1,3)-1, dims=[-1])
        face_triangles = tri[None,:,:].expand(bs,-1,-1)

        recon_mesh, recon_img = self.renderer(face_shape,
                                              face_triangles,
                                              face_albedo,
                                              texture_type="vertex")
        recon_lmk = recon_mesh.vertices[:, self.BFM_model.keypoints.long(), :]

        # Compute loss
        # remove the alpha channel
        mask = (recon_img[:,-1:,:,:].detach() > 0).float()
        # Image loss
        img_loss = self.mse_criterion(recon_img[:,:3,:,:], gt_img*mask)
        # Landmark loss
        recon_lmk_2D_rev = (recon_lmk[:,:,:2]+1)*250./2.
        recon_lmk_2D = (recon_lmk[:,:,:2]+1)*250./2.
        recon_lmk_2D[:,:,1] = 250.-recon_lmk_2D_rev[:,:,1] 
        lmk_loss = self.sl1_criterion(recon_lmk_2D, gt_lmk.float())
        # face recog loss
        recon_feature = self.face_net(recon_img[:,:3,:,:])
        gt_feature = self.face_net(gt_img*mask)
        recog_loss = self.mse_criterion(recon_feature, gt_feature)
        all_loss = img_loss + self.lmk_loss_w*lmk_loss + self.recog_loss_w*recog_loss
        return all_loss, img_loss, lmk_loss, recog_loss, recon_img


class BFMFaceLossUVMap(BFMFaceLoss):
    def __init__(self, small_weight, sym_weight, img_size, *args, **kwargs):
        super(BFMFaceLossUVMap, self).__init__(*args, **kwargs)
        self.small_offset_weight = small_weight
        self.sym_weight = sym_weight
        uv_idx, uv_weight = load_uv_map(img_size=img_size)
        self.register_buffer('uv_idx', torch.Tensor(uv_idx).long())
        self.register_buffer('uv_weight', torch.Tensor(uv_weight))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = LaplacianLoss(self.BFM_model.meanshape.reshape(-1,3).cpu(), 
                                            torch.flip(self.BFM_model.tri.reshape(-1,3)-1, dims=[1]).cpu(),
                                            self.device)
        # self.flatten_loss = sr.FlattenLoss(torch.flip(self.BFM_model.tri.reshape(-1,3)-1, dims=[1]).cpu())

    def bilinear_sampling(self, mesh, offset_map):
        """
        Bilinearly sample the offset value from the predicted texture map and the shape map
        """
        sampled_value = torch.stack([offset_map[:,:,self.uv_idx[:,1,0], self.uv_idx[:,0,0]],
                                     offset_map[:,:,self.uv_idx[:,1,1], self.uv_idx[:,0,1]],
                                     offset_map[:,:,self.uv_idx[:,1,2], self.uv_idx[:,0,2]],
                                     offset_map[:,:,self.uv_idx[:,1,3], self.uv_idx[:,0,3]],
                                    ], dim=-1)
        uv_weight = self.uv_weight[None, None, :, :]
        offset = torch.sum(sampled_value*uv_weight, dim=-1).permute(0,2,1)
        return offset

    def forward(self, params, shape_offset_map, color_offset_map, gt_img, gt_lmk):
        bs = params.shape[0]
        id_coef, ex_coef, tex_coef, angles, gamma, tranlation, scale = self.split(params)

        face_shape = self.BFM_model.get_shape(id_coef, ex_coef)
        face_shape[:,:,-1] *= -1
        face_albedo = self.BFM_model.get_texture(tex_coef) 
        face_albedo = face_albedo.reshape(bs, -1, 3)/255.

        shape_offset = self.bilinear_sampling(face_shape, shape_offset_map)
        albedo_offset = self.bilinear_sampling(face_albedo, color_offset_map)
        face_shape = face_shape+shape_offset
        face_albedo = face_albedo+albedo_offset

        # laplacian loss and flatten loss for the smoothness of the mesh structure
        laplacian_loss = self.laplacian_loss(face_shape).mean()
        # flatten_loss = self.flatten_loss(face_shape).mean()

        # face model scaling, rotation and translation
        rotation_matrix = self.BFM_model.compute_rotation_matrix(angles)
        face_shape = torch.bmm(face_shape, rotation_matrix)
        # Compute the normal
        normal = self.compute_norm(face_shape)
        
        face_shape = (1+scale[:,:,None])*face_shape
        face_shape = face_shape+tranlation[:,None,:]

        face_albedo = self.lighting(normal, face_albedo, gamma)

        tri = torch.flip(self.BFM_model.tri.reshape(-1,3)-1, dims=[-1])
        face_triangles = tri[None,:,:].expand(bs,-1,-1)

        recon_mesh, recon_img = self.renderer(face_shape,
                                              face_triangles,
                                              face_albedo,
                                              texture_type="vertex")
        recon_lmk = recon_mesh.vertices[:, self.BFM_model.keypoints.long(), :]

        # Compute loss
        # remove the alpha channel
        mask = (recon_img[:,-1:,:,:].detach() > 0).float()
        # Image loss
        img_loss = self.mse_criterion(recon_img[:,:3,:,:], gt_img*mask)
        
        # Landmark loss
        recon_lmk_2D_rev = (recon_lmk[:,:,:2]+1)*250./2.
        recon_lmk_2D = (recon_lmk[:,:,:2]+1)*250./2.
        recon_lmk_2D[:,:,1] = 250.-recon_lmk_2D_rev[:,:,1] 
        lmk_loss = self.sl1_criterion(recon_lmk_2D, gt_lmk.float())
        
        # face recog loss
        recon_feature = self.face_net(recon_img[:,:3,:,:])
        gt_feature = self.face_net(gt_img*mask)
        recog_loss = self.mse_criterion(recon_feature, gt_feature)
        
        # Small offset loss
        shape_offset_l2 = torch.norm(shape_offset)
        albedo_offset_l2 = torch.norm(albedo_offset)
        # symmetric offset loss
        shape_offset_sym = self.mse_criterion(shape_offset_map, torch.flip(shape_offset_map, dims=[2]))
        albedo_offset_sym = self.mse_criterion(color_offset_map, torch.flip(color_offset_map,dims=[2]))

        # print (shape_offset_l2, albedo_offset_l2, shape_offset_sym, albedo_offset_sym)

        all_loss = 20*img_loss + self.lmk_loss_w*lmk_loss + self.recog_loss_w*recog_loss +\
                    self.small_offset_weight*(shape_offset_l2+albedo_offset_l2) +\
                    self.sym_weight*(shape_offset_sym+albedo_offset_sym) +\
                    0.08 * laplacian_loss 
                    # 0.0003 * flatten_loss

        return all_loss, img_loss, lmk_loss, recog_loss, recon_img 
