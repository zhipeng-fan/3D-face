import soft_renderer as sr
from torch import nn
import torch
import torch.nn.functional as f
import math

from BFM import BFM_torch

class BFMFaceLoss(nn.Module):
    """Decode from the learned parameters to the 3D face model"""
    def __init__(self, renderer, lmk_loss_w, device):
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
        face_id = self.BFM_model.tri-1
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

        Y0 = self.a0*torch.ones(bs, num_vertex, 1, device=self.device)
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



    def forward(self, params, gt_img, gt_lmk):
        bs = params.shape[0]
        id_coef, ex_coef, tex_coef, angles, gamma, tranlation, scale = self.split(params)
        face_shape = self.BFM_model.get_shape(id_coef, ex_coef)
        face_albedo = self.BFM_model.get_texture(tex_coef)

        face_shape = face_shape.reshape(bs, -1, 3)
        face_shape[:,:,-1] *= -1
        # Recenter the face mesh
        face_shape = face_shape - torch.mean(face_shape, dim=1, keepdim=True)        
        face_albedo = face_albedo.reshape(bs, -1, 3)/255.

        # face model scaling, rotation and translation
        rotation_matrix = self.BFM_model.compute_rotation_matrix(angles)
        face_shape = torch.bmm(face_shape, rotation_matrix)
        # Compute the normal
        normal = self.compute_norm(face_shape)
        face_shape = scale[:,:,None]*face_shape
        face_shape = face_shape+tranlation[:,None,:]

        face_albedo = self.lighting(normal, face_albedo, gamma)

        tri = self.BFM_model.tri.reshape(-1,3)-1
        face_triangles = tri[None,:,:].expand(bs,-1,-1)

        recon_mesh, recon_img = self.renderer(face_shape,
                                              face_triangles,
                                              face_albedo,
                                              texture_type="vertex")
        recon_lmk = recon_mesh.vertices[:, self.BFM_model.keypoints.long(), :]

        # Compute loss
        # remove the alpha channel
        img_loss = self.mse_criterion(recon_img[:,:3,:,:], gt_img)
        gt_lmk[:,:,1] = 250.-gt_lmk[:,:,1].float()
        lmk_loss = self.sl1_criterion(recon_lmk[:,:,:2], (gt_lmk.float()-125.)/125.)
        all_loss = img_loss + self.lmk_loss_w*lmk_loss
        return all_loss, img_loss, lmk_loss, recon_img