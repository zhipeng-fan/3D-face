import BFM
import os
import trimesh
import numpy as np
import utils
import torch

if not os.path.isfile("./BFM/BFM_model_front.mat"):
    utils.transferBFM09()

bfm_face_model = BFM.BFM()

# mesh = trimesh.Trimesh(vertices=bfm_face_model.meanshape.reshape(-1,3), 
#                        faces=bfm_face_model.tri.reshape(-1,3)-1, 
#                        vertex_colors=bfm_face_model.meantex.reshape(-1,3))

# mesh.show()



# def morph_face(bfm_model, id_param=None, ex_param=None, tex_param=None):
#     if id_param is None:
#         id_param = np.random.randn(80, 1)
#     if ex_param is None:
#         ex_param = np.random.randn(64, 1)
#     if tex_param is None:
#         tex_param = np.random.randn(80,1)

#     vertices = bfm_face_model.meanshape+np.dot(bfm_face_model.idBase, id_param)#+np.dot(bfm_face_model.exBase, ex_param)
#     colors = np.clip(bfm_face_model.meantex+np.matmul(bfm_face_model.texBase, tex_param), 0, 255)
#     mesh = trimesh.Trimesh(vertices=vertices.reshape(-1,3), 
#                             faces=bfm_face_model.tri.reshape(-1,3)-1, 
#                             vertex_colors=colors.reshape(-1,3))

#     mesh.show()


# morph_face(bfm_face_model)

# print (bfm_face_model.meanshape.shape)
# print (bfm_face_model.texBase.shape)
# print (bfm_face_model.exBase.shape)
# print (bfm_face_model.meantex.shape)
# print (bfm_face_model.idBase.shape)

face_model = BFM.BFM_torch(bfm_face_model)
shape = face_model.get_shape(torch.randn(1,80), torch.zeros(1,64))[0]
albedo = face_model.get_texture(torch.randn(1,80))[0]

print (albedo.shape)
print ((albedo.numpy().reshape(-1,3)))

mesh = trimesh.Trimesh(vertices=shape.numpy().reshape(-1,3), 
                       faces=bfm_face_model.tri.reshape(-1,3)-1, 
                       vertex_colors=np.clip(albedo.numpy().reshape(-1,3)/255.,0,1))
mesh.show()