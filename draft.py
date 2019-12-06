import BFM
import os
import trimesh
import numpy as np

if os.path.isfile("./BFM/BFM_model_front.mat"):
    BFM.transferBFM09()

bfm_face_model = BFM.BFM()

# mesh = trimesh.Trimesh(vertices=bfm_face_model.meanshape.reshape(-1,3), 
#                        faces=bfm_face_model.tri.reshape(-1,3)-1, 
#                        vertex_colors=bfm_face_model.meantex.reshape(-1,3))

# mesh.show()

type(bfm_face_model.idBase)

def morph_face(bfm_model, id_param=None, ex_param=None, tex_param=None):
    if id_param is None:
        id_param = np.random.randn(80, 1)
    if ex_param is None:
        ex_param = np.random.randn(64, 1)
    if tex_param is None:
        tex_param = np.random.randn(80,1)

    vertices = bfm_face_model.meanshape.T+np.dot(bfm_face_model.idBase, id_param)#+np.dot(bfm_face_model.exBase, ex_param)
    colors = np.clip(bfm_face_model.meantex.T+np.matmul(bfm_face_model.texBase, tex_param), 0, 255)
    mesh = trimesh.Trimesh(vertices=vertices.reshape(-1,3), 
                            faces=bfm_face_model.tri.reshape(-1,3)-1, 
                            vertex_colors=colors.reshape(-1,3))

    mesh.show()


morph_face(bfm_face_model)
print (bfm_face_model.texBase.shape)
print (bfm_face_model.idBase.shape)
print (bfm_face_model.exBase.shape)

