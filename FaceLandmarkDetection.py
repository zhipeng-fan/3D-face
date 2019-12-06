import face_alignment
import glob
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
import random

import h5py

random.seed(3)
print ("Set random seed to 3!")

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

image_list = glob.glob("../CACD2000/*.jpg")
random.shuffle(image_list)
num_image = len(image_list)

num_train_image = int(num_image*0.70)
num_val_image = int(num_image*0.15)
num_test_image = num_image-num_train_image-num_val_image

def prepare_dataset(name, size, _image_list):
    with h5py.File("./Data/CACD2000_{}.hdf5".format(name), 'w') as f:
        image_dset = f.create_dataset("img", shape=(size, 250, 250 , 3), dtype='uint8')
        landmark_dset = f.create_dataset("lmk_2D", shape=(size, 68, 2), dtype='uint8')

        for idx, image_path in tqdm(enumerate(_image_list), desc="Processing Landmark...", total=size):
            image = io.imread(image_path)
            image_dset[idx] = image
            lmk = fa.get_landmarks(image)
            if lmk is None:
                print ("Error in {}".format(image_path))
            else:
                landmark_dset[idx] = lmk[0][:,:2]

print ("Prepare validation set:")
prepare_dataset("val", num_val_image, image_list[:num_val_image])
print ("Prepare test set:")
prepare_dataset("test", num_test_image, image_list[num_val_image:num_val_image+num_test_image])
print ("Prepare train set:")
prepare_dataset("train", num_train_image, image_list[-num_train_image:])

