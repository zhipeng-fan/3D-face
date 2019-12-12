import glob
import os

base_path = "./result_of/"
frame_rate = 2

img_list = sorted(glob.glob(os.path.join(base_path, "*.png")))

for img in img_list:
    os.rename(img, img[:20].replace(":","")+".png")


cmd = ["ffmpeg", "-r", str(frame_rate), '-i', os.path.join(base_path, "Epoch%02d.png"), "Face_Recon.mp4"]