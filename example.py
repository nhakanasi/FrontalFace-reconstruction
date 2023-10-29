import os, sys
import cv2
import numpy as np
import dlib
import yaml
import datetime
from TDDFA import TDDFA
from utils.uv import uv_tex
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, load_and_resize


cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(**cfg)
detector = dlib.get_frontal_face_detector()

image = cv2.imread("images/img1.jpg")

img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
faces = detector(img_gray, 0)

bboxes =[]
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    bboxes.append([x1, y1,x2 ,y2])
    
n = len(bboxes)
if n == 0:
    print(f'No face detected, exit')
    sys.exit(-1)

name_save= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path_save =os.path.join("output", str(name_save))
os.makedirs(path_save, exist_ok = True)

param_lst, roi_box_lst = tddfa(image, bboxes)
ver_landmark = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
path_imglandmark = path_save + '/data_landmarks.jpg'
draw_landmarks(image, ver_landmark, show_flag=False, dense_flag=False, wfp=path_imglandmark)

ver_3d = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

wfp_obj = path_save + '/data3d.obj'
ser_to_obj(image, ver_3d, tddfa.tri, height=image.shape[0], wfp=wfp_obj)

path_textureface = path_save + '/data_textureface.jpg'
uv_tex(image, ver_3d, tddfa.tri, show_flag=False, wfp=path_textureface)
