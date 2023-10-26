import warnings
warnings.filterwarnings('ignore')
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, uic, QtCore, QtGui, QtWidgets
import os, sys
import sys
import cv2
import numpy as np
import datetime
from threading import Thread
from qt_thread_updater import get_updater
import dlib
import yaml
import shutil
from TDDFA import TDDFA
from utils.uv import uv_tex
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, load_and_resize
from show3d.show3d import show_3dface

class MainGUI(QtWidgets.QDialog):
    def __init__(self):
        super(MainGUI, self).__init__()
        self.ui = uic.loadUi("weights/Gui.ui", self)
        self.pushButton_OpenImg.clicked.connect(self.manual_img)
        self.pushButton_Reset.clicked.connect(self.reset)
        self.pushButton_OpenCam.clicked.connect(self.auto_camera)
        self.pushButton_Capture.clicked.connect(self.capture_image)
        self.pushButton_Confirm.clicked.connect(self.infer_image)
        self.pushButton_Next.clicked.connect(self.next_image)
        self.pushButton_Back.clicked.connect(self.back_image)
        self.pushButton_Save.clicked.connect(self.save_image)
        self.image_raw = None
        self.image_landmark = None
        self.image_uvtex = None
        self.image_pointcloud = None
        self.status_frame = None
        self.capture = None
        self.frame = None
        self.path_3dface = None
        self.check_saveimg =False

        self.face_detector = dlib.get_frontal_face_detector()
        
        cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
        self.tddfa = TDDFA(**cfg)

    def start(self):
        try:
            self.show()
            self.pushButton_Capture.setEnabled(False)
            self.pushButton_Confirm.setEnabled(False)
            self.pushButton_Next.setEnabled(False)
            self.pushButton_Back.setEnabled(False)
            self.pushButton_Save.setEnabled(False)
            
        except Exception as e:
            self.show_Text.addItems(["Error !"])
            sys.exit(1)
    
    def img_cv_2_qt(self, img_cv):
        img_cv = load_and_resize(img_cv, (960,720))
        height, width, channel = img_cv.shape
        bytes_per_line = channel * width
        img_qt = QtGui.QImage(img_cv, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        return QtGui.QPixmap.fromImage(img_qt)
    
    def reset(self):
        self.label_Image.clear()
        self.pushButton_Capture.setEnabled(False)
        self.pushButton_Confirm.setEnabled(False)
        self.pushButton_Next.setEnabled(False)
        self.pushButton_Back.setEnabled(False)
        self.pushButton_Save.setEnabled(False)
        self.pushButton_OpenImg.setEnabled(True)
        self.pushButton_OpenCam.setEnabled(True)
        self.image_raw = None
        self.image_landmark = None
        self.image_uvtex = None
        self.image_pointcloud = None
        self.status_frame = None
        self.path_3dface = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.frame = None
        self.show_Text.clear()
    
    def infer_image(self):
        image = self.image_raw
        self.pushButton_Capture.setEnabled(False)
        self.pushButton_OpenImg.setEnabled(False)
        self.pushButton_OpenCam.setEnabled(False)
        
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(img_gray, 0)
        bboxes =[]
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            bboxes.append([x1, y1,x2 ,y2])
        
        if len(bboxes) == 0:
            self.show_Text.addItems(["There are no faces !"])
        else:
            path_save =os.path.join(".cache")
            os.makedirs(path_save, exist_ok = True)

            param_lst, roi_box_lst = self.tddfa(image, bboxes)
            ver_landmark = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            path_imglandmark = path_save + '/data_landmarks.jpg'
            draw_landmarks(image, ver_landmark, show_flag=False, dense_flag=False, wfp=path_imglandmark)

            ver_3d = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            
            wfp_obj = path_save + '/data3d.obj'
            ser_to_obj(image, ver_3d, self.tddfa.tri, height=image.shape[0], wfp=wfp_obj)

            path_textureface = path_save + '/data_textureface.jpg'
            uv_tex(image, ver_3d, self.tddfa.tri, show_flag=False, wfp=path_textureface)
            self.status_frame = 2
            self.image_landmark = cv2.imread(path_imglandmark)
            self.image_uvtex = cv2.imread(path_textureface)
            get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(self.image_landmark))
            self.pushButton_Next.setEnabled(True)
            self.pushButton_Back.setEnabled(True)
            self.pushButton_Save.setEnabled(True)
            self.check_saveimg =False
            self.path_3dface = wfp_obj
            self.show_Text.addItems(["3D model created successfully!"])
            self.show_Text.addItems(["Face's landmarks image"])
    
    def show_3dface(self):
        if self.path_3dface is not None:
            show_3dface(self.path_3dface)
            self.show_Text.addItems(["Choose Back/Reset"])        

    def next_image(self):
        self.status_frame +=1
        if self.status_frame ==4:
            self.show_Text.addItems(["3D model; Press ESC to close window!"])
            self.show_3dface()
            self.pushButton_Next.setEnabled(False)
            self.pushButton_Back.setEnabled(True)
            self.label_Image.clear()
        elif self.status_frame ==3:
            self.pushButton_Back.setEnabled(True)
            self.show_Text.addItems(["Face's texture image"])
            get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(self.image_uvtex))
        elif self.status_frame ==2:
            self.pushButton_Back.setEnabled(True)
            self.show_Text.addItems(["Face's landmarks image"])
            get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(self.image_landmark))
        elif self.status_frame ==1:
            self.pushButton_Back.setEnabled(True)
            self.show_Text.addItems(["Input image"])
            get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(self.image_raw))
    
    def save_image(self):
        try:
            if self.check_saveimg ==False:
                self.check_saveimg =True
                name_save= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path_save =os.path.join("output", str(name_save))
                os.makedirs(path_save, exist_ok = True)
                path_imglandmark = path_save + '/data_landmarks.jpg'
                cv2.imwrite(path_imglandmark, self.image_landmark)
                wfp_obj = path_save + '/data3d.obj'
                shutil.copy(self.path_3dface , wfp_obj)
                path_textureface = path_save + '/data_textureface.jpg'
                cv2.imwrite(path_textureface, self.image_uvtex)
                self.show_Text.addItems(["Image saved successfully !"])
            else:
                self.show_Text.addItems(["Image saved!"])
        except:
            self.show_Text.addItems(["Error ! Save unsuccessfully"])

         
    def back_image(self):
        self.status_frame -=1
        if self.status_frame ==3:
            self.pushButton_Next.setEnabled(True)
            self.show_Text.addItems(["Face's texture image"])
            get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(self.image_uvtex))
        elif self.status_frame ==2:
            self.pushButton_Next.setEnabled(True)
            self.show_Text.addItems(["Face's landmarks image"])
            get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(self.image_landmark))
        elif self.status_frame ==1:
            self.show_Text.addItems(["Input image"])
            self.pushButton_Back.setEnabled(False)
            self.pushButton_Next.setEnabled(True)
            get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(self.image_raw))

    def auto_camera(self):
        self.label_Image.clear()
        self.pushButton_Capture.setEnabled(True)
        self.pushButton_OpenImg.setEnabled(False)
        self.capture  = cv2.VideoCapture(0)
        ret, self.frame = self.capture.read()
        if self.capture is None or not self.capture.isOpened():
            self.show_Text.addItems(["Camera không hợp lệ !"])
        else:
            self.show_Text.addItems(["Succesfully Open Camera !", "Please choose Capture"])
            self.timer = QTimer()
            self.timer.timeout.connect(self.display_video_stream)
            self.timer.start(0)

    def display_video_stream(self):
        if self.capture is not None:
            _, self.frame = self.capture.read()
            
        if self.frame is not None:
            image_show= self.frame.copy()
            get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(image_show))
    
    def capture_image(self):
        if self.frame is not None:
            self.show_Text.addItems(["Succesfully Capture image !", "Please choose Confirm"])
            self.image_raw = self.frame.copy()
            if self.capture is not None:
                self.capture.release()
                self.capture = None
                self.frame = None
            self.status_frame = 1
            get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(self.image_raw))
            self.pushButton_Confirm.setEnabled(True)
            self.pushButton_OpenCam.setEnabled(False)
            
        else:
            self.show_Text.addItems(["Capture unsuccessfully!"])
            
    def manual_img(self):
        self.label_Image.clear()
        self.pushButton_Capture.setEnabled(False)
        self.pushButton_OpenCam.setEnabled(False)
        try:
            options = QtWidgets.QFileDialog.Options()
            img_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
            if img_file:
                image= cv2.imread(img_file)
                self.image_raw = image.copy()
                self.status_frame = 1
                get_updater().call_latest(self.label_Image.setPixmap, self.img_cv_2_qt(self.image_raw))
            
                self.pushButton_Confirm.setEnabled(True)
                self.show_Text.addItems(["Succesfully Open Image !","Please choose Confirm"])
            else:
                self.show_Text.addItems(["Invalid image \n Error !"])
        except:
            self.show_Text.addItems(["Invalid image \n Error !"])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainGUI()
    main.setWindowTitle('3D Face Reconstruction')
    main.setWindowIcon(QtGui.QIcon("weights/face.png"))
    main.start()
    sys.exit(app.exec_())
