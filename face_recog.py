import numpy as np

import os
import cv2
import dlib
from imutils import face_utils

import tkinter as tk
from tkinter import filedialog
from threading import Thread
from face_texture import FaceTexture

# for import, maybe develop into app later
root = tk.Tk()
root.withdraw()
final = None
class FacialCharactors:
	def __init__(self,frame,index,pos):
		self.frame = frame
		self.i = index
		self.x1,self.x2,self.y1,self.y2 = pos 
		self.run()

	def run(self):
		predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
		image = self.frame[self.y1+2:self.y2-2,self.x1+2:self.x2-2]
		image_name = "image/face_"+ str(self.i) + ".png"
		# perform turning image gray
		gray_crop = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# landmakrs
		shape = predictor(gray_crop, dlib.rectangle(left=0, top=0, right=self.x2-self.x1, bottom=self.y2-self.y1))
		shape = face_utils.shape_to_np(shape)
		pts = create_pts(shape=shape)

		# save an rename files
		check_and_rename(image_name,image)
		self.image_path = final

		# print a picture with face landmarks to validate
		for (x,y) in shape:
			cv2.circle(image,(x,y),2,(0,0,255),-1)
		# write landmarks file
		with open("image/pts.pts","w") as f:
			f.write(pts)

	def return_files(self):
		return ("image/pts.pts", self.image_path)

# for saving face landmarks' coordinates
def create_pts(shape):
	image_pts = f"version: 1\nn_points:  68\n{{\n"
	for (x, y) in shape:
		image_pts += f"{x:.6f} {y:.6f}\n"
	image_pts += "}"
	return image_pts

# rename purpose
def check_and_rename(file_name, file, add=0):
		global final
		final = file_name
		if add != 0: 
			split = file_name.split(".")
			if add == 1:
				first = split[0] + "_" + str(add)
			else:
				first = "_".join(split[0].split("_")[:2])+ "_" +str(add)
			final = ".".join([first,split[1]])

		if not os.path.isfile(final):
			cv2.imwrite(final,file)
			print(final)
		else:
			add +=1
			check_and_rename(final,file,add)

# threading for better performance
def obj_and_png(args):
	frame,i,pos = args
	facial = FacialCharactors(frame,i,pos)
	print(facial.return_files())
	FaceTexture(facial.return_files())

class FaceReg2PNG:
	# clean up process depend on format
	def clean_up(self,format):
		if format == "webcam":
			self.vc.release()
			cv2.destroyAllWindows()
		elif format == "images":
			cv2.destroyAllWindows()

	# choose media
	def media(self,format):
		if format == "webcam":
			return cv2.VideoCapture(0)
		elif format == "images":
			name = filedialog.askopenfilename() 
			if name == "":
				return None
			else: 
				return cv2.imread(name)

	# run
	def main(self,format):
		self.vc = self.media(format)
		if self.vc == None:
			print("No media detected")
			return 
		
		# prepare
		prototxtPath = "data/deploy.prototxt"
		caffemodelPath = "data/res10_300x300_ssd_iter_140000.caffemodel"
		net = cv2.dnn.readNetFromCaffe(prototxt=prototxtPath, caffeModel=caffemodelPath)

		run = False

		while True:

			# grab the frame from video stream and resize
			if format == "webcam":
				ret, frame = self.vc.read()
				if ret==False:
					break
			else:
				frame = self.vc
			frame = cv2.flip(frame,1)
			frame = cv2.resize(frame,(frame.shape[1], frame.shape[0]))

			# format
			cv2.putText(frame, f"{format}", (0, 30),cv2.LINE_AA, 0.45, (0, 255, 0), 2)

			# grab the frame dimensions and convert it to a blob
			(h, w) = frame.shape[:2]
			inputBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1, (300, 300), (104, 177, 123))
			
			# process with AI
			net.setInput(inputBlob)
			detections = net.forward()

			for i in range(0, detections.shape[2]):
				prediction_score = detections[0, 0, i, 2]
				if prediction_score < 0.5:
					continue

				# compute the (x, y)-coordinates of the bounding box for the
				# object and for better landmark detection
				box = detections[0, 0, i, 3:7] * np.array([w*0.8, h*0.7, w*1.1, h*1.05])
				(x1, y1, x2, y2) = box.astype("int")
				y = y1 - 10 if y1 - 10 > 10 else y1 + 10

				# Make the prediction and transfom it to numpy array
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

				# show faces with confidence score
				prediction_score_str = "{:.2f}%".format(prediction_score * 100)
				label = "{})".format(prediction_score_str)

				cv2.putText(frame, label, (x1, y),
							cv2.LINE_AA, 0.45, (0, 255, 0), 2)
				
				pos = [x1,x2,y1,y2]
				if cv2.waitKey(1) & 0xFF == ord('c') and run == False:
					args = [frame,i,pos]
					export = Thread(target=obj_and_png,args=[args])
					export.start()
					export.join()

			# show video
			cv2.imshow('frame',frame)

			# quit using Q
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		# clean up
		self.clean_up(format)
	
	# choose mode
	def __init__(self, format):
		self.main(format)

if __name__ == "__main__":
	options = ["images","webcam"]
	print("""Choose format:
       - Images
       - Webcam""")
	print("Media", end= ": ")
	choice = input()
	if choice.lower() in options:
		FaceReg2PNG(choice)
	else:
		print("Not supported")