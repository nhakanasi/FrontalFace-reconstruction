# FrontalFace-reconstruction
Using numerous python library,  this project hope to create a program that allow user to reconstruct their frontal lobe in a 3d file to import into games and 3d animation.

An exe file is comming soon!

## Usage
To run this project, you need to have the following libraries: 
* Opencv: https://pypi.org/project/opencv-python/
* Dlib: https://pypi.org/project/dlib/
* Imutils: https://pypi.org/project/imutils/
* Eos: https://github.com/patrikhuber/eos

## Notes
- In this projects, there are a few files you can change to better suit your purpose or accuracy such as the data folder and share folder. I advise that if you want to change the share folder, you should get help from the creator of EOS - patrikhuber.
- Right now the output would bee a .obj file and a texture.png file in a output folder when you run, you would need to apply the texture in a program like blender to see the final result
- Running the main file will result in a UI for better usage
- It is missing a .dat file in data folder, if you are using it now, please download https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat and put it in the data folder
