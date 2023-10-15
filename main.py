import customtkinter as ctk 
import sys,os
import tkinter as tk
from tkinter import filedialog
from face_recog import *
from PIL import ImageTk,Image
class App(ctk.CTk):
    def __init__(self):
        
        #set up
        super().__init__()
        ctk.set_appearance_mode('dark')
        container = ctk.CTkFrame(self)
        container.pack(side="top", fill="both", expand=True)
        self.title("Face reconstruct")
        self.frames = {}
        self.minsize(400,400)
        for F in (Selection, Next):
            page_name = F.__name__
            frame = F(parent=container, controll=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        #widget
        self.next("Selection")
        #run
        self.mainloop()
    
    def next(self,frame):
        frame_n = self.frames[frame]
        frame_n.tkraise()

class Next(ctk.CTkFrame):
    def __init__(self,parent,controll):
        super().__init__(master= parent)
        ctk.CTkLabel(text="",image=ImageTk.PhotoImage(Image.open("instruction.png")),master= self,).pack()
        ctk.CTkButton(text="Back",master= self,command= lambda: controll.next("Selection"),width=30,height=30,fg_color='transparent').place(relx = 0.97, rely = 0.95, anchor='ne')

class Selection(ctk.CTkFrame):
    #cover the window
    def __init__(self,parent,controll):
        super().__init__(master= parent)
        self.parent= parent
        ctk.CTkLabel(text= """After obtaining the image window, 
press C to process image/webcam""",master=self,).pack_configure(expand = True,pady = 10,padx = 20,anchor="center")
        ctk.CTkButton(text="Open image",master= self,command= self.open_image,height=40).pack(expand = True,pady = 10)
        ctk.CTkButton(text="Open webcam",master= self,command= self.open_webcam,height=40).pack(expand = True,pady = 10)
        ctk.CTkButton(text="Next ->",master= self,command= lambda: controll.next("Next"),width= 40,height = 40).pack(expand = True,pady = 20)


    def open_image(self):
        string = filedialog.askopenfile().name
        FaceRegAndPNG(string)
    def open_webcam(self):
        FaceRegAndPNG()

App()