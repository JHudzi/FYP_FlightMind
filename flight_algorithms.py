"""
Author: Josh Hudziak
Date: 30/04/21
Copyright:

Dependencies: Ubuntu 18.04, Olympe, Python, OpenCV, Numpy, TensorFlow GPU, Sklearn, Imutils, Pickle

Installation: Ubuntu 18.04 ONLT!
                OLYMPE:
                    cd $HOME
                    mkdir code/parrot-groundsdk
                    repo init -u https://github.com/Parrot-Developers/groundsdk-manifest.git
                    repo sync
                    (If some libraries do not install when activating olympe env use pip install to get them)

                CUDDN (for nvidia GPUs only):
                    Follow Ubuntu 18.04 instructions on
                    https://tensorflow.org/install/gpu

                Tensorflow:
                    pip install tensorflow *Remeber to be in the Olympe enviroment *version >2.0 will be GPU capable


"""
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
import math
from time import sleep
import cv2
import random
import time
import os
from threading import Thread

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

"""
write_navigation_output(pred)
    Parameters: pred - prediction string that comes from yuv_show_frame
    Writes the prediction output of the model to navigation.txt.
    Allows the controller to simultaniously predict frames and issue the UAV commands.
"""
def write_navigation_output(pred):

    f_handle = open("navigation.txt", "w")
    f_handle.write(str(pred))
    f_handle.close()

"""
read_navigation_output()
    Read the prediction output of the model in navigation.txt.
    Allows the controller to simultaniously predict frames and issue the UAV commands.
"""
def read_navigation_output():

    f_handle = open("navigation.txt", "r")
    state = f_handle.readline()
    f_handle.close()
    if state == "left":
        return "left"
    elif state == "right":
        return "right"
    else:
        return "centre"

"""
write_stop_output()
    Writes the to navigation.txt file: 'stop'.
    Allow the flight algorithm to know when it's time to stop controlling the drone.
"""
def write_stop_output():

    stop = "stop"
    f_handle = open("navigation.txt", "w")
    f_handle.write(str(stop))
    f_handle.close()


"""
def flight_time():
    *
    Time begins when the first navigation operation begins
    :return: totalTime
    *
    
    start_time = time.time()
    while read_navigation_output() != "stop":
        continue
    totalTime = time.time() - start_time
    return totalTime
"""