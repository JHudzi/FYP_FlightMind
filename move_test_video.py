"""
Author: Josh Hudziak
Date: 30/04/21
Copyright:      This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
                To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/.

Dependencies: Ubuntu 18.04, Olympe, Python, OpenCV, Numpy, TensorFlow GPU, Sklearn, Imutils, Pickle

Purpose:        Thisfile is used to test the automation process and the evaluation of the model when it is running on a video that is loaded. 
                The predictions made upon the video will then control the drone.

Installation: Ubuntu 18.04 ONLT!
                OLYMPE:
                    cd $HOME
                    mkdir code/parrot-groundsdk
                    repo init -u https://github.com/Parrot-Developers/groundsdk-manifest.git
                    repo sync
                    (If some libraries do not install when activating olympe env use pip install to get them)
                    cd ~/build.sh -p olympe-linux -A all final -j
                    (Again if there are missing libraries during the buils, pip install them)

                CUDDN (for nvidia GPUs only):
                    Follow Ubuntu 18.04 instructions on
                    https://tensorflow.org/install/gpu

                Tensorflow:
                    pip install tensorflow *Remeber to be in the Olympe enviroment *version >2.0 will be GPU capable
"""
"""
Import Libraries
"""
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.Piloting import moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged

from flight_algorithms import *
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
from collections import deque
import numpy as np
import pickle
import cv2
import os
import time
from time import sleep

"""
video_classification()
        Test a model on a video. Let the drone make actions based on model predictions that is being run on a video.
        Load in a model, the labels and a video directed by paths.
        Use the Queue function to make a pool of predictions.
        Initialize a writer. This will be set to the shape of incomming frames.
        Loop over each frame, make a copy of the original frame and then resize it to fit into our model
        Make prediction on this frame and add it to the queue. Average these predictions and get the largest probability.
        Pass this probability into the LabelBinarizer to sort it's class. use this label for a screen overlay on frames going out.
        Use the label to control the drone via moveby functions.
        Make sure the writer is available, if not create it and show it using OpenCV.
        Use 'q' to break the loop and land the drone
"""
def video_classification(drone):

    # load the trained model and label binarizer from disk
    #print("[INFO] loading model and label binarizer...")
    model = load_model("MODELS/Spear.h5")
    lb = pickle.loads(open("MODELS/lb.pickle", "rb").read())

    # initialize the predictions queue
    Q = deque(maxlen = 2)

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture("VIDEO/skip-right-1.MOV")
    writer = None
    (W, H) = (None, None)

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            
        # clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 250x250
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (250, 250)).astype("float32")

        # make predictions on the frame and then update the predictions
        # queue
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        
        # perform prediction averaging over the current history of
        # previous predictions
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        
        label = lb.classes_[i]
        # draw the activity on the output frame
        text = "Operation: {}".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 255, 0), 5)

        if label == "left":
            print("-----LEFT------")
            left()
        elif label == "right":
            print("-----RIGHT------")
            right()
        else:
            print("-----CENTRE------")
            centre()

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('Video_With_Classification', fourcc, 30,
                (W, H), True)
            
        # write the output frame to disk
        writer.write(output)
        # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
            
    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()

"""
left()
    Command the drone to rorate Left.
"""
def left():
    drone(moveBy(0, 0, 0, -1))

"""
right()
    Command the drone to rorate Right
"""
def right():
    drone(moveBy(0, 0, 0, 1))

"""
centre()
    Command the drone to stay centered. Add variable to dx to move forward.
    For testing purposes this is set to zero.
"""
def centre():
    drone(moveBy(0, 0, 0, 0))

"""
main()
    Start the drone using connection() and TakeOff(). Next start video_classification.
    A window will pop up of video playback that is labeled with predictions.
    Stop the drone with a Landing() and disconnection.
"""
def main(drone):
    """
    Taking Off
    """
    print("-----Taking Off-----")
    drone.connection()
    drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=5)).wait()

    """
    Video Processing
    """
    print("-----Video Processing-----")
    video_classification(drone)

    """
    Landing
    """
    write_stop_output()
    print("*****Landing*****")
    drone(Landing() >> FlyingStateChanged(state="landed", _timeout=5)).wait()
    
    #print("**** Total Flight Time: " + flight_time() + "*****")
    drone.disconnection()
"""
__main__
    Initialise the drones IP to either the Skycontroller or the drone itself. Pass this as drone to the main function.
"""
if __name__ == "__main__":

    SkyCntrl_IP = "192.168.53.1"
    Anafi_IP = "192.168.42.1"

    Anafi_URL = "http://{}/".format(Anafi_IP)

    with olympe.Drone(SkyCntrl_IP) as drone:
        main(drone)