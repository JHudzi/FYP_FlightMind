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
"""
Import Libraries
"""
import csv
import cv2
import math
import os
import queue
import shlex
import subprocess
import tempfile
import threading
import traceback
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
from collections import deque
import numpy as np
import pickle
from flight_algorithms import write_navigation_output, write_stop_output, read_navigation_output

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.Piloting import moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged


"""
Global Variables-
    DRONE_IP: connection ip for the drone. Using Sky controller IP for range and control.
    model:    Call the ResNet model using Tensorflow library functions. This is used for predictions
    lb:       Pickle file stores all label classes and is used to associate the prediction with an understandable lable.
"""
olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

DRONE_IP = "192.168.53.1"

print("***** [INFO] loading model and label binarizer *****")
model = load_model("MODELS/New_Data_Deer.h5")
lb = pickle.loads(open("MODELS/lb.pickle", "rb").read())

"""
Creating a class for a drone object creation provides threading that makes asynchronous actions much easier and stops video feedback stopping.
"""
class Navigation(threading.Thread):

    """
    Construct a drone object with a frame queue to implement threading on video feedback
    """
    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.drone = olympe.Drone(DRONE_IP)
        self.tempd = tempfile.mkdtemp(prefix="olympe_streaming_test_")
        print("Olympe streaming example output dir: {}".format(self.tempd))
        self.frame_queue = queue.Queue()
        self.flush_queue_lock = threading.Lock()
        super().__init__()
        super().start()

    """
    start()
        Initialises the drone object connecting it to Olympe controller(laptop) via ip address. Callbacks for image processing are initiated which
        provide video feedback with prediction labels.
    """
    def start(self):
        # Connect the the drone
        self.drone.connect()

        # Setup your callback functions to do some live video processing
        self.drone.set_streaming_callbacks(
            raw_cb=self.yuv_frame_cb,
            start_cb=self.start_cb,
            end_cb=self.end_cb,
            flush_raw_cb=self.flush_cb,
        )
        # Start video streaming
        self.drone.start_video_streaming()

    """
    stop()
        Once the drone has finished call write to stop function from flight_algorithms, next, The drone will initiate landing,
        then, stop the streaming output and finally disconnect 
    """
    def stop(self):
        # Properly stop the video stream and disconnect
        write_stop_output()

        print("Landing...")
        self.drone(
            Landing()
            >> FlyingStateChanged(state="landed", _timeout=5)
        ).success()
        print("Landed\n")

        self.drone.stop_video_streaming()

        self.drone.disconnect()
        
    """
    yuv_frame_cb()
        parameter: 
                yuv_frame:Take a raw yuv frame from the drone's video stream and add it to the queue via reference.
        Add a yuv frame reference to the queue so that threading can take place. This unables the drone to carry out asynchronous movements.
    """
    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.

            :type yuv_frame: olympe.VideoFrame
        """
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)

    """
    flush_cb()
        Return drames from the queue while waiting or entering without blocking past items on the queue
    """
    def flush_cb(self):
        with self.flush_queue_lock:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait().unref()
        return True

    def start_cb(self):
        pass

    def end_cb(self):
        pass


    """
    show_yuv_frame()
        parameters:
            window_name: Name of the window opened.
            yuv_frame: Yuv frame which will be outputted.

        The yuv frame is first converted using OpenCV.

        The new frame drom the queue is being passed to another queue Q of frames.
        Here the frames are passed into the Tensorflow Navigation model, This model predicts the probability 
        the current frame should be labelled as a right turn left turn or centre.

        These probabilities are grouped where a mean of the probability is gotten for 10 frames. This probability 
        is labelled using the LabelBinarizer from Sklearn. This label will be left, right or centre.
        This Label is added to a text file Navigation.txt. From there the drone can uses that file to make a movement.

        Next the label can also be added to the writer for OpenCV. This will add text to the current frame when it is outputted.
        While the OpenCV window is open if the 'q' button is pressed run stop() function to land drone and cut video streaming.
    """
    def show_yuv_frame(self, window_name, yuv_frame):
        info = yuv_frame.info()
        height, width = info["yuv"]["height"], info["yuv"]["width"]

        # convert pdraw YUV flag to OpenCV YUV flag
        cv2_cvt_color_flag = {
            olympe.PDRAW_YUV_FORMAT_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.PDRAW_YUV_FORMAT_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[info["yuv"]["format"]]

        # Use OpenCV to convert the yuv frame to RGB
        cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)
        Q = deque(maxlen = 10)
    
        writer = None
        (W, H) = (height, width)

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = cv2frame.shape[:2]

        #clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 250x250, and then
        output = cv2frame.copy()
        cv2frame = cv2.cvtColor(cv2frame, cv2.COLOR_BGR2RGB)
        cv2frame = cv2.resize(cv2frame, (250, 250)).astype("float32")

        # make predictions on the frame and then update the predictions queue
        preds = model.predict(np.expand_dims(cv2frame, axis=0))[0]

        Q.append(preds)
        
        # perform prediction averaging over the current history of
        # previous predictions
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        
        # extract the label for the maxmimum label probability 
        label = lb.classes_[i]

        # Add the label to Navigation.txt using flight_algorithm function write_navigation_output
        write_navigation_output(label)

        # draw the activity on the output frame
        text = "Operation: {}".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 255, 0), 5)
        
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('UAV CNN Navigation', fourcc, 30,
                (W, H), True)
            
        # write the output frame
        writer.write(output)

        # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            self.stop()

    """
    run()
        create a window to proccess threading of the frames. Here frames can be passed to show_yuv_frame and 
        be outputed in another window. Each frame can then be dropped from the queue and resources can be restored.
    """
    def run(self):
        window_name = "UAV CNN Navigation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        main_thread = next(
            filter(lambda t: t.name == "MainThread", threading.enumerate())
        )
        while main_thread.is_alive():
            with self.flush_queue_lock:
                try:
                    yuv_frame = self.frame_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                try:
                    self.show_yuv_frame(window_name, yuv_frame)
                except Exception:
                    # We have to continue popping frame from the queue even if
                    # we fail to show one frame
                    traceback.print_exc()
                finally:
                    # Don't forget to unref the yuv frame. We don't want to
                    # starve the video buffer pool
                    yuv_frame.unref()
        cv2.destroyWindow(window_name)

    """
    fly()
        Allow the drone to takeoff.
        If the drone is not labelled as stop, allow a loop to control the drone. The read_navigation_output function will take a string
        to be cross referenced through conditional operations. Judging the string, if it is 'left', the drone will turn left and wait for the next instruction.
        If it is 'right' the drone will turn right and wait for the next instruction to come through. If the string is 'centre' the drone will move forward.
        (For testing purposes the centre function is set to move 0 as to examine the feature without damges occuring)
    """
    def fly(self):
        # Takeoff, fly, land, ...
        """
        Uncomment the code snippet below, enabiling the user to use the Sky Remote on the drone
        while the drone can still returns Video feed with Operation predictions on Screen.
        """
        """
        # Takeoff, fly, land, ...
        print("Takeoff if necessary...")
        self.drone(
            FlyingStateChanged(state="hovering", _policy="check")
            | FlyingStateChanged(state="flying", _policy="check")
            | (
                GPSFixStateChanged(fixed=1, _timeout=10, _policy="check_wait")
                >> (
                    TakeOff(_no_expect=True)
                    & FlyingStateChanged(
                        state="hovering", _timeout=10, _policy="check_wait")
                )
            )
        ).wait()
        """
        # This landing condition is used as a redundancy incase of stop() failing.
        if read_navigation_output == 'stop':
            print("Landing...")
            self.drone(
                Landing()
                >> FlyingStateChanged(state="landed", _timeout=5)
            ).success()
            print("Landed\n")

        print("***** Takeoff if necessary *****")
        self.drone(TakeOff(_no_expect=True) >> FlyingStateChanged(state="hovering", _timeout=10, _policy="check_wait")).wait().success()

        # If the UAV has not stopped
        while read_navigation_output() != 'stop':

            print("**** Navigation has started ****")

            if read_navigation_output() == 'left':
                self.drone(moveBy(0, 0, 0, -0.2)).wait().success()
                print('LEFT')

            if read_navigation_output() == 'right':
                self.drone(moveBy(0, 0, 0, 0.2)).wait().success()
                print('RIGHT')

            if read_navigation_output() == 'centre':
                self.drone(moveBy(0, 0, 0, 0)).success()
                print('CENTRE')
"""
__main__
    create the Navigation instance of a drone. Activate the streaming protocols and start the fly processes.
    If the CLI is interupted use write_stop_output to cease drone movement and threads will move onto the stopping function. 
"""
if __name__ == "__main__":

    flightMind = Navigation()
    # Start the video stream
    flightMind.start()
    print("******START COMPLETE*****")
    # Perform some live video processing while the drone is flying
    try:
        flightMind.fly()
    except KeyboardInterrupt:
        write_stop_output()
        flightMind.stop()