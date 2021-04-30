# Final Year Project: FlightMind
***
## Author: Josh Hudziak
## Date: 30/04/21
***
## Copyright:
    This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/.

## Dependencies: 
    * Ubuntu 18.04
    * Olympe 
    * Python
    * OpenCV
    * Numpy
    * TensorFlow GPU
    * Sklearn
    * Imutils
    * Pickle

## Installation: (Ubuntu 18.04 ONLT!)
    ### OLYMPE:
        1. cd $HOME
        2. mkdir code/parrot-groundsdk
        3. repo init -u https://github.com/Parrot-Developers/groundsdk-manifest.git
        4. repo sync
        + (If some libraries do not install when activating olympe env use pip install to get them)

    ### CUDDN (for nvidia GPUs only):
        Follow Ubuntu 18.04 instructions on
        https://tensorflow.org/install/gpu
        
    ### Tensorflow:
        pip install tensorflow *Remeber to be in the Olympe enviroment *version >2.0 will be GPU capable