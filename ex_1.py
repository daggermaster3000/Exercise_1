"""
A python code to simulate the output of a cochlear implant...
"""

#import libraries

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import read
import GammaTones as gt
import PySimpleGUI as sg
from sksound.sounds import Sound


#get parameters for the simulation
numElectrodes = 20
Fmin, Fmax = (200, 500) #[Hz] = [1/s]
win_size = 6e-3 #[s]
win_step = 5e-4 #[s]

#get input filename using GUI
def get_input_file():

    absolutePath = os.path.abspath(__file__)
    dirName = os.path.dirname(absolutePath)
    filename = sg.popup_get_file('', no_window = True, initial_folder = dirName+"/sounds")
    
    rate, data = read(filename)
    return data, rate, filename

data, rate, filename = get_input_file()

#simulate
def simulate(filename, data, rate, numElectrodes, Fmin, Fmax, win_size, win_step):

    #get numChannels and remove second channel if input sound file is stereo 
    input_sound = Sound(filename)
    (source, rate2, numChannels, totalSamples, duration, bitsPerSample) = input_sound.get_info()

    if numChannels == 2:
        data.astype(float)
        input = data[:, 0]
    else:
        input = data

    #Computes the filter coefficients for a bank of GammaTone filters
    (forward, feedback, fcs, ERB, B) = gt.GT_coefficients(rate, numElectrodes, Fmin, Fmax, "moore")

    #Apply GammaTone to input file
    processedData = gt.GT_apply(input, forward, feedback)


    #window input file
        #map the max energy to the closest electrode

    #output file and playback