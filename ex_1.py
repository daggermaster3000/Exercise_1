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


#get parameters for the simulation
n_electrodes = 20
freq_lower, freq_upper = (200, 500) #Hz
win_size = 6e-3 #s
win_step = 5e-4 #s

#get input filename using GUI
def get_input_file():

    absolute_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(absolute_path)
    filename = sg.popup_get_file('', no_window = True, initial_folder = dir_name+"/sounds")
    
    rate, data = read(filename)
    return data, rate

data, rate = get_input_file()

#simulate
def simulate(data, rate, n_electrodes, freq_lower, freq_upper, win_size, win_step):
    #gamma tone process input file
    (forward, feedback, fcs, ERB, B) = gt.GT_coefficients(rate, n_electrodes, freq_lower, freq_upper, "moore")
    processed_data = gt.GT_apply(data, forward, feedback)


    #window input file
        #map the max energy to the closest electrode

    #output file and playback