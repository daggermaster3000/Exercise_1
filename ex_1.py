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


# main function
def main():
    #get parameters for the simulation
    numElectrodes = 20
    Fmin, Fmax = (200, 500) #[Hz] = [1/s]
    win_size = 6e-3 #[s]
    win_step = 5e-4 #[s]
    n_out_m = True

    #prompt the user for input file and extract the data, rate and filename
    data, rate, filename = get_input_file()

    #run the simulation and get the processed data
    simulate(filename,data,rate,numElectrodes,Fmin,Fmax,win_size,win_step)

    #play the sound






#get input filename using GUI
def get_input_file():

    #Get the absolute path to this file's location
    absolutePath = os.path.abspath(__file__)
    dirName = os.path.dirname(absolutePath)

    #prompt the user to choose an audio file from the /sounds folder and store the filename
    filename = sg.popup_get_file('', no_window = True, initial_folder = dirName+"/sounds")

    #extract rate and data from the audio file
    rate, data = read(filename)
    
    return data, rate, filename



#simulate
def simulate(filename, data, rate, numElectrodes, Fmin, Fmax, win_size, win_step,n_out_m):

    #get numChannels and remove second channel if input sound file is stereo 
    input_sound = Sound(filename)
    (source, rate2, numChannels, totalSamples, duration, bitsPerSample) = input_sound.get_info()
    
    #Check if the audio file is in stereo and keep only the first channel if this is the case (we should merge them instead)
    if numChannels == 2:
        data.astype(float)
        input = data[:, 0]/2 + data[0,:]/2   
    else:
        input = data

    #Computes the filter coefficients for a bank of GammaTone filters
    (forward, feedback, cfs, ERB, B) = gt.GT_coefficients(rate, numElectrodes, Fmin, Fmax, "moore") #cfs is the frequency at which there is an electrode

    #Apply GammaTone to input file
    filtered_data = gt.GT_apply(input, forward, feedback)


    #Window the filtered data
    #Get the window size and step size in terms of index
    win_size = win_size*rate
    win_step = win_step*rate
    win_interval = win_size+win_step

    #pre-allocate memory for the processed data
    processed_data = np.zeros((numElectrodes, len(data)),dtype=np.float64)
    t = np.arange(0, duration, 1/rate)

    for electrode in numElectrodes:

        for win_start in range(0,len(filtered_data),win_interval):
            
            win_stop = win_start + win_size

            # Broadcasting to avoid loops
            amps = filtered_data[electrode,win_start:win_stop]
            omega = 2 * np.pi * cfs[electrode]
            processed_data[electrode,win_start:win_stop] = amps @ np.sin(omega * t)
        
        #finish the last points
        amps = filtered_data[electrode,win_stop:]
        processed_data[electrode,win_stop:] = amps @ np.sin(omega * t)


    #n out of m
    if n_out_m == True:

        for win_start in range(0,len(filtered_data),win_interval):
            
            if win_start>(len(filtered_data)-(win_interval)):
                break

            for electrode in numElectrodes:


        

    #return processed data

if __name__ == "__main__":
    main()