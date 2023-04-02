"""
A python code to simulate the output of a cochlear implant...
"""

#import libraries
import os
import numpy as np
import PySimpleGUI as sg
import GammaTones as gt
from scipy.io.wavfile import read
from sksound.sounds import Sound


def main():
    '''
    Main method: set parameters, read in sound file and simulate cochlear implant audio 
    transformation. Saves output file in current directory. 
    '''

    #get parameters for the simulation
    numElectrodes = 20 
    Fmin, Fmax = (200, 1500) 
    win_size = 6e-2 
    win_step = 7e-2 

    #n-out-of-m strategy (m corresponds to numElectrodes)
    n_out_m = True #if True, the strategy will be applied
    n = 6

    #prompt the user for input file; extracts data and rate 
    data, rate, input_sound, filename = get_input_file()
    
    #play the input sound file
    print('Now playing input file!')
    input_sound.play()
    print()

    #run the simulation and get the processed data
    sound_out = simulate(input_sound, data, rate, numElectrodes, Fmin, Fmax, win_size, win_step, 
                         n_out_m, n)

    #set output directory by removing ".wav" and adding "_out.wav" to input file directory
    output_name = "{}_out.wav".format(filename[:-4])
    sound_out.write_wav(output_name)

    #play the sound
    print('Now playing output file!')
    sound_out.play()


def get_input_file():
    ''' get input filename using GUI and extracts data and rate using scipy.io.wavfile read()
        function
    '''

    #Get the absolute path to this current file's location and its directory
    absolutePath = os.path.abspath(__file__)
    dirName = os.path.dirname(absolutePath)

    #prompt the user to choose an audio file from the /sounds folder and store the filename
    filename = sg.popup_get_file('', no_window = True, initial_folder = dirName+"/sounds")

    #extract rate and data from the audio file; save audio file as a sound class
    rate, data = read(filename)
    inputSound = Sound(filename)
    
    return data, rate, inputSound, filename


#simulate
def simulate(inputSound, data, rate, numElectrodes, Fmin, Fmax, win_size, win_step, n_out_m, n):
    '''This function simulates a cochlear implant audio transformation

    Gammatone-filters are applied to the whole file. The output is then subdivide into 
    time-windows. 

    Parameters
    ----------
    inputSound: .. audio file as a sound class 
    data: ........ data
    rate: ........ rate [Hz]
    numElectrodes: number of available electrodes of the (simulated) cochlear implant
    Fmin: ........ minimum frequency of cochlear implant electrodes [Hz]
    Fmax: ........ maximum frequency of cochlear implant electrodes [Hz]
    win_size: .... duration of the time-window [s]
    win_step: .... duration of the time-step [s]
    n_out_m: ..... option to apply n-out-of-m strategy. If True, this strategy is applied
                   m corresponds to numElectrodes
    n: ........... number of  electrodes with the largest stimulation that will be activated 
                   at any given time

    Returns:
    -------
    output_sound_object: a one-dimensional array containing the processed audio
    '''

    #get information from input sound: number of channels, data length, duration
    (source, rate2, numChannels, totalSamples, duration, bitsPerSample) = inputSound.get_info()
    nData = totalSamples

    #Check if the audio file is in stereo and keep only the first channel if this is the case (we should merge them instead)
    if numChannels == 2:
        data.astype(float)
        input = np.sum(data,axis=1)/2
        print(np.shape(input))
    else:
        input = data

    #Computes the filter coefficients for GammaTone filters. 
    #cfs is the frequency at which there is an electrode
    (forward, feedback, cfs, ERB, B) = gt.GT_coefficients(rate, numElectrodes, Fmin, Fmax, "moore") 

    #Apply GammaTone to input file
    filtered_data = gt.GT_apply(input, forward, feedback)

    #Window the filtered data
    #Get the window size and step size in terms of indices
    win_size = int(win_size*rate)
    win_step = int(win_step*rate)
    win_interval = win_size + win_step

    #pre-allocate memory for the processed data
    processed_data = np.zeros((numElectrodes, nData), dtype = np.float64)
    t = np.arange(0, duration, 1/rate)

    for electrode in range(numElectrodes):

        for win_start in range(0, len(filtered_data), win_interval):
            
            win_stop = win_start + win_size

            # Broadcasting to avoid loops
            amps = filtered_data[electrode, win_start:win_stop]
            omega = 2 * np.pi * cfs[electrode]
            processed_data[electrode, win_start:win_stop] = amps * np.sin(omega * t[win_start:win_stop])
        
        #finish the last points
        amps = filtered_data[electrode, win_stop:]
        processed_data[electrode, win_stop:] = amps * np.sin(omega * t[win_stop:])

    #Compute the sum of all electrodes to get a single audio track

    #n out of m
    if n_out_m == True:
        outputSound = np.zeros(nData, dtype = float)
        #get the intensities for each electrode in the given time window
        for win_start in range(0, len(filtered_data), win_interval):
            win_stop = win_start + win_size
            Intensities = []
            
            for electrode in range(numElectrodes):
                Intensities.append(np.square(np.sum(processed_data[electrode, win_start:win_stop])))
            #add the n most activated electrodes to the output
            n_elec = np.argpartition(Intensities, n)[:n]
            outputSound[win_start:win_stop] = np.sum(processed_data[n_elec, win_start:win_stop], axis=0)
        
        #do the same for the last points 
        Intensities = []
        for electrode in range(numElectrodes):
            Intensities.append(np.square(np.sum(processed_data[electrode, win_stop:])))
        #add the n most activated electrodes to the output
        n_elec = np.argpartition(Intensities, n)
        outputSound[win_stop:] = np.sum(processed_data[n_elec, win_stop:], axis=0)

    else:
        outputSound = np.sum(processed_data, axis=0)
    output_sound_object = Sound(inData = outputSound, inRate = rate)

    return output_sound_object


if __name__ == "__main__":
    main()