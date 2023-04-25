
#import libraries
import os
import numpy as np
import PySimpleGUI as sg
import GammaTones as gt
from sksound.sounds import Sound


def main():
    '''
    Main method: set parameters, read in sound file and simulate cochlear implant audio 
    transformation. Saves output file in current directory. 
    '''

    #get parameters for the simulation
    numElectrodes = 20 
    Fmin, Fmax = (200, 5000) 
    win_size = 1e-3
    win_step = 5e-4

    #n-out-of-m strategy (m corresponds to numElectrodes)
    n_out_m = True #if True, the strategy will be applied
    n = 1

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
    ''' get input filename using GUI, saves it as sound class; extracts data and rate with 
        sksound.sounds.Sound() function
    '''

    #Get the absolute path to this current file's location and its directory
    absolutePath = os.path.abspath(__file__)
    dirName = os.path.dirname(absolutePath)

    #prompt the user to choose an audio file from the /sounds folder and store the filename
    filename = sg.popup_get_file('', no_window = True, initial_folder = dirName+"/sounds")

    #extract rate and data from the audio file; save audio file as a sound class
    inputSound = Sound(filename)
    rate = inputSound.rate
    data = inputSound.data

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

    #Check if the audio file is in stereo
    #merge the two channels if the audio file is stereo
    if numChannels == 2:
        data.astype(float)
        input = np.sum(data, axis=1)/2
        
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

    #start
    for electrode in range(numElectrodes):
         
        for i in range(0, nData + win_step, win_step):

            win_start = i
            win_stop = i + win_size
            if win_stop > 0 and win_stop < nData:
                ft = filtered_data[electrode, win_start:win_stop]
                stim = np.sum(np.square(ft))
                amp = np.sqrt(stim)
                omega = 2 * np.pi * cfs[electrode]

                processed_data[electrode, win_start:win_stop] = amp * np.sin(omega * t[win_start:win_stop])

        #if windowing does not match enitre duration, add the last point (i.e, the last smaller window ) 
        ft = filtered_data[electrode, win_stop:]
        stim = np.sum(np.square(ft))
        amp = np.sqrt(stim)
        processed_data[electrode, win_stop:] = amp * np.sin(omega * t[win_stop:])

    #n out of m strategy
    if n_out_m == True:
        outputSound = np.zeros((n, nData), dtype = np.float64)
        for j in range(0, nData + win_step, win_step):
                win_start = j
                win_stop = j + win_size 
                if win_stop > 0 and win_stop < nData:
                    Intensities = []
                    for electrode in range(numElectrodes):
                        intensity = np.sum(np.square(filtered_data[electrode, win_start:win_stop]))
                        Intensities.append([intensity, electrode])
                    Intensities = sorted(Intensities, key=lambda x: x[0])
                    topIntensities = Intensities[-n:] #list with lists [intensity, number_electrode] for top n intensities

                    for k in range(n):
                        outputSound[k, win_start:win_stop] = processed_data[topIntensities[k][1], win_start:win_stop]
                        outputSound[k, win_stop:] = processed_data[topIntensities[k][1], win_stop:]
        
        outputSound = np.sum(outputSound, axis = 0)

    
    else:
        outputSound = np.sum(processed_data, axis = 0)

    output_sound_object = Sound(inData = outputSound, inRate = rate)

    return output_sound_object


if __name__ == "__main__":
    main()