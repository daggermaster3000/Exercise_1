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


#get input filename using GUI
filename = sg.popup_get_file('', no_window=True)

audioDir = 'Sounds'
soundFile = os.path.join(audioDir, filename)
rate, data = read(soundFile)
print(data)

#get settings for the simulation
n_electrodes = ...
freq_range = ...
win_size = ...
win_type = ...
sample_rate = ...



#gamma tone process input file
(forward, feedback, fcs, ERB, B) = gt.GT_coefficients(rate, 50, 200, 3000, "moore")
y = gt.GT_apply(data, forward, feedback)

# Show the plots
fig, axs = plt.subplots(1, 2)

# Show all frequencies, and label a selection of centre frequencies
gt.show_basilarmembrane_movement(y, rate, fcs, [0, 9, 19, 29, 39, 49], axs[0])

# For better visibility, plot selected center-frequencies in a second plot.
# Dont plot the centre frequencies on the ordinate.
gt.show_basilarmembrane_movement(y[[0, 9, 19, 29, 39, 49], :], rate, [], [], axs[1])
plt.show()
plt.close()




#window input file
    #map the max energy to the closest electrode

#output file and playback