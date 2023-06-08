"""
Exercise 3: Simulation of a "visual prosthesis"
Authors: Quillan Favey, Alessandro Pasini
Version: 1.0
Date: 24.05.2023

This file contains code for simulating a visual prosthesis

OUTPUTS: 
      FILES                     DESCRIPTION                                         VALUES


NAMING CONVENTIONS: 


NOTES/USAGE:
    - Run it from your IDE or alternatively from a command prompt by activating 
      your environment and running with: python ...

"""

# import libraries 
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage
import cv2
import PySimpleGUI as sg
from skimage import color,filters
import sys
import threading
import time
import itertools



class Retina:
    ''' Class for simulating ganglion cells and V1 cells responses. '''



    def get_file(self, in_file = None):
        ''' Get file (if not specified before) using a popup window '''

        if in_file == None:
            self.fileName = sg.popup_get_file("", no_window=True)
        else:
            self.fileName = in_file
        self.raw_data = plt.imread(self.fileName)
        print(f'Image {self.fileName} successfully read.')
    

    def adjust_image(self):
        ''' If RGB, converts image to grayscale. Then, gets the size of the image. '''

        if len(self.raw_data.shape) == 3:  # if RGB, convert to greyscale
            self.data = color.rgb2gray(self.raw_data)
            print('Image converted to grayscale.')
        self.size = np.shape(self.data)
        self.size_xy = (self.size[1], self.size[0])
        print(f'Image size in y,x coordinates: {self.size_xy} pixels.')


    def fix_point(self):
        ''' Show the image, and get the focus point for the subsequent calculations. 
            To select a fixation point, click on the image '''
        
        plt.imshow(self.data, 'gray')
        selected_focus = plt.ginput(1)
        plt.close()
        # ginput returns a list with one selection element, take the first
        # element in this list
        self.focus = np.rint(selected_focus)[0].tolist()
        self.focus.reverse()
        print(f'selected fixation point in y,x coordinates: {self.focus}.')

    
    def furthest_corner(self):
        ''' Finds the furthest corner from the selected focus '''
        self.corners = np.array([0,0,
                                 self.size_xy[0],0,
                                 0,self.size_xy[1],
                                 self.size_xy[0],self.size_xy[1]]).reshape(4,2)
        self.fp = np.tile(self.focus, 4).reshape(4,2)
        self.distances = np.sqrt(np.sum((self.corners - self.fp)**2, axis = 1))     # calculate the euclidian distances
        self.maxDistance = np.max(self.distances)
        print(f'Furthest corner is at {np.round(self.maxDistance, decimals = 0)} pixels.')
    
    
    def make_zones(self):
        ''' Creates "numZones" different circular regions around the chosen focus 
            on the greyscaled image. Calculations are based on the maximum radius,
            i.e. the maximal distance from the selected focus to the furthest corner'''
        # Set the number of zones
        self.numZones = 10

        # The np.meshgrid function is used to create coordinate grids X and Y that span the 
        # dimensions of the size array.
        X, Y = np.meshgrid(np.arange(self.size[0]) + 1, np.arange(self.size[1]) + 1)

        # Calculating distances from focus: Euclidean distances from the focus point to each 
        # point in the grid are calculated. The resulting distances are stored in RadFromFocus. 
        self.RadFromFocus = np.sqrt((X.T - self.focus[0]) ** 2 + (Y.T - self.focus[1]) ** 2) # px
        
        # Assign each value to a Zone, based on its distance from the focus. 
        # Every radius is normalized to the maxDistance radius ([0, 1])
        # By multiplyng for the number of zones, and flooring the result, every radius will be 
        # categorized as 0, 1, 2, 3, ..., numZones-1
        self.Zones = np.floor(self.RadFromFocus / self.maxDistance * self.numZones).astype(np.uint8)
        self.Zones[self.Zones == self.numZones] = self.numZones - 1  # eliminate the few maximum radii 
                                                           # (if a radio si long as the max radius, it will fall in the numZones zone. 
                                                           # Since zones start from 0 to numZones-1
                                                           # it should be put in the numZones - 1 zone)

        #just to observe: colormap to visualize zones by color. HAS TO BE REMOVED
        """
        cmap = plt.get_cmap('jet', self.numZones)
        plt.imshow(self.Zones, cmap=cmap)
        plt.colorbar(ticks=np.arange(self.numZones))
        plt.show()
        """
        print("zones done")

   
    def make_filters(self):
        ''' Converts pixel location to retinal location: each eccentricity is converted in distance 
            from the fovea. Then, receptive fields are calculated (in arcmin) and translated to the 
            display image (in pixels).'''
        # parameters
        display_resolution = 1400 # px
        height = 0.3 # m
        display_distance = 0.6 # m
        r_eye = 1.25e-2 # m
        px_to_m = height / display_resolution
        m_to_px = 1 / px_to_m
        self.filtered = []
        
        self.RFS_arcmin = []
        for ii in range(self.numZones):
            zoneRad = (self.maxDistance / self.numZones * (ii + 0.5)) # eccentricity = average radius in a zone, in pixel
            eccentricities = zoneRad * px_to_m # in meters
            angles_rad = np.arctan2(eccentricities, display_distance) # get the angle between the horizontal line crossing through the fovea to the pixel in radians
            eccentricity = angles_rad * r_eye * 1000 # get the distance from fovea in millimeters
            rfs_arcmin = 6 * eccentricity # receptive field of magnocellular cells in arcmin
            self.RFS_arcmin.append(rfs_arcmin) 

        self.RFS_px = []
        for rf in self.RFS_arcmin:
            angle_deg = rf / 60 # convert arcmin to degrees
            rfs_m = np.abs(np.tan(angle_deg)*display_distance)
            rfs_px = rfs_m * m_to_px
            self.RFS_px.append(rfs_px)
        print("applying filter")
        for zone in range(0,len(self.Zones)-1):
            sig1 = self.RFS_px[zone]
            sig2 = sig1*1.6
            self.filter_matrix = DoG_kernel(self.RFS_px,sig1,sig2)
            current = cv2.filter2D(self.data,cv2.CV_32F,self.filter_matrix)
            self.filtered.append(current)
        print("filter applied")

        ## Paste image together
        self.out = np.zeros_like(self.data)
        for i in reversed(0,len(self.Zones)-1):
            self.out[self.zones[i][0]] = self.filtered[i] \
                                         [self.zones[i][0]]
        print('DoG kernels were applied, output image was created.')
        plt.imshow(self.out, 'gray')
        outname = 'ganglion_{}'.format(self.imgtitle)
        plt.savefig(outname)
        print('Successfully wrote file {}'.format(outname))     

        
def DoG(x, sig1, sig2):
    ''' 
    Approximation of the response of ganglion cells with a Difference of Gaussians
    PARAMETERS:
    x: The radius of the kernel
    '''
    output = (1/(sig1*np.sqrt(2*math.pi)))*np.exp(-(x**2/(2*sig1**2))) \
        - (1/(sig2*np.sqrt(2*math.pi)))*np.exp(-(x**2/(2*sig2**2)))
    return output

def DoG_kernel(rfs,sig1,sig2):
    '''A function to create a DoG kernel'''
    dim = 10*sig1
    filterLength = np.arange(-int(2 * np.ceil(dim/2)),int(2 * np.ceil(dim/2)))

    x,y = np.meshgrid(-(filterLength/2),(filterLength/2)) # create a grid of adequate dimensions
    kernel = DoG(np.sqrt(np.square(x),np.square(y)),sig1,sig2)
    
    return kernel
  

def main():
    retina = Retina()
    retina.get_file()
    retina.adjust_image()
    retina.fix_point()
    retina.furthest_corner()
    retina.make_zones()
    
    retina.make_filters()

main()

