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

      
TODO
- V1 cells simulation                                               DONE
- All image formats supported
- Propre and check for plagiarism in apply gaussians                hahahha

"""

# import libraries 
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage  #??
import cv2 
import PySimpleGUI as sg
from skimage import color,filters
import os



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
        self.basename = os.path.basename(self.fileName)
    

    def adjust_image(self):
        ''' If RGB, converts image to grayscale. Then, gets the size of the image. '''

        if len(self.raw_data.shape) == 3:  # if RGB, convert to greyscale
            self.data = color.rgb2gray(self.raw_data)
            print('Image converted to grayscale.')
        else:
            self.data = self.raw_data
        self.size = np.shape(self.data)
        self.size_xy = (self.size[1], self.size[0])
        print(f'Image size in x,y coordinates: {self.size_xy} pixels.')


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
        self.Zones = np.floor(self.RadFromFocus / self.maxDistance * self.numZones).astype(np.uint8)
        self.Zones[self.Zones == self.numZones] = self.numZones - 1  # eliminate the few maximum radii 
                                                          
        print("Zones are computed.")


    def get_RFS(self):
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
        self.RFS_px = []

        # Get the RFS for each zone
        for ii in range(self.numZones):
            zoneRad = (self.maxDistance / self.numZones * (ii + 0.5)) # eccentricity = average radius in a zone, in pixel
            eccentricities = zoneRad * px_to_m # in meters
            angles_rad = np.arctan2(eccentricities, display_distance) # in radians
            eccentricity = angles_rad * r_eye * 1000 # distance from fovea in millimeters
            rfs_arcmin = 6 * eccentricity # receptive field of magnocellular cells in arcmin
            self.RFS_arcmin.append(rfs_arcmin) 
        
        # Convert to px
        for rf in self.RFS_arcmin:
            angle_deg = rf / 60 # convert arcmin to degrees
            rfs_m = np.abs(np.tan(angle_deg*np.pi/180)*display_distance)
            rfs_px = rfs_m * m_to_px
            self.RFS_px.append(rfs_px)      

        print('Receptive fields sizes are computed.')


    def apply_filters_ganglion(self):
        """
        Apply filters to simulate ganglion cells response to input image. 
        """
        img_type = self.data.dtype
        current = np.zeros_like(self.data, dtype = img_type)
        final = np.zeros_like(self.data, dtype = img_type)
        filtered = []
        print(self.RFS_arcmin)
        print("applying filters...")
        # filter the image and store each filtered image in current
        for ii in np.arange(self.numZones):
            
            sigma1 = self.RFS_px[ii]/8    # sigma is side_length/8
            print('sigma', ii, 'is :', sigma1)
            sigma2 = sigma1*1.6
            filt = dog_filter(self.RFS_arcmin[ii],sigma1, sigma2)
            current = cv2.filter2D(self.data, cv2.CV_32F, filt)
            filtered.append(current)
           

        # store in final each filtered zone corresponding to each zone of the image in final
        for ii in np.arange(self.numZones):
            final[self.Zones==ii] = filtered[ii][self.Zones==ii]   
            
        plt.imshow(final, 'gray')
        outname = f'ganglion_{self.basename}'
        plt.savefig(outname)
        plt.show()

    def apply_filters_V1(self):
        ''' Apply gabor filters to the input image for different orientations (theta).
            Produces an image output which consists in the combination of the different
            gabor filters applied to the original image (depending on theta). '''
        
        thetas = np.array([0, 30, 60, 90, 120, 150]) # orientations
        fr = 0.2 # frequency

        # Initialize an empty array to store the merged output
        merged_output = np.zeros_like(self.data)

        # Apply the Gabor filter for each orientation and add the filtered outputs
        for theta in thetas:
            filtered_real, filtered_im = filters.gabor(image = self.data, frequency = fr, theta = np.deg2rad(theta))
            merged_output += filtered_real

        # Normalize the merged output
        merged_output /= len(thetas)

        # Rescale the merged output to the range [0, 255] for visualization
        merged_output = (merged_output - np.min(merged_output)) / (np.max(merged_output) - np.min(merged_output)) * 255
        print('Gabor filters applied to {}'.format(self.fileName))
        print('Output image created')
        plt.imshow(merged_output, 'gray')
        
        outname = f'V1_{self.basename}'
        plt.savefig(outname)
        print('{} successfully wrote.'.format(outname))

        plt.show()

   
def DoG(x, sig1, sig2):
    ''' 
    INPUTS:
    x:          The radius of the kernel
    sigma1: 
    sigma2: 
    ----------
    Returns:  Returns the approximation of a response of a ganglion cell with a DoG filter
    '''
    output = (1/(sig1*np.sqrt(2*math.pi)))*np.exp(-(x**2/(2*sig1**2))) - (1/(sig2*np.sqrt(2*math.pi)))*np.exp(-(x**2/(2*sig2**2)))
    return output

def dog_filter(dim,sigma1, sigma2):
    """
    INPUTS:
    sigma1: 
    sigma2: 
    ----------
    Returns:  Returns a DoG filter matrix/kernel of given size and given sigmas.
    """
    dim=np.round(dim)
    dim=max(3,dim)
    x = np.arange(-int(np.ceil(dim/2)),int(np.ceil(dim/2)))
    X,Y = np.meshgrid(x,x)
    m_filt = DoG(np.sqrt(np.square(X)+np.square(Y)),sigma1,sigma2)
   
    return m_filt/np.sum(np.sum(m_filt))
  

def main():
    retina = Retina()
    retina.get_file()
    retina.adjust_image()
    retina.fix_point()
    retina.furthest_corner()
    retina.make_zones()
    retina.get_RFS()
    retina.apply_filters_ganglion()
    retina.apply_filters_V1()

main()

