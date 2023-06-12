#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
COMPUTER SIMULATIONS OF SENSORY SYSTEMS: RETINAL IMPLANT

This script simulates the responses of Ganglion and V1
cells, respectively given a user-defined focal point and 
saves the convoluted output images to the folder of the
selected image.
'''

'''
Authors: Alexandra Gastone, Moritz Gruber, Patrick Haller
Date: June 2018
Version: 1.0

USAGE
=====

A. From python

1) Create an instance of the class Retina. Either pass the image 
   directly or select it in the gui.
2) Calling .get_input() will allow you to select a focal point. The
   function then also computes the distance to the furthest corner.
3) .make_zones() will create 10 concentric zones around the focal point
   and compute the associated receptive field sizes.
4) .apply_kernels() will apply the differnce-of-gaussian kernels to the
   respective zone of the image and paste the output image together.
5) .apply_gabor() will simulate the activity in V1, combining gabor filters
   with different orientations.
   orientation.

B. From the command line

1) 'python retina.py'

'''
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import cv2 as cv2
import skinematics as skin
import os
from numpy import sin, cos, exp, pi
from skimage.color import rgb2gray

class Retina(object):
    '''Class for simulating ganglion cell responses. '''

    def __init__(self,img=None):
        '''Initialization method'''
        self.img = None
        self.img = img if img != None else self.select_img()
        self.dims = self.img.shape

    def prep_img(self):
        '''Converts image to grayscale and 16-bit int and gets its
        dimensions.'''
        if len(self.dims)>2:         # check if RGB
            self.img = self.img.dot([0.2125, 0.7154, 0.0721])\
                                    .astype(np.uint16)
            print('Image converted to grayscale and 16 bit-int.')

    def select_img(self):
        '''Calls a GUI to manually select an image.'''
        (inFile, inPath) = skin.misc.get_file(DialogTitle='Select input image: ')
        self.filename = os.path.join(inPath, inFile)
        self.imgtitle = inFile
        self.img = plt.imread(self.filename)
        print('Image ' + inFile + ' successfully read.')
        return self.img

    def get_input(self):
        '''Lets the user select a fixation point on the image. Also 
        finds the furthest corner to the selected point.'''
        ## Show image, get input
        plt.gray()
        plt.imshow(self.img)
        self.xy = plt.ginput(1)
        plt.close()
        print('Input was registered.')

        ## Find furthest corner
        self.corners = np.array([0,0,self.dims[1],self.dims[1], \
                    0,self.dims[0],0,self.dims[0]]).reshape(2,4)
        self.repeat_fp = np.repeat(self.xy[0],4).reshape(2,4)
        self.xy_dist_sq = np.square(self.corners-self.repeat_fp)
        self.dist_abs = np.sqrt(np.sum(self.xy_dist_sq,axis=0))
        self.max_dist = np.max(self.dist_abs)
        print('The furthest corner is ',np.round(self.max_dist,0), \
               'pixels away.')

    def make_zones(self):
        '''Creates 10 concentric zones around the focal point and 
        computes the corresponding eccentricities, and receptive
        field sizes in px.'''

        self.height_screen = 0.3    # screen height
        self.distance_screen = 0.6          # distance from screen
        self.radius_eye = 0.0125  # radius of the eye

        self.zones = []
        xs = np.arange(0,self.dims[0])-self.xy[0][1]
        ys = np.arange(0,self.dims[1])-self.xy[0][0]
        X,Y = np.meshgrid(xs,ys)

        # radii in pixels 
        self.radii_px = np.linspace(self.max_dist/10,self.max_dist,10)
        # radii in meters
        self.radii_m = self.radii_px/self.dims[1]*self.height_screen
        # angles in meters
        self.angles = np.abs(np.arctan(self.radii_m/self.distance_screen))
        # distance from fovea in meters
        self.eccentricity = self.angles*self.radius_eye
        # receptive field size radii in degree
        self.rfs_degree = 10 * self.eccentricity * 1000 / 60
        # receptive field size radii in meters
        self.rfs_m = np.abs(np.tan(self.rfs_degree)*self.distance_screen)
        # receptive field size radii in pixel
        self.rfs_px = (self.rfs_m/self.height_screen)*self.dims[1]

        for k in range(10): # Create one mask per zone
            self.zones.append(np.array([norm(X.T,Y.T)< \
                              self.radii_px[k]], dtype=bool))
        print('Zones and receptive field sizes were computed.')
    
    def apply_kernels(self):
        '''Convolves DoG kernels with their respective zones and pastes
        the final image together .'''
        plt.figure()
        self.filtered = []
        for i in range(10):
            sig_1 = self.rfs_px[i]/30
            sig_2 = sig_1*1.6
            self.m_filt = dog_matrix(sig_1,sig_2)
            current = cv2.filter2D(self.img,cv2.CV_32F,self.m_filt)
            self.filtered.append(current)

        ## Paste image together
        self.out = np.zeros_like(self.img)
        for i in reversed(range(10)):
            self.out[self.zones[i][0]] = self.filtered[i] \
                                         [self.zones[i][0]]
        print('DoG kernels were applied, output image was created.')
        plt.imshow(self.out, 'gray')
        outname = 'ganglion_{}'.format(self.imgtitle)
        plt.savefig(outname)
        print('Successfully wrote file {}'.format(outname))

    def apply_gabor(self):
        ''' Applies Gabor kernel to the original image for varying
        orientations(theta) [0,30,60,90,120,150], and outputs combined image
                Gabor parameters:
                        lambda: wavelength of the sinusoidal factor (more or less narrow)
                        theta: orientation of the normal to the parallel stripes of Gabor function
                        psi: phase offset
                        sigma: standard deviation of Gaussian enveloppe'''

        sigma = 0.2
        Lambda = 0.5
        theta = (np.array([0, 30, 60, 90, 120, 150])) * pi / 180.  # in rad
        psi = 90 * pi / 180.  # in rad
        kernel_size = 22

        xs = np.linspace(-1., 1., kernel_size)
        ys = np.linspace(-1., 1., kernel_size)
        x, y = np.meshgrid(xs, ys)

        plt.figure()

        self.gabor_values = []
        for i in range(len(theta)):
            x_theta = x * cos(theta[i]) + y * sin(theta[i])
            y_theta = -x * sin(theta[i]) + y * cos(theta[i])
            current = np.array(exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2) *
                               cos(2. * pi * x_theta / Lambda + psi), dtype=np.float32)
            self.gabor_values.append(current)
            self.filtered_gabor = cv2.filter2D(self.img, cv2.CV_32F, current)
            plt.imshow(self.filtered_gabor, 'gray')


        print('Gabor kernels were applied, output image was created.')
        #plt.show()
        outname = 'v1_{}'.format(self.imgtitle)
        plt.savefig(outname)
        print('Successfully wrote file {}'.format(outname))
        plt.show()

def dog(x,sig_1,sig_2):
    '''Evaluates a radial difference of Gaussians filter.'''
    out = (1/(sig_1*np.sqrt(2*math.pi)))*np.exp(-(x**2/(2*sig_1**2))) \
        - (1/(sig_2*np.sqrt(2*math.pi)))*np.exp(-(x**2/(2*sig_2**2)))  
    return out
    
def dog_matrix(sig_1,sig_2):
    '''Returns a DoG filter matrix of given size and given sigmas.'''
    dim = 10*sig_1
    x = np.arange(-int(np.ceil(dim/2)),int(np.ceil(dim/2)))
    X,Y = np.meshgrid(x,x)
    m_filt = dog(np.sqrt(np.square(X)+np.square(Y)),sig_1,sig_2)
    return m_filt/np.sum(np.sum(m_filt))

def norm(x,y):
    '''Returns the euclidean norm.'''
    return np.sqrt(x**2+y**2)


def main():
    # Run pipeline
    r = Retina()
    r.get_input()
    r.prep_img()
    r.make_zones()
    r.apply_kernels()
    r.apply_gabor()

if __name__ == '__main__':
    main()