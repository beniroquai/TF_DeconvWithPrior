#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:50:30 2016

@author: useradmin
"""
import tensorflow as tf
import numpy as np
import h5py 
from matplotlib import pyplot as plt
import scipy as scy
import scipy.io as sio
import time



def batch_fftshift2d(tensor):
        # Shifts high frequency elements into the center of the filter
        indexes = len(tensor.get_shape()) - 1
        top, bottom = tf.split(tensor, 2, indexes)
        tensor = tf.concat([bottom, top], indexes)
        left, right = tf.split(tensor, 2, indexes - 1)
        tensor = tf.concat([right, left], indexes - 1)

        return tensor

def batch_ifftshift2d(tensor):
        # Shifts high frequency elements into the center of the filter
        indexes = len(tensor.get_shape()) - 1
        left, right = tf.split(tensor, 2, indexes - 1)
        tensor = tf.concat([right, left], indexes - 1)
        top, bottom = tf.split(tensor, 2, indexes)
        tensor = tf.concat([bottom, top], indexes)
        return tensor    

def tf_FT(input):
    return batch_fftshift2d(tf.fft2d(input))
    

def tf_iFT(input):
    return tf.ifft2d(batch_ifftshift2d(input))


def rr(inputsize_x=256, inputsize_y=256, inputsize_z=1):
    x = np.linspace(-inputsize_x/2-1,inputsize_x/2, inputsize_x)
    y = np.linspace(-inputsize_y/2-1,inputsize_y/2, inputsize_y)
    z = np.linspace(-inputsize_z/2-1,inputsize_z/2, inputsize_z)
    
    xx, yy, zz = np.meshgrid(x, y, z)
    r = np.sqrt(xx**2+yy**2+zz**2)
    r = np.squeeze(r)
    return r
    
    
def FT(input):
    return np.fft.fftshift(np.fft.fft2(input))
    

def iFT(input):
    return (np.fft.ifft2(np.fft.ifftshift(input)))

def SimNoiseImg(ObjName = '../images/barbara256.tif', rRatio=8, MaxPhotons=100):
    
    #ObjName = '../images/einstein1.tif'
    #ObjName = '../images/devoogd.tif'
    rRatio=8
    MaxPhotons=100
    
    # Read image data, representing the object
    img_raw =  np.resize(scy.misc.imread(ObjName).astype(float), [256, 256])
    #
    #try img_raw.shape(2) > 0:
    #    obj_original = img_raw[:,:,0]
    #else:
    
    obj_original = img_raw
    
    # Get size of object (assume equal size in vertical and horizontal
    # direction)
    mysize = np.size(obj_original,1);
    
    # Get Fourier representation
    ft_obj = FT(obj_original);
    
    
    ## Calculate PSF
    # Generate pupil
    
    r = rr(mysize, mysize)
    r0 = np.floor(mysize/rRatio)
    pupil = r < r0
    
    
    # Calculate APSF, IPSF and OTF
    apsf = iFT(pupil)
    ipsf = np.real(apsf*np.conj(apsf))
    otf = FT(ipsf)
    otf = otf/np.max(np.abs(otf))
    
    # Calculate the image of the object
    ft_img = ft_obj * otf
    img = np.real(iFT(ft_img));
    
    # Normalize to interval 0...255; for later visual comparison
    #img = img - min(img);
    img = MaxPhotons/np.max(img) * img
    
    
    #plt.imshow(img, cmap = 'gray')
    #plt.imshow(noisy_img, cmap = 'gray')
    
    
    
    
    #%% Add Poisson-distributed noise to the image 
    #% Use 'noise'-function included in dipLib
    #
    # create signal-dependent noisy image
    img_noise = np.random.poisson(img,size=None).astype(float) #add poison noise, where each pixel value corresponds to the number of photons
    
    obj_original = obj_original / np.mean(obj_original) * np.mean(img) #to get the correct scaling

    return img_noise,img,otf,obj_original
    
    
    
    
    
    
def PoissonErrorAndDeriv(guess,measured,otf,lambda0): 
    NormFac = 1;
    
    Fwd = np.real(iFT(FT(guess) * otf))+0j
    totalError = np.sum((Fwd-measured) - measured * np.log(Fwd/(measured+0.1)))
    totalError = totalError + lambda0*np.sum(guess**2);
    
    myRes = (1 - measured / Fwd)
    Bwd = np.real(iFT(FT(myRes) * np.conj(otf)))
    Bwd = Bwd + lambda0*2*guess
    
    myDeriv = Bwd * NormFac
    totalError = totalError * NormFac
#/Users/Bene/Dropbox/Dokumente/Promotion/PYTHON/TF_Deconv/generatePSF
    return totalError, myDeriv
    
    
def PoissonErrorAndDerivTF(tf_guess,tf_measured,tf_otf,tf_mylambda): 
    NormFac = 1;

    # tensorflow's FFT cannot handle non-complex values -> convert! 
    tf_guess_cmplx = tf.complex(tf_guess, tf.zeros(tf_guess.get_shape()));
    tf_otf_cmplx = tf.complex(tf_otf, tf.zeros(tf_guess.get_shape()));
    
    Fwd = tf.real(tf_iFT(tf_FT(tf_guess_cmplx) * tf_otf_cmplx))
    totalError = tf.reduce_sum((Fwd-tf_measured) - tf_measured * tf.log(Fwd/(tf_measured)), reduction_indices=[0, 1])
    totalError = totalError + tf_mylambda*tf.reduce_sum(tf_guess**2, reduction_indices=[0, 1]);
    
    myRes = (1 - tf_measured / Fwd)
    totalError = tf.real(totalError * NormFac)
    

    return totalError, Fwd
    
def ForcePosSqr(anImg,aFkt):
#%minVal = 0.1;
#%anImg(anImg < minVal) = minVal; % Force MinVal

    [myerr,myder]=aFkt(abssqr(anImg));
    myder = myder * 2 * anImg # This is exp(anImg) but was overwritten
    

    return myerr, myder