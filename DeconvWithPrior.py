#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:04:52 2016

@author: useradmin
"""

import SimNoiseImg as simn
from SimNoiseImg import tf_iFT, tf_FT
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

img_noise,img,otf,obj = simn.SimNoiseImg('./devoogd.tif');


plt.imshow(img_noise)
plt.show()


guess = img_noise*0.0+np.mean(img_noise)
mylambda=1

#totalError, myDeriv = simn.PoissonErrorAndDeriv(guess,img_noise,otf,mylambda)


# typeconversion from python to tensorflow - all cmplx values correct?!
tf_otf = tf.constant(otf, dtype=tf.float32, name = "tf_otf")
tf_measured = tf.constant(img_noise, dtype = tf.float32, name = "tf_measured")
tf_guess = tf.Variable(guess, dtype= tf.float32, name = "tf_guess")
tf_mylambda = tf.constant(mylambda, dtype = tf.float32, name = "tf_mylambda")


# define error metric
#totalError, myFWD = simn.PoissonErrorAndDerivTF(tf_guess,tf_measured,tf_otf,tf_mylambda)


NormFac = 1;

# tensorflow's FFT cannot handle non-complex values -> convert! 
tf_guess_cmplx = tf.complex(tf_guess, tf.zeros(tf_guess.get_shape()));
tf_otf_cmplx = tf.complex(tf_otf, tf.zeros(tf_guess.get_shape()));

Fwd = tf.real(tf_iFT(tf_FT(tf_guess_cmplx) * tf_otf_cmplx))
totalError = tf.reduce_sum((Fwd-tf_measured) - tf_measured * tf.log(Fwd/(tf_measured)), reduction_indices=[0, 1])
totalError = totalError + tf_mylambda*tf.reduce_sum(tf_guess**2, reduction_indices=[0, 1]);

myRes = (1 - tf_measured / Fwd)
totalError = tf.real(totalError * NormFac)


# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_step = optimizer.minimize(totalError)

# Initializing the variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# start inference using Tensorflow and its autodifferenciation asöldkfjadflsöjk
NEPOCH = 100
for i in range(NEPOCH):
    sess.run(train_step)
    
    
    if(np.mod(i, 10)):
        tf_loss = sess.run(totalError)
        result = tf_guess.eval()     
        plt.imshow(result, cmap='gray')
        plt.show()
        print(tf_loss)
       
# display all the different results       
# reconstructed result
result = tf_guess.eval()     
plt.imshow(result, cmap='gray')
plt.show()

# degraded result
plt.imshow(img_noise, cmap='gray')
plt.show()

# original
plt.imshow(obj, cmap='gray')
plt.show()


