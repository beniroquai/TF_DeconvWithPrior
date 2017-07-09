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


img_noise,img,otf,obj = simn.SimNoiseImg('./devoogd.tif');


guess = img_noise*0+np.mean(img_noise)
mylambda=0.1

#totalError, myDeriv = simn.PoissonErrorAndDeriv(guess,img_noise,otf,mylambda)


# typeconversion from python to tensorflow - all cmplx values correct?!
tf_otf = tf.constant(otf, dtype=tf.float32, name = "tf_otf")
tf_measured = tf.constant(img_noise, dtype = tf.float32, name = "tf_measured")
tf_guess = tf.Variable(guess, dtype= tf.float32, name = "tf_guess")
tf_mylambda = tf.constant(mylambda, dtype = tf.float32, name = "tf_mylambda")


# define error metric
totalError, myFWD = simn.PoissonErrorAndDerivTF(tf_guess,tf_measured,tf_otf,tf_mylambda)


# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(totalError)

# Initializing the variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# start inference using Tensorflow and its autodifferenciation asöldkfjadflsöjk
NEPOCH = 100
for i in range(NEPOCH):
    sess.run(train_step)
    
    print(tf_loss)
    if(np.mod(i, 10)):
        tf_loss = sess.run(totalError)
        result = tf_guess.eval()     
        plt.imshow(result, cmap='gray')
        plt.show()

       
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


