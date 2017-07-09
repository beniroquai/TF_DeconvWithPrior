# TF_DeconvWithPrior
A basic framework to use bayesian loglikelihood estimation for image restoration using Tensorflow's automated differentation. 

Use the file DeconvWithPrior.py to simulate a distorted image and restorate it using the error metric in a maximum log-likelihood manner.

This works using Tensorflow 1.2 rc 2.0 on a MAC where I baked the FFT section into the current build. One can find a detailed description of the algorithm i.e. in this publication: 


Rainer Heintzmann, Estimating missing information by maximum likelihood deconvolution, Micron, Volume 38, Issue 2, 2007, Pages 136-144, ISSN 0968-4328, http://dx.doi.org/10.1016/j.micron.2006.07.009.
(http://www.sciencedirect.com/science/article/pii/S0968432806001272)
