multiplegp
==========

Code for the paper "Collaborative multi-output Gaussian processes". 
Collaborative multi-output Gaussian processes (COGP) is the first scalable multi-output GPs model capable of dealing with very large number of inputs and outputs (big data, if you will).

If you use the code or data we provide, please acknowledge by citing our paper:
Trung V. Nguyen and Edwin V. Bonilla, Collaborative multi-output Gaussian processes, in UAI 2014

This documentation is quite rudimentary and will be updated for more details when time allows.

Contents

- data : 3 datasets (fx : foreign exchange rates, weather : air temperature, sarcos : robot inverse dynamics)
- libs : dependent libraries for the code
- note and paper : latex source of the paper (note is for the earlier version -- can be ignored)
- results : contain figures in the paper
- src: the main code directory

Code structure (src directory)

- datautils: utilities for reading and processing data
- gpsvi : old implementation of Stochastic variational inference for Gaussian processes (Hensman's UAI paper) -- replaced by svi_*.m files
- test : gradient checks for the optimisation code
- tmp: old content -- can be ignored
- scripts : scripts used to run the experiments

How to use

The demo script src/scripts/demo_slfm.m contains an example.
Once the data structure is set up properly (see the script file), slfm_learn() can be used to train a model and slfm_predict() is used to make prediction.

