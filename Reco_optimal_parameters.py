#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:06:44 2017

@author: bgris
"""

""" reconstruction with optimal parameter of simulated Skull CT data """

import odl
import numpy as np
import adutils
import FOM
# Discretization
reco_space = adutils.get_discretization(use_2D=True)

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, use_2D=True)

# Define fbp
fbp = adutils.get_fbp(A, use_2D=True)

# Data
rhs = adutils.get_data(A, use_2D=True)
#rhs.show('data')
name_reco='/home/bgris/ad-skull-reconstruction-master/data/Simulated/120kV/reference_reconstruction_512_512.npy'
reco=np.load(name_reco).astype('float32')
reco=reco_space.element(reco)
#reco.show()



phantom = reco_space.element(adutils.get_phantom(use_2D=True))
labels = adutils.get_phantom(use_2D=True,get_Flags=True)

#label = phantom.asarray()
#label[label == 1] = 0.0156 #Shift water
#label[label == 2] = 0.0162 #Shift grey matter
#label[label == 3] = 0.0160 #Shift white matter
#label[label == 4] = 0.0401 #Shift bone

#phantom = reco_space.element(label)





# Define fbp
#fbp = adutils.get_fbp(A, use_2D=True)

# Data
rhs = adutils.get_data(A, use_2D=True)

# Reconstruct
#x = fbp(rhs)

# Show result
#x.show()








#phantom.show()

noiseless_data=A(phantom)
#noiseless_data.show()



lam=10
#fbp = odl.tomo.fbp_op(A,padding=False,filter_type='Hamming', frequency_scaling=lam)
# Reconstruct
#x = reco_space.element(fbp(rhs))
#x.show(clim=[0.015,0.02])
#((x-phantom)**2).show()




##%% Rebin
#
#rebin_factor = 10
#
## Discretization
#reco_space = adutils.get_discretization()
#
##Forward operator
#A = adutils.get_ray_trafo(reco_space, use_rebin=True, rebin_factor=rebin_factor)
#
## Data
#rhs = adutils.get_data(A, use_rebin=True, rebin_factor=rebin_factor)


#%% Learning parameter

import scipy


def optimal_parameters(reconstruction, fom, phantoms, data,
                       initial_param=0):
    """Find the optimal parameters for a reconstruction method.
    Parameters
    ----------
    reconstruction : callable
        Function that takes two parameters:
            * data : The data to be reconstructed
            * parameters : Parameters of the reconstruction method
        The function should return the reconstructed image.
    fom : callable
        Function that takes two parameters:
            * reconstructed_image
            * true_image
        and returns a scalar Figure of Merit.
    phantoms : sequence
        True images
    data : sequence
        The data to be reconstructed
    initial_param : array-like
        Initial guess for the parameters
    """
    def func(lam):
        # Function to be minimized by scipy

        value= sum(fom(reconstruction(datai, lam), phantomi)
                   for phantomi, datai in zip(phantoms, data))
        print('value={}, lam={}'.format(value,lam))
        return value
    # Pick resolution to fit the one used by the space
    xtol = ftol = np.finfo(phantom.space.dtype).resolution * 10

    # Use a gradient free method to find the best parameters
    parameters = scipy.optimize.fmin_powell(func, initial_param,
                                            xtol=xtol,
                                            ftol=ftol,
                                            disp=False)
    return parameters

#### For fbp
#def reconstruction(proj_data, lam):
#    lam = np.exp(lam)
#    fbp_op = odl.tomo.fbp_op(A,padding=True,
#                             filter_type='Hann', frequency_scaling=1 / lam)
#    return fbp_op(proj_data)
#
#initial_param = 0.001
#

## For Landweber
#def reconstruction(proj_data, lam):
#    y= A.domain.zero()
#    odl.solvers.iterative.iterative.landweber(A, y,proj_data, 20, omega=lam, projection=None, callback=None)
#    y.show()
#    return y

####for TV
ray_trafo=A
def reconstruction(proj_data, lam):
    lam = float(lam)

    lam=np.exp(lam)
    print('lam = {}'.format(lam))


    # Construct operators and functionals
    gradient = odl.Gradient(phantom.space)
    op = odl.BroadcastOperator(ray_trafo, gradient)

    g = odl.solvers.ZeroFunctional(op.domain)

    l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(proj_data)
    l1_norm = lam * odl.solvers.GroupL1Norm(gradient.range)
    f = odl.solvers.SeparableSum(l2_norm, l1_norm)

    # Select solver parameters
    op_norm = 1.1 * odl.power_method_opnorm(op)

    niter = 1000  # Number of iterations
    tau = 1.0 / op_norm  # Step size for the primal variable
    sigma = 1.0 / op_norm  # Step size for the dual variable
    gamma = 0.3

    # Run the algorithm
    # Initialisation thanks to fbp result
    fbp_op = odl.tomo.fbp_op(A,padding=True,
                             filter_type='Hann', frequency_scaling=0.8)
    x =fbp_op(proj_data)
    odl.solvers.chambolle_pock_solver(
        x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma)

    return x

initial_param = 0.1
#%%
phantoms=[]
phantoms.append(phantom)
rhs_list=[]
rhs_list.append(rhs)

#def fom(I0,I1):
#    gradient_op=odl.Gradient(I0.space)
#    return reco_space.dist(I0,I1) + 100*gradient_op(I0).dist(gradient_op(I1))

#mask=np.zeros(reco.shape,dtype=bool)
#diff=300
#mini=100
#for i in range(diff):
#    for j in range(diff):
#        mask[mini+i][mini+j]=True

mask = ((labels == 3) | (labels == 2))

def fom(I0,I1):
    return 1-FOM.mean_square_error(I0,I1,mask=mask,normalize=True)

# Find optimal lambda
optimal_param = optimal_parameters(reconstruction, fom,
                                        phantoms,rhs_list,
                                        initial_param=initial_param)
#
#%%
optimal_reco=reconstruction(rhs, optimal_param)
optimal_reco.show('tv l2 normOptimal parameter bis',clim=[0.018,0.022])
initial_reco=reconstruction(rhs, initial_param)
initial_reco.show('tv l2 Initial parameter',clim=[0.018,0.022])
#initial_reco.show('test lam=1 tv',clim=[0.012,0.018])
#((optimal_reco-initial_reco)**2).show()