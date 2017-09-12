"""
TV reconstruction example for simulated Skull CT data
"""

import odl
# import odl.contrib.fom as fom
import numpy as np
import os
import adutils

from opt_param import optimal_parameters, mean_squared_error

# Discretization
reco_space = adutils.get_discretization(use_2D=True)

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, use_2D=True)

# Get data and phantom
rhs = adutils.get_data(A, use_2D=True)
phantom = reco_space.element(adutils.get_phantom(use_2D=True))

# Gradient operator
gradient = odl.Gradient(reco_space, method='forward')

# Column vector of operators
op = odl.BroadcastOperator(A, gradient)

Anorm = odl.power_method_opnorm(A, maxiter=2)
Dnorm = odl.power_method_opnorm(gradient,
                                xstart=odl.phantom.white_noise(gradient.domain),
                                maxiter=10)

# Estimated operator norm, add 10 percent
op_norm = 1.1 * np.sqrt(Anorm**2 + Dnorm**2)

print('Norm of the product space operator: {}'.format(op_norm))


# Defining the huber norm reco. Select value of smoothness via epsilon
huber_epsilon = 0.01


def huber_reconstruction(proj_data, parameters):
    # Extract the separate parameters
    lam, sigma = parameters
    lam=np.exp(lam)
    sigma=np.exp(sigma)
    print('lam = {}, sigma = {}'.format(lam, sigma))

    # We do not allow negative parameters, so return a bogus result
    if lam <= 0 or sigma <= 0:
        return np.inf * A.range.one()

    # Create data term ||Ax - b||_2^2
    l2_norm = odl.solvers.L2NormSquared(A.range)
    data_discrepancy = l2_norm * (A - proj_data)

    # Create regularizing functional huber(|grad(x)|)
    l1_norm = odl.solvers.GroupL1Norm(gradient.range)
    smoothed_l1 = odl.solvers.MoreauEnvelope(l1_norm, sigma=sigma)
    regularizer = smoothed_l1 * gradient

    # Create full objective functional
    obj_fun = data_discrepancy + lam * regularizer

    # Pick parameters
    maxiter = 100
    num_store = 5

    # Run the algorithm - initialize with FBP
    x = adutils.get_initial_guess(reco_space)
    odl.solvers.bfgs_method(
        obj_fun, x, maxiter=maxiter, num_store=num_store,
        hessinv_estimate=odl.ScalingOperator(
                reco_space, 1 / odl.power_method_opnorm(A) ** 2))

    return x


initial_param = [-2, -3]

# Data to train on
phantoms = []
phantoms.append(phantom)
rhs_list = []
rhs_list.append(rhs)

# Create a mask for the fom
labels = adutils.get_phantom(use_2D=True, get_Flags=True)
mask = ((labels == 3) | (labels == 2))


def my_fom(I0, I1):
    """ I0 is ground truth, I1 is reconstruction."""
    return mean_squared_error(I0, I1, mask=mask, normalized=True)

# Find optimal lambda
optimal_param = optimal_parameters(huber_reconstruction, my_fom, phantoms,
                                   rhs_list, initial_param=initial_param)

# Make a reconstruction with the optimal parameter
x_reco = huber_reconstruction(rhs, optimal_param)
x_reco.show(clim=[0.018, 0.022])

# Run such that last iteration is saved (saveReco = 1) or none (saveReco = 0)
saveReco = False
savePath = '/home/aringh/git/ad-skull-reconstruction/data/Simulated/120kV/'

if saveReco:
    saveName = os.path.join(savePath,'reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Huber_2D.npy')
    adutils.save_image(x_reco, saveName)
