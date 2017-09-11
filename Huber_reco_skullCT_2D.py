"""
TV reconstruction example for simulated Skull CT data
"""

import odl
import odl.contrib.fom as fom
import numpy as np
import os
import adutils

from Huber_func import HuberNorm
from opt_param import optimal_parameters

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


def huber_reconstruction(proj_data, lam):
    """Defines the Huber norm reconstruction."""
    # Transform to make reg.param. always positive
    lam = float(lam)
    lam = np.exp(lam)
    print('lam = {}'.format(lam))

    g = odl.solvers.ZeroFunctional(op.domain)

    l2_norm = odl.solvers.L2NormSquared(A.range).translated(proj_data)
    l1_norm = lam * HuberNorm(space=gradient.range, epsilon=huber_epsilon)
    f = odl.solvers.SeparableSum(l2_norm, l1_norm)

    # Select solver parameters
    niter = 1000  # Number of iterations
    tau = 1.0 / op_norm  # Step size for the primal variable
    sigma = 1.0 / op_norm  # Step size for the dual variable
    gamma = 0.3

    # Run the algorithm - initialize with FBP
    x = adutils.get_initial_guess(reco_space)
    odl.solvers.chambolle_pock_solver(
        x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma)

    return x

initial_param = 0.1

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
    return fom.mean_square_error(I0, I1, mask=mask, normalize=True)

# Find optimal lambda
optimal_param = optimal_parameters(huber_reconstruction, my_fom, phantoms,
                                   rhs_list, initial_param=initial_param)

# Make a reconstruction with the optimal parameter
x_reco = huber_reconstruction(rhs, optimal_param)


# Run such that last iteration is saved (saveReco = 1) or none (saveReco = 0)
saveReco = False
savePath = '/home/aringh/git/ad-skull-reconstruction/data/Simulated/120kV/'

if saveReco:
    saveName = os.path.join(savePath,'reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Huber_2D.npy')
    adutils.save_image(x_reco, saveName)
