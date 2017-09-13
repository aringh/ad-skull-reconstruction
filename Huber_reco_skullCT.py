"""
TV reconstruction example for simulated Skull CT data
"""

import odl
import numpy as np
import os
import adutils

savePath = '/home/aringh/git/ad-skull-reconstruction/data/Simulated/120kV/reco/'
print('The save path exists = {}'.format(os.path.exists(savePath)))

# Discretization
reco_space = adutils.get_discretization()

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space)

# Data
rhs = adutils.get_data(A)

# Gradient operator
gradient = odl.Gradient(reco_space, method='forward')

# Column vector of operators
op = odl.BroadcastOperator(A, gradient)

Anorm = odl.power_method_opnorm(A[1], maxiter=2)
Dnorm = odl.power_method_opnorm(gradient,
                                xstart=odl.phantom.white_noise(gradient.domain),
                                maxiter=10)

# Estimated operator norm, add 10 percent
op_norm = 1.1 * np.sqrt(len(A.operators)*(Anorm**2) + Dnorm**2)

print('Norm of the product space operator: {}'.format(op_norm))


# Define the reoncstruction procedure
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
    callback = odl.solvers.CallbackPrintIteration()
    odl.solvers.bfgs_method(
        obj_fun, x, maxiter=maxiter, num_store=num_store,
        hessinv_estimate=odl.ScalingOperator(
                reco_space, 1 / odl.power_method_opnorm(A) ** 2),
        callback=callback)

    return x


# Optimally selected parameters from 2D-code. Observe the transformation in the
# reconstruction algorithm
lamb = -2.23695027
huber_epsilon = -2.99999541
param = [lamb, huber_epsilon]

# Do the reconstruction
x_reco = huber_reconstruction(rhs, param)

saveReco = True
# savePath = '/home/aringh/git/ad-skull-reconstruction/data/Simulated/120kV/reco/'

if saveReco:
    saveName = os.path.join(savePath,'Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Huber')
    adutils.save_image(x_reco, saveName)
