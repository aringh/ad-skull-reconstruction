"""
TV reconstruction example for simulated Skull CT data
"""

import odl
import numpy as np
import os
import adutils

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

lamb = 0.005  # l2NormGrad/l1NormGrad = 0.01

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(A.range).translated(rhs)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = lamb * odl.solvers.L1Norm(gradient.range)

# Combine functionals
f = odl.solvers.SeparableSum(l2_norm, l1_norm)

# Set g functional to zero
g = odl.solvers.ZeroFunctional(op.domain)

# Accelerataion parameter
gamma = 0.4

# Step size for the proximal operator for the primal variable x
tau = 1.0 / op_norm

# Step size for the proximal operator for the dual variable y
sigma = 1.0 / op_norm  # 1.0 / (op_norm ** 2 * tau)

# Reconstruct
callbackShowReco = (odl.solvers.CallbackPrintIteration() &  # Print iterations
                    odl.solvers.CallbackShow(coords=[None, 0, None]) &  # Show parital reconstructions
                    odl.solvers.CallbackShow(coords=[0, None, None]) &
                    odl.solvers.CallbackShow(coords=[None, None, 60]))

callbackPrintIter = odl.solvers.CallbackPrintIteration()

# Use initial guess
x = adutils.get_initial_guess(reco_space)

# Run such that last iteration is saved (saveReco = 1) or none (saveReco = 0)
saveReco = False
savePath = '/home/user/Simulated/120kV/'
niter = 100
odl.solvers.chambolle_pock_solver(x, f, g, op, tau=tau, sigma = sigma,
				  niter = niter, gamma=gamma, callback=callbackPrintIter)
if saveReco:
    saveName = os.path.join(savePath,'reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_TV_' +
                                          str(niter) + 'iterations.npy')
    adutils.save_image(x, saveName)
