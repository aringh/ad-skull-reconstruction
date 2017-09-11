"""Code from different parts of odl."""

import numpy as np
import scipy
import odl


# Code from the example in odl/examples/solvers/find_optimal_parameters.py
# for finding optimal regularization parameters.
def optimal_parameters(reconstruction, fom, phantoms, data,
                       initial_param=0):
    """Find the optimal parameters for a reconstruction method.

    Notes
    -----
    For a forward operator :math:`A : X \\to Y`, a reconstruction operator
    parametrized by :math:`\\theta` is some operator
    :math:`R_\\theta : Y \\to X`
    such that

    .. math::
        R_\\theta(A(x)) \\approx x.

    The optimal choice of :math:`\\theta` is given by

      .. math::
        \\theta = \\arg\\min_\\theta fom(R(A(x) + noise), x)
    where :math:`fom : X \\times X \\to \mathbb{R}` is a figure of merit.

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
        and returns a scalar figure of merit.
    phantoms : sequence
        True images.
    data : sequence
        The data to reconstruct from.
    initial_param : array-like
        Initial guess for the parameters.
    Returns
    -------
    parameters : 'numpy.ndarray'
        The  optimal parameters for the reconstruction problem.
    """

    def func(lam):
        # Function to be minimized by scipy
        return sum(fom(reconstruction(datai, lam), phantomi)
                   for phantomi, datai in zip(phantoms, data))

    # Pick resolution to fit the one used by the space
    tol = np.finfo(phantoms[0].space.dtype).resolution * 10

    initial_param = np.asarray(initial_param)

    # We use a faster optimizer for the one parameter case
    if initial_param.size == 1:
        bracket = [initial_param - tol, initial_param + tol]
        result = scipy.optimize.minimize_scalar(func,
                                                bracket=bracket,
                                                tol=tol,
                                                bounds=None,
                                                options={'disp': False})
        return result.x
    else:
        # Use a gradient free method to find the best parameters
        parameters = scipy.optimize.fmin_powell(func, initial_param,
                                                xtol=tol,
                                                ftol=tol,
                                                disp=False)
        return parameters


# The MSE fom
def mean_squared_error(data, ground_truth, mask=None, normalized=False):
    """Return L2-distance between ``data`` and ``ground_truth``.
    Evaluates `mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between
    input (``data``) and reference (``ground_truth``) Allows for normalization
    (``normalized``) and a masking of the two spaces (``mask``).
    Notes
    ----------
    The FOM evaluates
    .. math::
        \| f - g \|^2_2,
    or, in normalized form
    .. math::
        \\frac{\| f - g \|^2_2}{\| f \|^2_2 + \| g \|^2_2}.
    The normalized FOM takes values in [0, 1].
    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`, optional
        Mask to define ROI in which FOM evaluation is performed. The mask is
        allowed to be weighted (i.e. non-binary), see ``blurring`` and
        ``false_structures.``
    normalized  : bool, optional
        Boolean flag to switch between unormalized and normalized FOM.
    Returns
    -------
    fom : float
        Scalar (float) indicating mean squared error between ``data`` and
        ``ground_truth``. In normalized form the FOM takes values in
        [0, 1], with higher correspondance at lower FOM value.
    """
    l2_normSquared = odl.solvers.L2NormSquared(data.space)

    if mask is not None:
        data = data * mask
        ground_truth = ground_truth * mask

    diff = data - ground_truth
    fom = l2_normSquared(diff)

    if normalized:
            fom /= (l2_normSquared(data) + l2_normSquared(ground_truth))

    return fom
