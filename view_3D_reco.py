"""
TV reconstruction example for simulated Skull CT data
"""

import numpy as np
import adutils

# Discretization
reco_space = adutils.get_discretization()

reco = np.load('/home/aringh/git/ad-skull-reconstruction/data/Simulated/' +
               '120kV/reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Huber.npy')
reco_odl = reco_space.element(reco)

reco_odl.show(clim=[0.018, 0.022])
