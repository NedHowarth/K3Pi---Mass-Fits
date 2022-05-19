#imports
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit,cost
import uproot
import Fit_utils

#load in data to numpy 
file = uproot.open("/eos/lhcb/user/j/jcobbled/K3Pi/skimmed_full_production_tuples/refined_selections/WS_Data/ws_no_clone_multiple_candidates_file_for_ned.root:DecayTree")
#mass data
D0_M = file["Dst_ReFit_D0_M_best"].array(library = 'np')

#fitting limits and creating mass window
high_mass_lim = 1935
low_mass_lim = 1810
xdata = D0_M[(D0_M >= low_mass_lim) & (D0_M <= high_mass_lim)]

#binning scheme
bin_no = (high_mass_lim - low_mass_lim)*10
bin_edges = np.linspace(low_mass_lim,high_mass_lim,bin_no + 1)

#initialising fitting class
fit = Fit_utils.Fit_Mass(low_mass_lim,high_mass_lim)
#cost function - binned fit requires; x - bin centre, y- normalised bin height/count, ye- error on bin count
#x,y,ye calculated in custom function, ye is poisson error
x,y,ye = Fit_utils.hist_norm(xdata,bin_no)
c = cost.LeastSquares(x,y,ye, fit.model)

#param names and setting start values 
#m - gradient of linear background
#mu, sig - mean value and width for gaussian
#alpha, n - params for crystal ball - cuttoff and exp slope
#F - fractional contribution of signal componants
#params with 1 & 2 for crystal balls and 3 for pure gaussian
#start vals are arbitrary changing them within reason should not affect minimisation
paramNames=['m','mu1','sig1','alpha1','n1','mu2','sig2','alpha2','n2','mu3','sig3','F1','F2','F3']
startvals = {paramNames[0]:0,
             paramNames[1]:1865,
             paramNames[2]:4,
             paramNames[3]:1.2,
             paramNames[4]:1.4,
             paramNames[5]:1864,
             paramNames[6]:-6,
             paramNames[7]:2.5,
             paramNames[8]:1.8,
             paramNames[9]:1865,
             paramNames[10]:8,
             paramNames[11]:0.25,
             paramNames[12]:0.30,
             paramNames[13]:0.25}

#creating minimiser object
M = Minuit(c, **startvals)
#limits for crystal ball functions 
#one width has to be -ve and one +ve to aproximate both tails
M.limits['sig1'] = (0,10)
M.limits['n1','n2'] = (0,10)
M.limits['sig2'] = (-10,0)
#small tolerance- ncreases stability
M.tol = 0.000001
#runing minimisation and displaying output
result = M.migrad(ncall = 10000)
print(result)
