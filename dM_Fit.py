#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
from iminuit import Minuit,cost
import Fit_utils

#fitting range for DeltaM
low_mass_lim = 142
high_mass_lim = 152.5
#load in file
file_ws = uproot.open("/eos/lhcb/user/j/jcobbled/K3Pi/skimmed_full_production_tuples/refined_selections/WS_Data/ws_no_clone_multiple_candidates_file_for_ned.root:DecayTree")
#DeltaM and Mass values loaded into np
xdata_ws = file_ws["deltam_ReFit"].array(library = 'np')
xdata_ws_D0 = file_ws["Dst_ReFit_D0_M_best"].array(library = 'np')
#binning scheme
bin_no = (high_mass_lim - low_mass_lim)*100
bin_edges = np.linspace(low_mass_lim,high_mass_lim,int(bin_no + 1))

#mass cut for delta mass to remove some combinatorial D0
xdata_ws = xdata_ws[(xdata_ws_D0 <= 1890) & (xdata_ws_D0 >= 1840)]
#mass window for fitting and plotting
xdata_ws = xdata_ws[(xdata_ws <= high_mass_lim) & (xdata_ws >= low_mass_lim)]

#initialising fitting class with fit range and pion mass as power law threshold
fit = Fit_utils.Fit_dM(139.57039,low_mass_lim,high_mass_lim)
#cost function and error on normalised deltaM hist
x_ws,y_ws,ye_ws = Fit_utils.hist_norm(xdata_ws,int(bin_no))
c_ws = cost.LeastSquares(x_ws, y_ws, ye_ws, fit.model_ws)

#params for WS minimisation
#a,b,k are power law params
#mu, sig - mean value and width for gaussian
#alpha, n - params for crystal ball - cuttoff and exp slope
#F - fractional contribution of signal componants
#params with 1 & 2 for crystal balls and 3 for pure gaussian
paramNames_ws=['a','b','k','mu1','sig1','alpha1','n1','F1','mu2','sig2','alpha2','n2','F2','mu3','sig3','F3']
#Set the start values
startvals_ws = {paramNames_ws[0]:5.4,
                paramNames_ws[1]:0,
                paramNames_ws[2]:1.7,
                paramNames_ws[3]:145.5,
                paramNames_ws[4]:-0.25,
                paramNames_ws[5]:0.5,
                paramNames_ws[6]:10,
                paramNames_ws[7]:0.025,
                paramNames_ws[8]:145.6,
                paramNames_ws[9]:0.1,
                paramNames_ws[10]:1.8,
                paramNames_ws[11]:8,
                paramNames_ws[12]:0.03,
                paramNames_ws[13]:145.4,
                paramNames_ws[14]:0.3,
                paramNames_ws[15]:0.025}

#minimiser object
M_ws = Minuit(c_ws, **startvals_ws)
M_ws.tol = 10e-12
#limits and fixing for stable fits
#only nescessary for crystal ball params n and alpha 
M_ws.limits['a'] = (-15,15)
M_ws.limits['b'] = (-15,15)
M_ws.limits['k'] = (0,30)
M_ws.limits['mu1','mu2','mu3'] = (144.5,146)
M_ws.limits['sig2','sig3'] = (0.1,1)
M_ws.limits['sig1'] = (-1,0)
M_ws.fixed['alpha1'] = 0.5
M_ws.fixed['alpha2'] = 1.8
M_ws.limits['n1','n2'] = (2,30)
M_ws.limits['F2','F1','F3'] = (0,0.3)
result_ws = M_ws.migrad(ncall = 20000)
#display minimisation results
print(result_ws)

#-------------------------------
#plotting
#y_vals to plot un-normalised mass spectrum overlayed on histogram
#C2 is chisq/ndf
C2,y_ws,y_bck,y_sig = Fit_utils.get_yvals_WS(fit,M_ws,xdata_ws,x_ws,bin_edges)

#axis for plotting- includes pannes for residual pulls
ax1,ax2 = Fit_utils.pannel_plot_set()
#hhistogram
n,bins,patches = ax1.hist(xdata_ws,bins = bin_edges,density = 0,label = None)
#calculate and plot pulls on ax2 (pannel)
Pull = Fit_utils.error_plot(n,x_ws,y_ws,ax2)

#plotting calculated spectrum from fit
#full spectrum
ax1.plot(x_ws,y_ws,linewidth = 2, c = 'r',label = 'deltaM Spectrum - Chisq/ndf: ' + str(np.round(C2,4)))
#background
ax1.plot(x_ws,y_bck,'--', linewidth = 2, c = 'black',label = 'Background')
#signal
ax1.plot(x_ws,y_sig, linewidth = 2, c = 'black',label = 'Signal')
ax1.legend(fontsize = 15)
ax2.set_xlabel(r'$\Delta$M  $[MeV]$',fontsize = 20)
ax2.set_ylabel('Pull',fontsize = 20)
ax1.set_ylabel('Counts per 0.01MeV',fontsize = 20)

plt.savefig('Delta_M_min.png')
plt.clf()
