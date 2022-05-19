#script containing classes for minimisation
#also custom plotting functions

#imports
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#function calculates normalised error on histogram
def hist_norm(data,bin_no):

    #unnormalised histogram
    y1, bins, patches1 = plt.hist(data, bin_no)
    plt.clf()
    #bin centres 
    bin_centres = bins[:-1] + (np.diff(bins)/2)

    x = bin_centres
    #error on unormalised hist bins
    ye = np.sqrt(y1)

    #normalised hist
    y,bins,patches = plt.hist(data,bin_no,density = 1)
    plt.clf()
    #error on normalised hist
    ye = ye*y/y1
    
    #return values for minuit binned cost func
    return x,y,ye


class Fit_Mass:

    #initialisation method
    def __init__(self,lox,hix):
        #limits of spectrum for integration
        self.lox = lox
        self.hix = hix

    def Line(self,x,m):
        #pivot in centre of mass range
        #height of pivot as normalisation
        pivot = 1/(self.hix-self.lox)
        xmid = (self.hix-self.lox)/2
        return (m*(x-xmid)) + pivot

    def Gaus(self,x,mu,sig):
        #normalised gaussian
        y = lambda t: np.exp(- ((t-mu)**2)/(2*(sig**2)))
        #numericle integration for normalisation
        I,err = integrate.quad(y, self.lox, self.hix)
        N = 1/I

        f = np.exp(- ((x-mu)**2)/(2*(sig**2)))
        return N * f

    def CB_norm(self,x,mu,sig,alpha,n,A,B):
        #method returns normalisation for crystal ball
        #conditional statement for which tail contains exponential portion
        if sig > 0:
            #integration of gaussian portion
            y1 = lambda t: np.exp(- ((t-mu)**2)/(2*(sig**2)))
            I1,err = integrate.quad(y1, mu-(alpha*sig), self.hix)
            #exp portion
            y2 = lambda t: A * ((B - ((t-mu)/sig))**(-n))
            I2,err = integrate.quad(y2,self.lox,mu-(alpha*sig))
            
            #summs and returns normalisation constant for given params
            I = I1 + I2
            return 1/I

        elif sig < 0:
            #same as above but with limits flipped
            y1 = lambda t: np.exp(- ((t-mu)**2)/(2*(sig**2)))
            I1,err = integrate.quad(y1, self.lox,mu-(alpha*sig))

            y2 = lambda t: A * ((B - ((t-mu)/sig))**(-n))
            I2,err = integrate.quad(y2, mu-(alpha*sig), self.hix)

            I = I1 + I2
            return 1/I


    def CB(self,x,mu,sig,alpha,n):
        #normalised Crystal Ball function
        #CB parameters for exponential
        A = ((n/np.abs(alpha))**n) * np.exp(- ((np.abs(alpha))**2)/2)
        B = (n/np.abs(alpha)) - np.abs(alpha)
        
        #test for exp or gauss
        X = (x-mu)/sig
        f = []

        for i in range(len(X)):
            #xval lies in gaussian portion
            if X[i] > -alpha:
                f.append(np.exp(-0.5 * (X[i]**2)))
            #exp portion
            else:
                f.append(A *((B - (X[i]))**(-n)))
        #retrive normalisiation constant
        N = self.CB_norm(x,mu,sig,alpha,n,A,B)
        #return normed CB
        return N * np.array(f)

    def model(self,x,m,mu1,sig1,alpha1,n1,mu2,sig2,alpha2,n2,mu3,sig3,F1,F2,F3):
        #combined normalised spectrum
        #signal as 2 Crystal balls and a gaussian
        cb = self.CB(x,mu1,sig1,alpha1,n1)
        cb2 = self.CB(x,mu2,sig2,alpha2,n2)
        gaus = self.Gaus(x,mu3,sig3)
        #background as a line
        line = self.Line(x,m)
        #fractional combination of normalised contributions
        #returns normalised spectrum
        return  (F1 * cb) + (F2*cb2) + (F3*gaus) + ((1-F1-F2-F3)* line)

#class for delta M spectrum minimisation
#Same as above only with different background and addistional initialised param for mass threshold
#really should be combined into one class
class Fit_dM:
    def __init__(self,m0,lox,hix):
        self.m0 = m0
        self.lox = lox
        self.hix = hix

    def Power(self,x,a,b,k):
        #normalised power law as background
        #lifted from root documentation - RooDstD0BG
        y = lambda t:  ((1-np.exp(-(t-self.m0)/k)) * (((t/self.m0)**a))) + (b*((t/self.m0)-1))
        norm,err = integrate.quad(y, self.lox, self.hix)
        n = 1/norm
        return  n * ((1-np.exp(-(x-self.m0)/k)) * (((x/self.m0)**a))) + (b*((x/self.m0)-1))

    def Gaus(self,x,mu,sig):
        y = lambda t:  np.exp(-0.5 * ((t-mu)**2)/(sig**2))
        norm,err = integrate.quad(y, self.lox, self.hix)
        n = 1/norm
        return  n * np.exp(-0.5 * ((x-mu)**2)/(sig**2))

    def CB_norm(self,x,mu,sig,alpha,n,A,B):

        if sig > 0:
            y1 = lambda t: np.exp(- ((t-mu)**2)/(2*(sig**2)))
            I1,err = integrate.quad(y1, mu-(alpha*sig), self.hix)

            y2 = lambda t: A * ((B - ((t-mu)/sig))**(-n))
            I2,err = integrate.quad(y2,self.lox,mu-(alpha*sig))

            I = I1 + I2
            return 1/I

        elif sig < 0:
            y1 = lambda t: np.exp(- ((t-mu)**2)/(2*(sig**2)))
            I1,err = integrate.quad(y1, self.lox,mu-(alpha*sig))

            y2 = lambda t: A * ((B - ((t-mu)/sig))**(-n))
            I2,err = integrate.quad(y2, mu-(alpha*sig), self.hix)

            I = I1 + I2
            return 1/I

    def CB(self,x,mu,sig,alpha,n):
        A = ((n/np.abs(alpha))**n) * np.exp(- ((np.abs(alpha))**2)/2)
        B = (n/np.abs(alpha)) - np.abs(alpha)

        X = (x-mu)/sig
        f = []

        for i in range(len(X)):

            if X[i] > -alpha:
                f.append(np.exp(-0.5 * (X[i]**2)))
            else:
                f.append(A *((B - (X[i]))**(-n)))

        N = self.CB_norm(x,mu,sig,alpha,n,A,B)
        return N * np.array(f)

    def model_rs(self,x,a,b,k,mu1,sig1,alpha1,n1,F1,mu2,sig2,alpha2,n2,F2,mu,sig,F,mu3,sig3,F3):
        #model for deltaM RS sample
        #background as power law
        p = self.Power(x,a,b,k)
        #signal as 2 crystal ball funcs and 2 gaussians
        cb1 = self.CB(x,mu1,sig1,alpha1,n1)
        cb2 = self.CB(x,mu2,sig2,alpha2,n2)
        g = self.Gaus(x,mu,sig)
        g2 = self.Gaus(x,mu3,sig3)
        #fractional combinations of PDFs
        return ((1-F1-F-F2-F3) * p) + (F1 *cb1) + (F2 *cb2) + (F*g) + (F3*g2)

    def model_ws(self,x,a,b,k,mu1,sig1,alpha1,n1,F1,mu2,sig2,alpha2,n2,F2,mu3,sig3,F3):
        #model for deltaM WS sample
        #background as power law
        p = self.Power(x,a,b,k)
        #signal as 2 crystal ball functions and additional gaussian
        cb1 = self.CB(x,mu1,sig1,alpha1,n1)
        cb2 = self.CB(x,mu2,sig2,alpha2,n2)
        g = self.Gaus(x,mu3,sig3)
        #fractional combinations of PDFs
        return ((1-F1-F3-F2) * p) + (F1 *cb1) + (F2 *cb2) + (F3*g)

#plot function for residual pulls
def error_plot(bin_counts,bin_centres,yvals,ax2):
    #calc pull
    residual = (bin_counts - yvals)/np.sqrt(bin_counts)
    #plot in pannel as scatter
    ax2.scatter(bin_centres,residual,s = 2)
    ax2.grid()

    return residual
#creating axis' for plots with pulls in panel below
def pannel_plot_set():
    #plot dimensions
    left = 0.1
    width = 0.8
    plt.figure(figsize= (16,12))
    ax1 = plt.axes([left, 0.4, width, 0.45])
    ax2 = plt.axes([left, 0.2, width, 0.19],sharex = ax1)
    
    #ticks
    ax1.xaxis.tick_top()
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax2.xaxis.tick_bottom()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    
    #returns axis
    return ax1,ax2

#example function to get yvals for plotting
#not: xdata_ws is mass values and x_ws is bin centers
def get_yvals_WS(fit,M_ws,xdata_ws,x_ws,bin_edges):
    #chisq/ndf
    Chisq_ndf_ws = M_ws.fval/M_ws.ndof
    #renormalistion = number of datapoints * bin width
    renorm = len(xdata_ws)*np.mean(np.diff(bin_edges))
    #yvals for spectrum
    yvals_ws = fit.model_ws(x_ws,*M_ws.values)*renorm
    
    #yvals for signal and background
    #have to consider fractions
    yvals_bgd_ws = fit.Power(x_ws,*M_ws.values[0:3])*renorm*(1-M_ws.values['F1']-M_ws.values['F3']-M_ws.values['F2'])
    yvals_sig1_ws = fit.CB(x_ws,M_ws.values['mu1'],M_ws.values['sig1'],M_ws.values['alpha1'],M_ws.values['n1'])*renorm*M_ws.values['F1']
    yvals_sig2_ws = fit.CB(x_ws,M_ws.values['mu2'],M_ws.values['sig2'],M_ws.values['alpha2'],M_ws.values['n2'])*renorm*M_ws.values['F2']
    yvals_sig3_ws = fit.Gaus(x_ws,M_ws.values['mu3'],M_ws.values['sig3'])*renorm*M_ws.values['F3']
    yvals_sig_ws = yvals_sig1_ws + yvals_sig2_ws + yvals_sig3_ws

    return Chisq_ndf_ws,yvals_ws,yvals_bgd_ws,yvals_sig_ws
