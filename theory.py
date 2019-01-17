import numpy as np
from numpy import log, exp
np.seterr(over='ignore', invalid='ignore', divide='ignore')
from scipy.special import gammainc, gamma
import functools
import pandas as pd
from numpy.random import exponential
class first:
    Ud = 1e-8*700
    Up = 1e-8*5e6
    sd = 0.1
    sp = 1e-3
    N_0 = 1e3
    t_max = 10000

    def __init__(self, **params):
        self.__dict__.update(params)

    @property
    def f(self): return self.Ud*self.sd/(1 + self.sd)

    @property
    def dN(self): return self.sd

    @property
    def vp(self): return self.sp*self.Up

    @property
    def N_eq(self): 
        assert self.vp > 0, 'vp: {x.vp:g}, Up: {x.Up:g}, sp: {x.sp:g}, N: {x.N_0}'.format(x = self )
        assert self.f > 0,  ' f: {x.f:g} , Up: {x.Up:g}, sp: {x.sp:g}, N: {x.N_0}'.format(x = self )
        assert self.dN > 0, 'dN: {x.dN:g}, Up: {x.Up:g}, sp: {x.sp:g}, N: {x.N_0}'.format(x = self )
        return self.vp/(self.f*log(1 + self.dN))
    
    def __repr__(self):
        return """Ud =    {x.Ud:g}
Up =    {x.Up:g}
sd =    {x.sd:g}
sp =    {x.sp:g}
N_0 =   {x.N_0:g}
t_max = {x.t_max:g}
N_eq =  {x.N_eq:g}
""".format(x=self)

    def velocity(model, x): return x*model.N_eq*model.vp*(x - 1)

    def potential(model, x): return model.N_eq*model.vp*x*x*(0.5 - x/3)

    def pC(model, x=None):
        D = 2/log(1+model.dN)
        if x is None:
            x = model.N_0/model.N_eq
        return 1 - gammainc(D, D/x) 

def clonalInterference(model):
    """ CI -> d<d>/dt : sp = 0, Ud ~ sd"""
    from scipy.optimize import fsolve
    sd = model.sd
    def f(V):
        A = V/model.Ud; B = log(A);
        return model.N - (exp(V/2/sd*((B-1)**2+1)-0.5*(log(sd**3/(A*B))- B if V>=sd else 0) ) if V>sd/B else 0.5*A/sd)
    return fsolve(f, sd)

@functools.lru_cache(maxsize=None)
def fAnddN(Up, sd, sp, N):
    from scipy.stats import poisson
    L = Up/sp
    i_max = int(np.ceil(log(1+sd)/log(1+sp)))
    k = (poisson.cdf(np.arange(L), L) < 0.5*pow(N,-0.5)).sum()
    X = np.arange(i_max)
    W = (1+sd)*(1+sp)**-X
    f = poisson.pmf(X+k, L)/(1-poisson.cdf(k-1, L))
    fP = f*(W-1)/W
    dNp = np.empty(i_max)
        
    Lambda = fP.sum()
    assert Lambda < sd/(1+sd), 'wtf!'
    
    dNp[i_max - 1] = i_max - 1
    for i in range(i_max-2, -1, -1): dNp[i] = (i + sp*dNp[i+1])/(1 + sp)
    V = (1 + sd)*(1 + sp)**-dNp
    assert (V <= 1+sd).all()
    return Lambda, fP.dot(V)/Lambda - 1

def moran(Up, sp, N):
    x0 = exp(-Up/sp)
    spp = sp/(np.e-1)
    Tclick = exp(0.5*spp*N*x0)/spp
    return Up*1/Tclick

def wave(Up, sp, N, v_0=1e-12):
    from scipy.optimize import brentq
    x2 = sp*log(N)
    if Up < x2:
        No = N*exp(-Up/sp)
        pi_p = sp/((1 + sp)**No - 1) 
        assert No*Up*pi_p > 0
        return  No*Up*pi_p 
    
    iL = sp/Up
    Nd2spSqrtiL = N/2*sp*np.sqrt(iL)
    fiveSixthsiL = 5*iL/6
    def f(v):
        A = log(np.e*Up/v)
        C = v/Up 
        return 1 - 0.5*C*(A*A + 1) - iL*log(Nd2spSqrtiL*np.sqrt(v*C*C/(Up - v))*A/(1 - C*A + fiveSixthsiL))
    vp = brentq(f, v_0, Up - v_0)
    assert vp > 0
    return min(vp, Up)

@functools.lru_cache(maxsize=None)
def Vp(Up, sp, N):
    L = Up/sp
    if L > 1: 
        return wave(Up, sp, N)
    N_0 = N*exp(-L)
    return moran(Up, sp, N) if sp*N_0 > 1 else (1-L)*moran(Up, sp, N) + L*wave(Up, sp, N)

class second(first):
    @property
    def f(self): 
        return fAnddN(self.Up, self.sd, self.sp, self.N_0)[0]*self.Ud

    @property
    def dN(self): return fAnddN(self.Up, self.sd, self.sp, self.N_0)[1]

    @property
    def vp(self): 
        return self.sp*Vp(self.Up, self.sp, self.N_0)

def safe_integrate(func, a, b, **kwargs):
    from scipy.integrate import quad, romberg
    y, err, *output = quad(func, a, b, full_output=1, **kwargs)
    out = romberg(func, a, b, **kwargs) if len(output) == 2 else y
    #print(out, func(a), func(b), func.__name__)
    return out

def mean_time(model, x, n_low=0.35, n_high=7):
    D = 2/model.dN

    N_eq = model.N_eq
    vp = model.vp
    p = model.pC(x)
    gD = gamma(D)
    
    #print(D, gD, vp, x, N_eq)

    def I1(x):
        p = model.pC(x)
        de_mN = (D/x)**D*exp(-D/x)/gD
        return (1 - p)*p/(de_mN*N_eq*N_eq)

    def I2(x):
        p = model.pC(x)
        de_mN = (D/x)**D*exp(-D/x)/gD
        return p*p/(de_mN*N_eq*N_eq)
    
    if x < n_high and gD < 1e200:
        C = D*N_eq/vp
        if x < n_low:
            return C*safe_integrate(I1, n_low, n_high) + log(n_low/x)/vp + x/(D*vp)
        else:
            return C*(safe_integrate(I1, x, n_high) + (1-p)/p*(gamma(D-1)*(1 - gammainc(D - 1, D/n_low))/(D*N_eq*gD) + safe_integrate(I2, n_low, x))) + 1/(n_high*vp)
    else:
        return 1/(vp*x)

def hGillespie(model, trials=10000000, x_max=2):
    """Hybrid gillespie model"""
    theta = 1 + model.dN
    tau_max = model.t_max*model.vp        # tau = t*vp
    dimmensionless_if = model.vp/model.f/model.N_0    # In units of 
    n_cancers = 0
    Taus = np.empty(trials, dtype=np.double)
    for i in range(trials):
        x = 1
        Taus[n_cancers] = 0
        while x < x_max:
            Q = 1 - exponential()*dimmensionless_if/x
            x *= Q*theta
            Taus[n_cancers] -= log(Q)
            if Taus[n_cancers] > tau_max or x < 0: 
                break
        else:    
            n_cancers += 1
    return pd.Series(dict(
        P_cancer    = n_cancers/trials, 
        generations = Taus[0:n_cancers]/model.vp))

empirical_Pcancer_coeff = {
    2 : np.array([5.72640947,  -4.84536097]),
    3 : np.array([5.77629529,  -4.94655988,    0.04817712]),
    4 : np.array([10.82164987, -16.63749275,   8.31660758,  -1.75101626]),
    5 : np.array([10.78285714, -16.57518638,   8.32469893,  -1.79888812,   0.01717027]),
    6 : np.array([12.38219463, -22.35712125,  15.97711017,  -6.31622209,   1.13227449,  -0.07230064])}

def empiricalPcancer10(y, order=2):
    """assumes sd=0.1 """
    B = empirical_Pcancer_coeff[order]
    return 1/(1+exp((B*pow(y, np.arange(len(B)))).sum() ))

