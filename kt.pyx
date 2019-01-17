from numpy cimport *
from numpy import empty, zeros, ones, corrcoef, tile, concatenate, fromiter, uint32, uint8, double, array, arange, dot, finfo, seterr, ndindex, r_, repeat, copy, loadtxt, outer, random, vstack, hstack, zeros_like, ones_like, empty_like, vectorize, transpose, pi, linspace, e, ceil, nan
import numpy as np
seterr(over='ignore', invalid='ignore', divide='ignore')
from scipy.special import gammainc, gamma, erf
import functools

cdef extern from "c/exponential.h":
    double exponential()
    void mt_init()

mt_init()               # see final.h

cdef extern from "math.h":
    double log( double x)
    double sqrt( double x)
    double exp( double x)
    double pow(double x, double y)

def getUdUp(u=1e-8, **otherParams):
    Tdf = 2*700
    Tpf = 2*5e6
    TdTp = otherParams.pop('TdTp', Tdf/Tpf)
    Td = sqrt(TdTp*Tpf*Tdf)
    Tp = sqrt(Tdf*Tpf/TdTp)
    return dict(Ud=u*Td, Up=u*Tp, **otherParams)

class first:
    Ud = 1e-8*700
    Up = 1e-8*5e6
    sd = 0.1
    sp = 1e-3
    N_0 = 1e3
    t_max = 10000
    verbose = False

    def p(self, txt):
        if self.verbose: print txt

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
    cdef double sd = model.sd
    def f(V):
        A = V/model.Ud; B = log(A);
        return model.N - (exp(V/2/sd*((B-1)**2+1)-0.5*(log(sd**3/(A*B))- B if V>=sd else 0) ) if V>sd/B else 0.5*A/sd)
    return fsolve(f, sd)

@functools.lru_cache(maxsize=None)
def fAnddN(double Up, double sd, double sp, double N):
    from scipy.stats import poisson
    cdef:
        double L = Up/sp
        int i
        int i_max = int(ceil(log(1+sd)/log(1+sp)))
        int k = (poisson.cdf(arange(L), L) < 0.5*pow(N,-0.5)).sum()
        ndarray[int64_t,  ndim=1] X = arange(i_max)
        ndarray[double_t, ndim=1] W = (1+sd)*(1+sp)**-X
        ndarray[double_t, ndim=1] f = poisson.pmf(X+k, L)/(1-poisson.cdf(k-1, L))
        ndarray[double_t, ndim=1] fP = f*(W-1)/W
        ndarray[double_t, ndim=1] dNp = empty(i_max)
        
    Lambda = fP.sum()
    assert Lambda < sd/(1+sd), 'wtf!'
    
    dNp[i_max - 1] = i_max - 1
    for i in range(i_max-2, -1, -1): dNp[i] = (i + sp*dNp[i+1])/(1 + sp)
    V = (1 + sd)*(1 + sp)**-dNp
    assert (V <= 1+sd).all()
    return Lambda, fP.dot(V)/Lambda - 1
    
from scipy.integrate import simps
def moran(double Up, double sp, double N, int r1=150, int r2=600):
    cdef double x0 = exp(-Up/sp), spp =sp/(e-1)
    a, x1, x2 = N*spp/(2*x0), linspace(x0/r1, x0, r1), linspace(x0, 1, r2)
    erfi = r_[erf(1j*sqrt(a)*(x1-x0-x1[0])).imag,0]
    I, G1, G2 = sqrt(pi/(4*a))*exp(-a*x0**2)*(erfi[1:]-erfi[0]), np.exp(a*x1*(x1-2*x0)), np.exp(a*x2*(x2-2*x0))
    Tclick = max(1/sp*(1-e*spp/Up), 0)
    Tclick += N/2*(simps(r_[1,I/(x1*G1)],dx=x1[0])+I[-1]*simps(1./(x2*G2),dx=x2[1]-x2[0]))
    return Up*1/Tclick

def moran(double Up, double sp, double N, int r1=150, int r2=600):
    cdef double x0 = exp(-Up/sp), spp =sp/(e-1)
    Tclick = exp(0.5*spp*N*x0)/spp
    return Up*1/Tclick

#def moran(double Up, double sp, double N, double alpha=1/(e-1)):
#    cdef:
#        double No = N*exp(-Up/sp)
#        double Nosa = No*sp*alpha
#    return sqrt(Nosa*alpha*alpha/pi)*exp(-Nosa)

def wave(double Up, double sp, double N, double v_0=1e-12):
    from scipy.optimize import brentq
    x2 = sp*log(N)
    if Up < x2:
        No = N*exp(-Up/sp)
        pi_p = sp/((1 + sp)**No - 1) #if sp > 5e-5 else 1/(No*(1+2*sp))
        #y1 = Up*N*sp/((1 + sp)**N - 1)    
        #y2 = wave(x2, sp, N)*Up/x2
        assert No*Up*pi_p > 0
        return  No*Up*pi_p #y2
    
    cdef:
        double iL = sp/Up
        double Nd2spSqrtiL = N/2*sp*sqrt(iL)
        double fiveSixthsiL = 5*iL/6
    def f(double v):
        cdef double A = log(e*Up/v), C = v/Up 
        return 1 - 0.5*C*(A*A + 1) - iL*log(Nd2spSqrtiL*sqrt(v*C*C/(Up - v))*A/(1 - C*A + fiveSixthsiL))
    cdef double vp = brentq(f, v_0, Up - v_0)
    assert vp > 0
    return min(vp, Up)

@functools.lru_cache(maxsize=None)
def Vp(double Up, double sp, double N):
    cdef double L = Up/sp
    if L > 1: 
        return wave(Up, sp, N)
    N_0 = N*exp(-L)
    return moran(Up, sp, N) if sp*N_0 > 1 else (1-L)*moran(Up, sp, N) + L*wave(Up, sp, N)

class second(first):
    @property
    def f(self): 
        print('New f')
        return fAnddN(self.Up, self.sd, self.sp, self.N_0)[0]*self.Ud

    @property
    def dN(self): return fAnddN(self.Up, self.sd, self.sp, self.N_0)[1]

    @property
    def vp(self): 
        print('New vp')
        return self.sp*Vp(self.Up, self.sp, self.N_0)

@functools.lru_cache(maxsize=None)
def simplified_fanddN(double sd, double sp, double Up, double N_0):
    from scipy.stats import poisson
    cdef:
        double Lambda = Up/sp
        ndarray[int64_t,  ndim=1] I = arange( int(sd/sp) + 1 )
        int k = (poisson.cdf(arange(int(Lambda)), Lambda) < 0.5*pow(N_0, -0.5)  ).sum()
        ndarray[double_t, ndim=1] f = poisson.pmf(I + k, Lambda)
        ndarray[double_t, ndim=1] W = sd - sp*I
    return dot(f, W/(1+W)), dot(f, W)/f.sum() 

class third(first):    
    @property
    def f(self): return self.Ud*simplified_fanddN(self.sd, self.sp, self.Up, self.N_0)[0]

    @property
    def dN(self): return simplified_fanddN(self.sd, self.sp, self.Up, self.N_0)[1]

    @property
    def vp(self):
        from scipy.stats import poisson
        cdef:
            double Up = self.Up
            double sp = self.sp
            double sd = self.sd
            double Lambda = Up/sp
            ndarray[int64_t,  ndim=1] I = arange( int(sd/sp) + 1 )
            int k = (poisson.cdf(arange(int(Lambda)), Lambda) < 0.5*pow(self.N_0, -0.5)  ).sum()
            double No = self.N_0*poisson.pmf(k, Lambda)            #exp(-Lambda) 
            double pi_p = sp/((1 + sp)**No - 1) if sp > 5e-5 else 1/(No*(1+2*sp))
        return Up*sp*No*pi_p if No > 1e-1 else Up*sp

def tC(model, x, n_low=0.35, n_high=7):
    from scipy.integrate import quadrature
    D = 2/model.dN
    N_eq = model.N_eq
    vp = model.vp
    p = model.pC(x)
    gD = gamma(D)

    def I1(x):
        p = model.pC(x)
        de_mN = (D/x)**D*exp(-D/x)/gD
        return (1 - p)*p/(de_mN*N_eq*N_eq)

    def I2(x):
        p = model.pC(x)
        de_mN = (D/x)**D*exp(-D/x)/gD
        return p*p/(de_mN*N_eq*N_eq)

    if x < n_high:
        C = D*N_eq/vp
        if x < n_low:
            return C*quadrature(I1, n_low, n_high)[0] + log(n_low/x)/vp + x/(D*vp)
        else:
            return C*(quadrature(I1, x, n_high)[0] + (1-p)/p*(gamma(D-1)*(1 - gammainc(D - 1, D/n_low))/(D*N_eq*gD) + quadrature(I2, n_low, x)[0])) + 1/(n_high*vp)
    else:
        return 1/(vp*x)

def hGillespie(model, long trials=10000000, double x_max=20):
    """Hybrid gillespie model"""
    cdef:
        double theta = 1 + model.dN
        double tau_max = model.t_max*model.vp        # tau = t*vp
        double dimmensionless_if = model.vp/model.f/model.N_0    # In units of 
        double x, Q
        long n_cancers = 0
        long i
        ndarray[double_t, ndim=1] Taus = empty(trials, dtype=double)
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
    return {'P_cancer':double(n_cancers)/trials, 'generations':Taus[0:n_cancers]/model.vp}

#def hGompGillespie(model, long trials=10000000, double y_max=20):
# Solution to waiting-time jump is transcendental equation, with Exponential Integral function in it!
#    """hybrid gillespie using gomp-ex death rate"""
#
#    cdef:
#        double vp = model.vp
#        double theta = 1 + model.dN
#        double jumpRate = model.f*model.N_0/vp
#        double em1dJump = (e - 1)/jumpRate
#        double omega_d
#        double sigma_d
#        double eSigma_d
#        double t
#    
#    for i in range(trials):
#        y = 1
#        sigma_d = 0
#        omega_d = -sigma_d*vp
#        t = 0
#        eSigma_d = exp(sigma_d)

B2 = array([5.72640947,  -4.84536097])
B3 = array([5.77629529,  -4.94655988,    0.04817712])
B4 = array([10.82164987, -16.63749275,   8.31660758,  -1.75101626])
B5 = array([10.78285714, -16.57518638,   8.32469893,  -1.79888812,   0.01717027])
B6 = array([12.38219463, -22.35712125,  15.97711017,  -6.31622209,   1.13227449,  -0.07230064])

def empiricalPcancer10(y, f=lambda y, B: 1/(1+exp((B*pow(y, arange(len(B)))).sum() )), B=B6):
    """assumes sd=0.1 """
    return f(y, B)




#Pcancer = {0.1: 0.415*N_eq,
#0.2:67, 
#0.05: 0.94*N_eq =
#0.96*N_eq = 
#0.98*N_eq =
#}


#        while y < y_max:
#            dt = log(em1dJump*omega_d*exponential() + 1)/omega_d
#            y *= eTheta 

#            t = log(omega_d*em1/(jumpRate*eSigma_d)*exponential() + exp(omega_d*t))/omega_d
#            sigma_d += theta
#            eSigma_d = nan
#            y *= eSigma 







#def ndnp(double Ud=1e-8, double sd=0.1, double sp=0.001, double K=1e3, double Td=700, double Tp=5000000, double Genome_size = 149948690/4, long trials=50000000, double Nmax=2):
#
#random.poisson(T*u*(Genome_size-2*Tp))
#    cdef double Ud = 2*u*Td, Up = 2*u*Tp
#    out = lambdaG(Ud, Up, sd, sp, No)
#    cdef double vp = sp*Vp(sp, Up, No), L = out[0], dN = out[1]+1, N, Q, T, m_hitchhikers = (sd - out[1])/sp
#    cdef ndarray[uint32_t, ndim=2] Trial = empty((Iter,3),uint32)
#    cdef uint32_t i, j = 0, d
#    for i in range(Iter):
#        N, T, d = No, 0, 0
#        while N > 0 and T < Time:
#            Q = 1 - exponential()*vp/(N*L)
#            N *= Q*dN
#            d += 1
#            T -= log(Q)/vp
#            if N > No*Top:
#                j+=1
#                break
#    print j
#    return Trial[:j,:]

