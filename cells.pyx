#!/usr/local/bin/ipython3 -i
from scipy.optimize import brentq, newton, fsolve
from collections import Counter
from kt import first
from numpy cimport *
import numpy as np
import array
import pandas as pd

cdef extern from "c/cell_functions.h":
    void setup(unsigned long N_min, unsigned long N_max, double U_d, double U_p, double s_d, double s_p)
    void end()
    ctypedef struct haplo_t:
        haplo_t *parent
        double ds
        double t
    unsigned long simulation(unsigned long N_0, unsigned long t_max, unsigned long *Nt, double *Di, haplo_t **CTCs, double *fitness_array)
    unsigned long get_haplotype(unsigned long i) 
    double get_fitness(unsigned long i) 
    double collect_driver()
    unsigned int fixed_passengers()

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)

implicit_parameters = dict(
    nmax          = lambda N_0: np.uint64(2*N_0), 
    N_init        = lambda N_0: int(round(N_0)), 
    fitness_array = lambda N_0: np.ones(int(round(N_0))))

class sim(first):
    #Attribute        #Default
    nu=                1.5
    death=            'linear'
    nmin=            1
    state=            0                #Demanded state: -1 = simulate until nmin, 0 = simulate once, 1 = simulate until nmax

    def _create_haplotype_branch(self, unsigned long haplo_id):
        cdef:
            haplo_t my_struct = (<haplo_t*>haplo_id)[0]
            unsigned long parent = <unsigned long>(my_struct.parent)
            haplo_t current_struct
            unsigned long next_parent
            unsigned long child_id
            
        assert parent < haplo_id, "parent was created after me"
        addons = [(haplo_id, dict(
            children = [], 
            parent   = parent, 
            ds       = my_struct.ds,
            t        = my_struct.t))]
    
        child_id = haplo_id
        while parent not in self.haplotypes:
            current_struct = (<haplo_t*>parent)[0]
            next_parent = <unsigned long>(current_struct.parent)
            addons.append((parent, dict(
                children=[],
                parent=next_parent,
                ds=current_struct.ds,
                t=current_struct.t)))
            child_id = parent
            parent = next_parent
    
        addons.append((parent, self.haplotypes.pop(parent)))
        addons.reverse()
        for (_parent_id, parent_dict), (current_id, current_dict) in zip(addons, addons[1:]):
            current_dict.update({
                'drivers':parent_dict['drivers'] + int(current_dict['ds'] > 0),
                'passengers':parent_dict['passengers'] + int(current_dict['ds'] <= 0)})
            parent_dict['children'].append(current_id)
        self.haplotypes.update(dict(addons))

    def getDemographics(self, int n):
        # All initial cells have `0` as their haplotype parent -- this serves as a fake root to the haplotype tree
        self.haplotypes = {0:dict(drivers=-1, passengers=0,children=[], parent=-1, ds=0, t=0)}    
        # Because all initial cell fitnesses are >0, the root cells will all have a ds > 0 that will be annotated as a driver (even though they have no drivers). Setting the first driver to -1 fixes this.
        
        self.leaves = pd.Series(Counter([(get_haplotype(n_i), get_fitness(n_i)) for n_i in range(n)])).reset_index()
        self.leaves.columns = ['ID', 'fitness', 'N']
        list(map(self._create_haplotype_branch, self.leaves.ID))
        fake_root = self.haplotypes.pop(0)
        self.roots = fake_root['children']
        self.haplotypes = pd.DataFrame(self.haplotypes).T.sort_index()
        self.haplotypes.index.names = ['ID']
        self.haplotypes.loc[self.roots, 'ds'] = np.nan  
        self.haplotypes.loc[self.roots, 't'] = 0
       
        self.leaves = self.leaves.set_index('ID')
        return pd.concat([self.leaves, self.haplotypes.loc[self.leaves.index, ['drivers', 'passengers']].astype(int)], axis=1)
    
    def getFixedMutations(self):
        fixed = []
        if len(self.roots) > 1:
            return np.array(fixed)
        self.MRCA = self.haplotypes.loc[self.roots[0]]
        while len(self.MRCA['children']) == 1:
            self.MRCA = self.haplotypes.loc[self.MRCA['children'][0]]
            fixed.append(self.MRCA.ds)
        return np.array(fixed)

    def __init__(self, **kargs):
        import warnings
        if 'N_0' in kargs: 
            self.N_0 = kargs['N_0']
        for param, funct in implicit_parameters.items():
            if not hasattr(self, param): setattr(self, param, funct(self.N_0))
        for key in kargs.keys():
            assert hasattr(self, key), "'{:}' is not a parameter.".format(key)
        self.__dict__.update(kargs)
        if self.nmax > 1e9:
            warnings.warn("Maximum population size will consume >4 GB of memory.")
        if self.nmax*np.log(self.nmax)*self.t_max > 1e12:
            warnings.warn("Individual simulation runtime may exceed several hours.")

        if self.N_init >= self.nmax:
            self.Nt = np.array([self.N_init])
            self.state = 1
            return

        assert self.nmin > 0, "Problems arise when nmin < 1."
        cdef:
            unsigned long generations=0, haplo 
            ndarray[dtype=double, ndim=1] Di = np.arange(self.nmin, self.nmax,dtype=np.double)
            ndarray[dtype=unsigned long, ndim=1] Nt = np.zeros(self.t_max+1,dtype=np.ulonglong)
            ndarray[dtype=double, ndim=1] fitnessArray = self.fitness_array 
            haplo_t **CTCs = NULL
# SETUP Di
        if   self.death=='linear':      Di = self.N_0/(Di*Di)
        elif self.death=='constant':    Di = 1/Di
        elif self.death=='Gomp-Ex':     Di = 1/(np.log(1 + Di*(np.e-1)/self.N_0)*Di)
        elif self.death=='logistic':    Di = np.pow(Di/self.N_0,self.nu)/Di
        elif self.death=='fixed':       Di = np.r_[np.repeat(np.inf, np.int(self.N_0)), np.zeros(self.nmax - np.int(self.N_0))]

        setup(self.nmin, self.nmax, self.Ud, self.Up, self.sd, self.sp)
        if self.ctcs: 
            assert self.tree, "Can't collect CTCs without enabling TREE."
            CTCs = <haplo_t **>malloc(self.t_max*sizeof(haplo_t*))    
            assert <unsigned long>CTCs != 0, "Failed to allocate."
        
        for 1 <= i < 999999:
            self.p('Trial ' + str(i))
            generations = simulation(self.N_init, self.t_max, <unsigned long*> (Nt.data), (<double*> (Di.data)) - <unsigned long>self.nmin, <haplo_t**>CTCs, <double*> (fitnessArray.data))
            n = Nt[generations]
            state = 1 if n >= self.nmax else (-1 if n <= self.nmin else 0)
            if self.state == 0 or self.state == state: 
                break
        self.state = state
        self.Nt = Nt[0:generations+1]

        cdef double driver = 1
        if self.tree and n > 1:
            # Get every driver ever created
            if self.driver_distribution > 0: 
                all_drivers = array.array('d')
                while driver >= 0:
                    driver = collect_driver()
                    all_drivers.append(driver)
                self.all_drivers = np.array(all_drivers)
            # Create haplotype tree
            self.dem = self.getDemographics(n)
            if self.ctcs: 
                self.CTCs = np.empty(len(self.Nt) - 1, dtype=[('drivers', np.int), ('passengers', np.int)])
                self.CTC_haplo_IDs = [<unsigned long>(ctc) for ctc in CTCs[0:len(self.Nt)-1]]
                free(CTCs)
            self.fixed_mutations = self.getFixedMutations()
        end()

    def plot(self, figname=None):
        from matplotlib import pyplot as plt
        ax = plt.plot(self.Nt)
        ax.set( xlabel='time (generations)', 
                ylabel='population')
        if figname is not None:
            plt.savefig(figname)
        else:
            return ax 

    def treat(self, double dsp=1, double du=1, dnmax=1.1):
        #assert self.state == 1, 'unsuccessful tumor'
        assert hasattr(self, 'dem'), 'Must simulate w/ tree'
        cdef:
            double numerator = 1+self.sd
            double denominator = 1/(1+self.sp*dsp)
        self.fitness_array = np.r_[tuple([np.repeat(numerator**d*denominator**p, n) for d, p, __old_fitness__, n in self.dem])]
        self.state = 0
        oldUdUp = np.array([self.Ud, self.Up])
        self.Ud, self.Up = du*oldUdUp
        self.N_init = len(self.fitness_array)
        self.nmax = int(dnmax*self.N_init)
        self.__init__()
        self.Ud, self.Up = oldUdUp

