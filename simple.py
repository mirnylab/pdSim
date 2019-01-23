#!/usr/bin/env python3
"""Simplified access to Chris McFarland's deleterious passenger (and advantageous driver) cancer simulator. 

Assumptions:
------------

    1) Two classes of mutations: Advantageous drivers & deleterious passengers
    2) Population is well-mixed (un-relaxable with this simulator),
    3) Epistasis is 'multiplicative',
    4) Infinite loci,
    5) 'Gomp-Ex' growth rate in absence of de novo mutations,

Access is condensed for easier usage. 
"""
import pandas as pd
import numpy as np
#from pmap import large_iter_pmap as pmap
from pmap import pmap
from theory import second as theoretical_model, mean_time
###############################################################################
# Input Parameters
###############################################################################

exponential_distributions = True    # Draw drivers/passengers from Exponential probability distributions, rather than sampling the exact same value every time:w

total_human_genes = 21686   # From Pfam-A (PMID: 14681378) 
mean_gene_length = 1298     # From Pfam-A (PMID: 14681378)
neutral_r = 2.7985          # The neutral dn/ds (from my COSMIC analyses)
P_nonsynonymous = neutral_r/(1 + neutral_r)
number_of_functional_loci = total_human_genes*mean_gene_length*P_nonsynonymous
t_max = 18500     # Maximum duration of simulation in cell division; if 1 division = 2 days, then 18,500 generations ~ 100 years
number_of_driver_genes = 30 # ?? You tell me
# Td = Target size of functional drivers
Td = number_of_driver_genes*P_nonsynonymous*mean_gene_length    

# To simplify ABC optimization, genome-wide mutation rates (mu) are not tunable 
# in this implementation. Instead, a genome-wide mutation rate is sampled 
# uniformly (in log-space) from a broad range of possible values. Extreme values 
# of mu naturally fail to progress to cancer because (1) very-low mutation rates
# evolve too slowly and do not grow sufficiently within `t_max` time, 
# while (2) very-high mutation rates suffer from mutational-meltdown. 

mu_min = 1e-12              # Minimum possible mutation rate (per nucleotide per generation) 
mu_max = 1e-7               # Max possible mutation rate

# mu > 1e-12 invariably fail to progress, while mu < 1-e7 invariably suffer from
# mutational meltdown, so it is unhelpful to sample beyond this range and will 
# hurt runtime (because more unsuccessful tumor populations are simulated), but
# there is no harm in expanding this range as the population dynamics constrain
# mu and not these explicit bounds. 

###############################################################################

from load_cells import cells    # Load cells compiles the simulator
simulate = cells(tree=1, driver_distribution=1, passenger_distribution=1) if exponential_distributions else cells(tree=1)
                                # The simulator is compiled with simulation options as declaratives for the precompiler
                                # (a la Gromacs). This maximizes compute performance. 

def single_simulation(simulation_kwargs):
    """Simulates a successful tumor population. 
    
Runs simulations until successful growth is obtained (state == 1) and re-samples
mu for every run."""
    np.random.seed()                    # Threads begin with same random state !!
    Td = simulation_kwargs.pop('Td')
    Tp = simulation_kwargs.pop('Tp')
    
    while True:
        mu = np.exp(np.random.uniform(np.log(mu_min), np.log(mu_max)))
        t = simulate(death='Gomp-Ex', Ud=mu*Td, Up=mu*Tp, **simulation_kwargs)
        assert t.Up == mu*Tp, 'Mutation rates fucked up!'
        if t.state == 1:
            return pd.Series(dict(
                Generations=len(t.Nt),
                Fixed_Passengers=(t.fixed_mutations <= 0).sum(),
                Fixed_Drivers=(t.fixed_mutations > 0).sum(),
                mu=mu,
                Mean_Drivers=t.dem.eval('N*drivers').sum()/t.Nt[-1],
                Mean_Passengers=t.dem.eval("N*passengers").sum()/t.Nt[-1], 
                MRCA_Age=len(t.Nt)*(t.MRCA.name - t.roots[0])/(t.leaves.index.to_series().mean() - t.roots[0]) if hasattr(t, 'MRCA') else 0)) 


def get_N_0(model, P_cancer=0.5):
    """Finds N_0 that leads to a particular P_cancer

Returns (N_0, limit) where limit can be 'time' (time-limited) or 'meltdown' (limited by mutational-meltdown events).
"""
    from scipy.optimize import brentq
    x = brentq(lambda x: model.pC(x) - P_cancer, 0.01, 10)
    AnticipatedTime = mean_time(model, x)
    isTimeLimited = AnticipatedTime > t_max
    if isTimeLimited:
        P_cancer = 0.5
        try:
            x = brentq(lambda x: mean_time(model, x) - t_max, 0.01, 7)
        except ValueError:
            x = brentq(lambda x: mean_time(model, x) - t_max, 0.01, 1e7)
    
    return x*model.N_eq, 'time' if isTimeLimited else 'meltdown'

def simplest_simulations(sd, sp, Td=Td, Tp=number_of_functional_loci-Td, P_cancer=0.5, max_N_0=1e5, relative_nmax=2, trials=2000, map=pmap):
    """Simulates WXS dataset with the fewest free parameters possible. 

sd (selection coeff. of driver) -> Free parameter (raison d'etre of ABC)
sp ( "  " of passsenger)        -> Free parameter (raison d'etre of ABC)
Tp (Target size of driver)      -> Free parameter (could be set using # of genes in COSMIC driver gene census).
Tp (Target size of passengers)  -> Constrained by size of proteome.
mu (Mutation rate)              -> Constrained as described above.
N_0 (Initial population size)   -> Constrained by a theoretical model (see below). 

Parameters:
-----------

P_cancer: Probability that a tumor progresses to cancer using first-order theoretical model. 
          Larger initial sizes make progression to cancer more likely, while smaller initial 
          sizes often suffer from mutational meltdown or do not evolve fast enough. The pop.
          gen. parameters (e.g. sd, sp, mu, ...) also affect progression probability in a 
          manner that the theoretical model can accommodate. Rather than set `N_0` explicitly
          it is more intuitive to pick an `N_0` that yields a reasonable probability of tumor
          progression. Previously, we assumed P_cancer ~ 10% or 1%, but 50% is less presumptuous 
          about why tumor progression is inefficient. 

relative_nmax: Size of tumor, relative to the equilibrium tumor size (N_eq), for which 
               carcinogenesis is defined. This size can be thought of as the size of 'transformation'
               or size at which 'inflation' begins in the Big Bang Model, but more generally it
               is the size at which the cancer's MRCA emerges. A larger nmax slows simulations
               and minimally-affects mutation accumulation statistics once its >1.5. Could be a
               free parameter moving forward -- I'm curious about the size at which the MRCA emerges. 

trials: # of successful tumors to simulate. 

map: map function to use to broadcast the simulations. pmap (parallel map) broadcasts simulations 
     across all threads on the computer. Moving forward, this algorithm may need to be fed a mapping 
     function that broadcasts simulations across Sherlock 2. 
"""
    
    model = theoretical_model(Ud=1e-8*Td, Up=1e-8*Tp, sd=sd, sp=sp, N_0=100)
    N_0, limit = get_N_0(model, P_cancer=P_cancer)

    info = pd.Series(dict(
        N_0 = N_0,
        limit=limit,
        aboveMaxSize=N_0 > max_N_0))

    output = pd.DataFrame([info for _ in range(trials)])
    if info['aboveMaxSize']:
        return output 
    
    ## Package parameters for broadcasting across distributed computing platform 
    absolute_nmax = int(round(max(N_0, model.N_eq)*relative_nmax))
    
    ## Simulate the tumors
    tumors = pd.DataFrame(list(map(single_simulation,
        [dict(Td=Td, Tp=Tp, nmax=absolute_nmax, sd=sd, sp=sp, N_0=N_0, t_max=t_max) for _ in range(trials)]))) 
    
    ## Return statistics with two ways to estimate dN/dS
    return pd.concat([output, tumors.assign(
        omega_d      = tumors['Mean_Drivers']/(tumors.eval('Generations * mu')*Td),          # Exact (theroetical) dN/dS of drivers using known mutation rate
        omega_p      = tumors['Mean_Passengers']/(tumors.eval('Generations * mu')*Tp),       #   "  "                    of passenger   "   "
        driver_dS    = (Td/neutral_r*tumors.eval("Generations * mu")).apply(np.random.poisson), # Poisson-sampling of a quantity of synonymous mutations in drivers given the theoretical mutation rate and driver target size
        passenger_dS = (Tp/neutral_r*tumors.eval("Generations * mu")).apply(np.random.poisson)) #   "  "                                                 in passengers  "  "
                     ], axis=1)

if __name__ == '__main__':
    results = simplest_simulations(0.1, 0.001, trials=2000)
    results['log10_mu'] = np.log10(results['mu'])
    means = results.mean()
    stds = results.std()
    print('Means:')
    print(means.to_string(float_format='{:.2f}'.format))
    print('Standard Deviations:')
    print(stds.to_string(float_format='{:.2f}'.format))
    
