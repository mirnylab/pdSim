# pdSim
Gillespie simulations of advantageous drivers &amp; deleterious passengers in cancer

## Overview

This simulator is written in C with an emphasis on very-fast simulations of moderately-sized tumors. A forward Gillespie algorithm (the Next Reaction) is used to model cell birth, death, and mutations. This code is written at a low-level and not designed for ease-of-use for the general population. 

## Installation

The simulator is best accessed via a python wrapper. To download the code & build the python wrapper:

```sh
$ git clone https://github.com/cd-mcfarland/pdSim.git # get repository
$ cd pdSim
$ ./setup.py build_ext  # Build the python wrapper that runs the simulator
```
The simulator has a number of precompiler directives that alter the kind of simulation that is run, including: 

1. The Distribution of Fitness Effects (DFEs) that driver & passenger mutations are drawn from,
2. Whether-or-not the entire lineage history of the dividing cells is tracked, 
3. The epistasis effect between multiple mutations, 
4. Whether-or-not Circulating Tumor Cells (CTCs) are generated (to be used to simulate tumor growth at distant stroma),
5. Whether-or-not drivers are drawn from a psuedo-random DFE, or an array of specific genes. 

`load_cells.py` organizes these precompiler options into keyword arguments of a python function that then compiles the code accordingly and then returns a python class `sim` (all at runtime). `sim` then executes a simulation upon instantiation. The various outputs of the simulation are accessed via methods and attributes of the returned object. An example of usage:

```python

>>> from load_cells import cells
>>> sim = cells(passenger_distribution=2, tree=1) # passengers are drawn from a Log-Normal DFE; lineage is tracked
>>> tumor = sim(sd=0.25, N_0=300)                 # simulate tumor growth with non-default values of sd, N_0
>>> passenger_mutations = tumor.fixed_mutations[tumor.fixed_mutations <= 0]  
# fixed_mutations is a numpy.ndarray that records the fitness effects of every mutation that sweeps to fixation
# in temporal order. Passengers are, by definition, neutral or deleterious mutations. 
>>> print('Mean fitness cost of fixated passengers: {:.3f}'.format(passenger_mutations.mean())
>>> tumor.plot()                                  # Plot the tumor's population size versus time.
```

The doc strings of the various methods and attributes should clarify the kinds of output you can retrieve. However, in general the code is not very accessible and was never written for general use. Knowledge of C & cython is needed, if you want to extend the code. 

`simple.py` also provides a useful, well-documented example of how the simulator might be used to model dN/dS statistics.

## Contact

Please contact [Christopher McFarland](mailto:christopherdmcfarland@gmail.com) for all questions. 
