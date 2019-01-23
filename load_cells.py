import os, time, sys, subprocess, numpy, warnings 

compilations = 0

def cells(optimization='O3', driver_distribution=0, passenger_distribution=0, tree=0, shape=1, epistasis=2, ctcs=0, explicit_drivers=0, loci=70, cleanup=True):
    """Compile the `Cells` python class for simulating driver & passenger cancer simulations. 

Compilation is performed using Cython & gcc. Several simulation options are specified at runtime:

Parameters:

optimization (str, default: O3): gcc optimization level to use

driver_distribution (int, default 0): Sampling distribution of driver alteration fitness benefits. 
    `sd` defines the scale of each distribution, while `shape` defines the shape parameter. All 
    scale and shape parameters are defined in the Wikipedia page for each distribution.
        0 ~ Fixed       : delta(x - sd)
        1 ~ Exponential : sd*exp(-x)
        2 ~ LogNormal[x | mu=sd, sigma=shape)
        3 ~ Gamma[x | theta=sd, k=shape]

passenger_distribution (int, default: 0): Same as `driver_distribution`, except for passengers. 

shape (float, default: 1): Shape parameter for driver/passenger distributions. 

tree (int, default: 0): If 1, will extract the entire phylogenetic tree (or lineage history) of the simulation. 
    Currently, the only output that is kept are the lineages that contribute to the surviving population at the
    end of simulation. 

epistasis ({1, 2}, default: 2): 1 ~ `Additive Epistasis` between mutation, 2 ~ `Multiplicative Epistasis`.

ctcs ({0, 1}, default: 0): If 1, create Circulating Tumor Cells (CTCs) for metastatic simulations. 

explicit_driver ({0, 1}, default: 0): If 1, an array of length `loci` (default: 70) is provided, which defines 
    the specific benefit of each driver mutation.

cleanup (bool, default: True): Remove compilation directory at exit.  
"""
    global compilations
    if compilations > 0:
        warnings.warn('Already compiled the code once in this session -- cannot compile twice on *some* setups.')
    FILE = 'cells'
    params = locals().copy()
    scripts = os.path.dirname(__file__)
    np_include = numpy.get_include()
    ID = 'pd_simulator_'+str(int(time.time()*100000000) + os.getpid())
    os.makedirs(ID)
    cythonize_command = 'cython -X language_level=2 -X boundscheck=False -X wraparound=False -X cdivision=True -o {ID}/{FILE}.c {scripts}/{FILE}.pyx'.format(**locals())
    definitions = ' '.join(['-D{0}={1}'.format(k.upper(), v) for k, v in params.items()]) 
    gcc_command = 'gcc -shared -fpic -fwrapv -I/usr/include/python3.6 -I{np_include} -std=c99 -{optimization} {definitions} -o {ID}/{FILE}.so {ID}/{FILE}.c'.format(**locals())
    for command in [cythonize_command, gcc_command]:
        line = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        if line: 
            print(line.decode("UTF-8"))
    
    compilations += 1
    sys.path.append(ID)
    t = __import__(FILE).sim
    for k, v in params.items(): 
        setattr(t, k, v)
    t.ID = ID
    if cleanup: 
        import atexit, shutil 
        atexit.register(lambda ID: shutil.rmtree(ID, ignore_errors=True), ID)
    return t

