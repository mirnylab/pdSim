#/bin/bash

# This is an example submission script of parameter_sweep_job.py that will run on Stanford's Sherlock2 Cluster.
# The cluster uses the SLURM job scheduler. 


NJOBS=2
DIRECTORY=`pwd`
NAME=`basename $DIRECTORY`
MEMORY="128GB" # Normal node contains 128 GB of memory & 20 cores
#MEMORY="196GB" # Normal node contains 128 GB of memory & 20 cores
NODES=1
CORES=20 #24
SCRIPT="$HOME/pdSim/parameter_sweep_job.py"
QUEUE="owners,hns"
TIME=2-0

sbatch --array 1-$NJOBS -J $NAME -D $DIRECTORY --mem=$MEMORY --get-user-env --time=$TIME -N $NODES -n $CORES -p $QUEUE --output=$DIRECTORY/%a.out --wrap $SCRIPT
