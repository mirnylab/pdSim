#/bin/bash

NJOBS=10
DIRECTORY=`pwd`
NAME=`basename $DIRECTORY`
MEMORY="128GB" # Normal node contains 128 GB of memory & 20 cores
#MEMORY="196GB" # Normal node contains 128 GB of memory & 20 cores
NODES=1
CORES=20 #24
SCRIPT="$HOME/pd_sim/single_sherlock_job.py"
QUEUE="owners,hns"
TIME=2-0

sbatch --array 1-$NJOBS -J $NAME -D $DIRECTORY --mem=$MEMORY --get-user-env --time=$TIME -N $NODES -n $CORES -p $QUEUE --output=$DIRECTORY/%a.out --wrap $SCRIPT
