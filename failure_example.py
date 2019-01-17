#!/usr/bin/env python3

#################
### Libraries ###
#################

from simple import simplest_simulations
from simple import single_simulation
import pandas as pd
import numpy as np
import sys
from bootstrap import sample
import os

##############################
### Directories and Inputs ###
##############################

#sd = float(sys.argv[1])
#sp = float(sys.argv[2])
#trialIndex = int(sys.argv[3]) ### string used for file naming
sd = 0.07498942 
sp = 0.0001
trialIndex = 1

#os.chdir('/home/groups/dpetrov/tilk/pd_sim')
#outputDir='/home/groups/dpetrov/tilk/simulationsOutput/'
#outputDir='/home/groups/dpetrov/tilk/simulationsOutput/individualSimulations/simulationsRanSeparately_1_13_19/'
outputDir=''
dirName=outputDir + 'sd_' + str(sd) + '_sp_' + str(sp) + '/'
if not os.path.exists(dirName):
    os.mkdir(dirName)

#################
### Functions ###
#################

'''
	Generate simulations of tumor growth according to free parameters of sd and sp.
'''

def generateSimulation(sd, sp, trials):
	try:
		results = simplest_simulations(sd, sp, trials=trials)
		results['log10_mu'] = np.log10(results['mu'])
		results['TotalSubstitutions'] = results['Fixed_Drivers'] + results['Fixed_Passengers'] + results['driver_dS'] + results['passenger_dS']
		results.to_csv(dirName + '/sd=' + str(sd) + '_sp=' + str(sp) + '_trials=' + str(trialIndex), sep='\t')
	except:
		results=pd.DataFrame()
		results.to_csv(dirName + '/sd=' + str(sd) + '_sp=' + str(sp) + '_trials=exceedsMax', sep='\t')
	return(results)

def binData(df):
	Mol_Clock = pd.Series(df['TotalSubstitutions'], name='mutRate')
	X_bins = np.r_[np.logspace(0, 4, numBins), Mol_Clock.max() + 1]
	xbin = np.searchsorted(X_bins, Mol_Clock)
	df.set_index([
		pd.Index(Mol_Clock, name='mutRate'), 
		pd.Index(xbin, name='xbin'),
		], inplace=True, append=True)
	df = df.swaplevel(0, -1).sort_index()
	return(df)

def kaks_passengers(df):
	if weightedByIndividualCounts:
		return(df.groupby(level='TotalSubstitutions').sum().eval('(Fixed_Passengers)/(passenger_dS)'))
	else:
		return(df.groupby(level='xbin').sum().eval('(Fixed_Passengers)/(passenger_dS)'))

def kaks_drivers(df):
	if weightedByIndividualCounts:
		return(df.groupby(level='TotalSubstitutions').sum().eval('(Fixed_Drivers)/(driver_dS)'))
	else:
		return(df.groupby(level='xbin').sum().eval('(Fixed_Drivers)/(driver_dS)'))

def getPlottingTable(data):
	highestNumberOfTotalMuts = data.reset_index()['mutRate'].max()
	X_bins = np.r_[np.logspace(0, 4, numBins), highestNumberOfTotalMuts + 1]
	nPatients = data.groupby(level=['xbin']).size().fillna(0).astype(int)
	col_names=['low','true','high']   
	drivers = sample(data, kaks_drivers).CI().replace(np.inf, 0)
	drivers['type'] = 'drivers'
	passengers = sample(data, kaks_passengers).CI().replace(np.inf, 0)
	passengers['type'] = 'passengers'
	appendDF = passengers.append(drivers)
	outDF = pd.DataFrame(columns = col_names)
	outDF['low']=np.repeat(appendDF['low'], 2)
	outDF['true']=np.repeat(appendDF['true'], 2)
	outDF['high']=np.repeat(appendDF['high'], 2)
	outDF['type']=np.repeat(appendDF['type'], 2)
	outDF['mutRate']= np.concatenate([np.repeat(X_bins, 2)[1:len(appendDF) +1], np.repeat(X_bins, 2)[1:len(appendDF) +1]])
	outDF['numBins']= numBins
	outDF.to_csv(dirName + 'sd=' + str(sd) + '_sp=' + str(sp) + '_trials=' + str(trialIndex) + '_binned', sep='\t') 
	return(outDF)

def getPlottingTableByWeights(data):
	data['countOfMuts'] = 1 
	data = data.groupby('TotalSubstitutions').sum()
	drivers = sample(data, kaks_drivers).CI().replace(np.inf, 0)
	drivers['type'] = 'drivers'
	passengers = sample(data, kaks_passengers).CI().replace(np.inf, 0)
	passengers['type'] = 'passengers'
	appendDF = passengers.append(drivers)
	appendDF['countOfMuts'] = data['countOfMuts']
	appendDF.to_csv(dirName + 'sd=' + str(sd) + '_sp=' + str(sp) + '_trials=' + str(trialIndex) + '_binnedByEachMutValue', sep='\t') 
	return(appendDF)

#######################
### FREE PARAMETERS ###
#######################

global weightedByIndividualCounts
numBins=9
trials=100
number_of_driver_genes = 30 # ?? You tell me
print('SD: ' + str(sd) + ' and SP: ' + str(sp) )

##############################
### ESTABLISHED PARAMETERS ###
##############################

total_human_genes = 21686   # From Pfam-A (PMID: 14681378) 
mean_gene_length = 1298     # From Pfam-A (PMID: 14681378)
neutral_r = 2.7985          # The neutral dn/ds (from my COSMIC analyses)
P_nonsynonymous = neutral_r/(1 + neutral_r)
number_of_functional_loci = total_human_genes*mean_gene_length*P_nonsynonymous
max_generations = 18500     # If 1 division = 2 days, then 18,500 generations ~ 100 years
Td = number_of_driver_genes*P_nonsynonymous*mean_gene_length   # Td = Target size of functional drivers
mu_min = 1e-12              # Minimum possible mutation rate (per nucleotide per generation) 
mu_max = 1e-7               # Max possible mutation rate

###########
### RUN ###
###########

#fileName=dirName + 'sd=' + str(sd) + '_sp=' + str(sp) + '_trials=' + str(trialIndex)
#if not os.path.isfile(fileName): 

df = generateSimulation(sd, sp, trials)

#weightedByIndividualCounts=True
#df_weighted = getPlottingTableByWeights(df)
#df = binData(df)
#weightedByIndividualCounts=False
#df_unweighted = getPlottingTable(df)


