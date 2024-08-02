__author__ = "EAM"

# import system functions
import os
import sys

# import packages
import numpy as np
import scipy.io

# import tools
cwd = os.getcwd()
basuite_path = cwd+r'\..\..\bayesian_averaging_suite'
sys.path.append(basuite_path)

from model_selection_algorithms import BMF
from plots_and_metrics import make_plots, make_metrics

# state desired leadatime
lead_time = 24

# load test data
import mat73
dm = mat73.loadmat("..\O6_designmatrix\dm_normalised.mat")
dm = dm['dm_normalised']
dm_means = scipy.io.loadmat("..\O6_designmatrix\dm_mean.mat")['dm_mean']
dm_stdevs = scipy.io.loadmat("..\O6_designmatrix\dm_std.mat")['dm_std']

target_id = -4
n_feat = 44

# split test and abms 

targets_abms = dm[:-40000,target_id,:].copy()
features_abms = dm[:-40000,:n_feat,:].copy()
features_abms = np.swapaxes(features_abms, 1, 2)

targets_test = dm[-40000:,target_id,:].copy()
features_test = dm[-40000:,:44,:].copy()
features_test = np.swapaxes(features_test, 1, 2)

targets_test_ext = dm[-40000:,-4,-1].copy()

# set directory 
NNSdirectory = cwd+r'\..\O7_lstm_train\wave_height_models'

robust_prediction = BMF(
                        msa='ABMS',
                        NNSdirectory=NNSdirectory,
                        workingdirectory=cwd,
                        features=features_test,
                        means=dm_means[0, target_id],
                        stdevs=dm_stdevs[0, target_id],
                        alpha=1.96,
                        lead_time=lead_time+1,
                        post_distro=None,
                        post_data=(features_abms, targets_abms),
                        estimation_strategy='GaussianMixture',
                        posterior_variable_list=['Target'],
                        normalised=False
                       )

test = (targets_test[:, lead_time+1]*dm_stdevs[0, target_id])+dm_means[0, target_id]

robust_save = (robust_prediction[0], robust_prediction[1], robust_prediction[2], test)

np.savetxt('FOC_ABMS_wave_height_output.csv', robust_save, delimiter=',')

make_plots(variable='Significant Wave Height', 
           robust_prediction=robust_prediction, 
           test=test, 
           normalised=False, 
           lead_time=lead_time, 
           msa='ABMS',
           plot_title='CMEMS Test', 
           plot_prefix='FOC_', 
           plot_directory=cwd)

make_metrics(variable='Significant Wave Height', 
             robust_prediction=robust_prediction, 
             test=test, 
             normalised=False, 
             metric_prefix='FOC_CONF95_', 
             metric_directory=cwd, 
             thres1=2.0, thres2=2.5, 
             lead_time=lead_time, msa='ABMS')
