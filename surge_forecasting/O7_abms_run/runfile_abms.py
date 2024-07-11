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

from model_selection_algorithms import msa

# load test data
norm_dm = scipy.io.loadmat("..\O5_designmatrix\dm_normalised.mat")['dm_normalised']
dm_means = scipy.io.loadmat("..\O5_designmatrix\dm_mean.mat")['dm_mean']
dm_stdevs = scipy.io.loadmat("..\O5_designmatrix\dm_std.mat")['dm_std']

# split test and abms 
targets_abms = norm_dm[:-40000,-1,25:].copy()
features_abms = norm_dm[:-40000,:,:].copy()
features_abms[:, -1, 25:] = 0
features_abms = np.swapaxes(features_abms, 1, 2)

targets_test = norm_dm[-40000:,-1,25:].copy()
features_test = norm_dm[-40000:,:,:].copy()
features_test[:, -1, 25:] = 0
features_test = np.swapaxes(features_test, 1, 2)

del norm_dm

# set directory
NNSdirectory = cwd+r'\..\O6_lstm_train\surge_models'

robust_prediction = msa(
                        msa='ABMS',
                        NNSdirectory=NNSdirectory,
                        workingdirectory=cwd,
                        features=features_test,
                        means=dm_means,
                        stdevs=dm_stdevs,
                        alpha=1.96,
                        lead_time=24,
                        post_distro=None,
                        post_data=(features_abms, targets_abms),
                        estimation_strategy='GaussianMixture',
                        posterior_variable_list=['Target'],
                        plot_title='ERA5 Test',
                        make_plot=True,
                        plot_prefix=None,
                        plot_directory=cwd,
                        make_metric=True,
                        metric_prefix=None,
                        metric_directory=cwd,
                        targets=targets_test,
                        normalised=False
                       )