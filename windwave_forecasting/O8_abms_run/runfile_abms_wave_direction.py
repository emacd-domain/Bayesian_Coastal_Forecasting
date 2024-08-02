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

target_id_EW = -2
target_id_NS = -1

n_feat = 44

# split test and abms 

targets_abms_EW = dm[:-40000,target_id_EW,:].copy()
targets_abms_NS = dm[:-40000,target_id_NS,:].copy()

features_abms = dm[:-40000,:n_feat,:].copy()
features_abms = np.swapaxes(features_abms, 1, 2)

targets_test_EW = dm[-40000:,target_id_EW,:].copy()
targets_test_NS = dm[-40000:,target_id_NS,:].copy()
features_test = dm[-40000:,:44,:].copy()
features_test = np.swapaxes(features_test, 1, 2)

targets_test_ext = (dm[-40000:, -4, lead_time].copy()*dm_stdevs[0,-4])+dm_means[0,-4]

# EAST/WEST

NNSdirectory = cwd+r'\..\O7_lstm_train\wave_direction_EW_models'

robust_prediction_EW = BMF(
                           msa='ABMS',
                           NNSdirectory=NNSdirectory,
                           workingdirectory=cwd,
                           features=features_test,
                           means=dm_means[0, target_id_EW],
                           stdevs=dm_stdevs[0, target_id_EW],
                           alpha=1.96,
                           lead_time=lead_time+1,
                           post_distro=None,
                           post_data=(features_abms, targets_abms_EW),
                           estimation_strategy='GaussianMixture',
                           posterior_variable_list=['Target'],
                           normalised=False
                          )

test_EW = (targets_test_EW[:, -1]*dm_stdevs[0, target_id_EW])+dm_means[0, target_id_EW]

# NORTH/SOUTH

NNSdirectory = cwd+r'\..\O7_lstm_train\wave_direction_NS_models'

robust_prediction_NS = BMF(
                           msa='ABMS',
                           NNSdirectory=NNSdirectory,
                           workingdirectory=cwd,
                           features=features_test,
                           means=dm_means[0, target_id_NS],
                           stdevs=dm_stdevs[0, target_id_NS],
                           alpha=1.96,
                           lead_time=lead_time+1,
                           post_distro=None,
                           post_data=(features_abms, targets_abms_NS),
                           estimation_strategy='GaussianMixture',
                           posterior_variable_list=['Target'],
                           normalised=False
                          )

test_NS = (targets_test_NS[:, -1]*dm_stdevs[0, target_id_NS])+dm_means[0, target_id_NS]

# COMBINE UNCERTAINTY
bound = np.zeros([len(test_NS), 4])

# calculate wind to
wind_dir_to_test = np.arctan2(test_EW, test_NS)*180/np.pi 
wind_dir_from_test = np.mod(wind_dir_to_test + 180, 360)

wind_dir_to = np.arctan2(robust_prediction_EW[1], robust_prediction_NS[1])*180/np.pi 

bounds = np.zeros([len(wind_dir_to), 2])
bounds[:,0] = (np.arctan2(robust_prediction_EW[2], robust_prediction_NS[0])*180/np.pi) # - wind_dir_to
bounds[:,1] = (np.arctan2(robust_prediction_EW[0], robust_prediction_NS[2])*180/np.pi) # - wind_dir_to

wind_from = np.mod(wind_dir_to + 180, 360)
bounds_wind_from = np.mod(bounds + 180, 360)

width = np.max(bounds_wind_from, 1) - wind_from
index_1 = (width) > 180

ub = wind_from + width
lb = wind_from - width

robust_prediction = (lb, wind_from, ub)

robust_save = (robust_prediction[0], robust_prediction[1], robust_prediction[2], wind_dir_from_test)

np.savetxt('ABMS_wave_direction_output.csv', robust_save, delimiter=',')

make_plots(variable='Mean Wave Direction', 
           robust_prediction=robust_prediction, 
           test=wind_dir_from_test, 
           normalised=False, 
           lead_time=lead_time, 
           msa='ABMS',
           plot_title='CMEMS Test', 
           plot_prefix='FOC_', 
           plot_directory=cwd,
           ext_wave=targets_test_ext,
           height_filter=0.05)

make_metrics(variable='Mean Wave Period', 
             robust_prediction=robust_prediction, 
             test=wind_dir_from_test, 
             normalised=False, 
             metric_prefix='FOC_CONF95_', 
             metric_directory=cwd, 
             thres1=2.0, thres2=2.5, 
             lead_time=lead_time, msa='ABMS',
             ext_wave=targets_test_ext,
             height_filter=0.05)


