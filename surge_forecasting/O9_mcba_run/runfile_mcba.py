__author__ = "EAM"

# import system functions
import os
import sys

# import packages
import numpy as np
import scipy.io
import pickle
import matplotlib.pyplot as plt

# import tools
cwd = os.getcwd()
basuite_path = cwd + r'\..\..\bayesian_averaging_suite'
sys.path.append(basuite_path)

# import local tools
from model_selection_algorithms_streamlined import msa
from posterior_evaluation_functions_streamlined import posterior_evaluation
from load_save_functions import load_models
from plots_and_metrics import make_plots, make_metrics

# models directory
NNSdirectory = cwd + r'\..\O6_lstm_train\surge_models'

#load test data
#norm_dm = scipy.io.loadmat("..\O5_designmatrix\dm_normalised.mat")['dm_normalised'][-50000:,:,:]
dm_means = scipy.io.loadmat("..\O5_designmatrix\dm_mean.mat")['dm_mean']
dm_stdevs = scipy.io.loadmat("..\O5_designmatrix\dm_std.mat")['dm_std']

dm_2020 = scipy.io.loadmat("..\O8_mcba_visual_crossing_forecast_variables\dm_abmc_2020.mat")['dm_abmc_2020']
dm_2021 = scipy.io.loadmat("..\O8_mcba_visual_crossing_forecast_variables\dm_abmc_2021.mat")['dm_abmc_2021']
dm_2022 = scipy.io.loadmat("..\O8_mcba_visual_crossing_forecast_variables\dm_abmc_2022.mat")['dm_abmc_2022']
dm_2023 = scipy.io.loadmat("..\O8_mcba_visual_crossing_forecast_variables\dm_abmc_2023.mat")['dm_abmc_2023']

# split test and abms
norm_dm=np.concatenate([dm_2020], axis=0)
abmc_targets = norm_dm[:,-1,25:].copy()
abmc_features = norm_dm[:,:,:].copy()
abmc_features[:, -1, 25:] = 0
abmc_features = np.swapaxes(abmc_features, 1, 2)

del norm_dm

idx_1 = np.sum(np.isnan(abmc_features), axis=(1,2)) + np.sum(np.isnan(abmc_targets), axis=1)

mod_list = load_models(NNSdirectory)

# fit GMM to posterior evaluation data
#GMList, PosteriorSamples, PosteriorVariables, = posterior_evaluation(post_data=(abmc_features[idx_1==0,:,:], abmc_targets[idx_1==0,:]), 
#                                                                      posterior_variable_list=['Target'], 
#                                                                      lead_time=24, 
#                                                                      mod_list=mod_list, 
#                                                                     estimation_strategy='GaussianMixture', 
#                                                                     make_plot=True,
#                                                                     plot_title="ERA5 Posteriors ",
#                                                                     plot_prefix='ERA5 Posteriors',
#                                                                     plot_directory=cwd+'\\gmm_plots')

filename = 'GMM_target.pkl'
#pickle.dump(GMList, open(filename, 'wb'))

# load the model from disk
GMList = pickle.load(open(filename, 'rb'))

# import error stdevs and thresholds
pc_error_stds = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\pc_binned_error.mat")['pc_binned_error']
pc_error_means = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\pc_binned_mean.mat")['pc_binned_mean']
pc_error_thresholds = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\pc_binned_error_thresholds.mat")['pc_binned_error_thresholds']
#
mslp_error_stds = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\mslp_binned_error.mat")['mslp_binned_error']
mslp_error_means = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\mslp_binned_mean.mat")['mslp_binned_mean']
mslp_error_thresholds = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\mslp_binned_error_thresholds.mat")['mslp_binned_error_thresholds']
#
u10_error_stds = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\u10_binned_error.mat")['u10_binned_error']
u10_error_means = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\u10_binned_mean.mat")['u10_binned_mean']
u10_error_thresholds = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\u10_binned_error_thresholds.mat")['u10_binned_error_thresholds']
#
v10_error_stds = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\v10_binned_error.mat")['v10_binned_error']
v10_error_means = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\v10_binned_mean.mat")['v10_binned_mean']
v10_error_thresholds = scipy.io.loadmat(r"..\O8_abmc_visual_crossing_forecast_variables\O8_4_visual_crossing_pc_bias_correction\v10_binned_error_thresholds.mat")['v10_binned_error_thresholds']

# concatenate errors and thresholds
bin_error_stdevs = np.concatenate([pc_error_stds, mslp_error_stds.T, u10_error_stds.T, v10_error_stds.T], axis=1)
bin_error_means = np.concatenate([pc_error_means, mslp_error_means.T, u10_error_means.T, v10_error_means.T], axis=1)
bin_thresholds = np.concatenate([pc_error_thresholds, mslp_error_thresholds.T, u10_error_thresholds.T, v10_error_thresholds.T], axis=1)

test_vc = np.concatenate([dm_2021, dm_2022, dm_2023], axis=0)
algorithm ='MCBA'

# assign test targets and test features
targets_test = test_vc[:,-1,25:].copy()
features_test = test_vc[:,:,:].copy()
features_test[:, -1, 25:] = 0
features_test = np.swapaxes(features_test, 1, 2)  

lobo = np.zeros([len(features_test)])
lobo[:] = np.nan
robo = np.zeros([len(features_test)])
robo[:] = np.nan
upbo = np.zeros([len(features_test)])
upbo[:] = np.nan
test = np.zeros([len(features_test)])
test[:] = np.nan

step = 1000


# toc = time.time()
# elapsed_time = round(toc - tic)

for i in range(0, len(features_test) + step, step): #
    print(i)
    rp, t = msa(
                  msa=algorithm, 
                  NNSdirectory=NNSdirectory,
                  models=mod_list,
                  workingdirectory=cwd,
                  features=features_test[i:i+step,:,:],
                  means=dm_means,
                  stdevs=dm_stdevs,
                  alpha=1.282,
                  lead_time=24,
                  post_distro=GMList,
                  post_data=None,
                  estimation_strategy='GaussianMixture',
                  posterior_variable_list=['Target'],
                  targets=targets_test[i:i+step,:],
                  normalised=False,
                  mc_samples=100, 
                  feature_error_standevs=bin_error_stdevs,
                  feature_error_means=bin_error_means,
                  feature_error_bins=bin_thresholds
                )
    
    lobo[i:i+step] = rp[0]
    robo[i:i+step] = rp[1]
    upbo[i:i+step] = rp[2]
    test[i:i+step] = t

robust_prediction = (lobo, robo, upbo)
robust_save = (lobo, robo, upbo, test)
#np.savetxt('ABMC_output.csv', robust_save, delimiter=',')

#robust_prediction = np.loadtxt('ABMC_output.csv', delimiter=',')

#make_plots(robust_prediction=robust_prediction, test=test, normalised=False, lead_time=24, msa='MCBA',
#                plot_title='2021-23 Millport', plot_prefix='FOC_ABMC_', plot_directory=cwd)

make_metrics(robust_prediction=robust_prediction, test=test, normalised=False, 
              metric_prefix='FOC_ABMC_80_', metric_directory=cwd, thres1=0.75, thres2=1.25, 
              lead_time=24, msa='ABMC')

# timeseries = np.arange(0,len(robo),1)

# plt.rc('font',family='Times New Roman')
# fig=plt.figure() #[8568:-24]
# plt.plot(timeseries[15960:16160], upbo[15960:16160], "r", markersize=1, label = "_nolegend_",alpha=0.8)
# plt.plot(timeseries[15960:16160], lobo[15960:16160], "r", markersize=1, label = "95% Confidence Interval",alpha=0.8)
# plt.plot(timeseries[15960:16160], test[15960:16160], "+k", markersize=5, label = "Observations",alpha=0.7)
# plt.plot(timeseries[15960:16160], robo[15960:16160], "b", markersize=4, label = "MCBA Averaged Prediction",linewidth=2.0)
# plt.title("24 Hour Predictions", fontdict={'fontsize' : 18})
# plt.xlabel("Date", fontdict={'fontsize' : 18})
# plt.ylabel("Surge Height [m]", fontdict={'fontsize' : 18})
# plt.grid(color='r', linestyle='--', linewidth=0.5)
# plt.legend(prop = {'size' : 12})
# plt.ylim([-0.5, 1.5])
# plt.xticks(ticks=np.arange(15960, 15960+(24*8),24).tolist(), labels=['27/10/2022',
#                                                                   '29/10/2022',
#                                                                   '30/10/2022',
#                                                                   '31/10/2022',
#                                                                   '01/11/2022',
#                                                                   '02/11/2022',
#                                                                   '03/11/2022',
#                                                                   '04/11/2022'],
#                                                                   rotation=45)

# plt.xlim([15960,16128])
# plt.tight_layout()
# plt.show()
# fname="timeseries_2022.png"
# fig.savefig(fname, 
#         dpi=500, format=None, metadata=None,
#         bbox_inches=None, pad_inches=0.1,
#         facecolor='auto', edgecolor='auto',
#         backend=None
#         )

# plt.rc('font',family='Times New Roman')
# fig=plt.figure() 
# plt.plot(timeseries[23784:23952], upbo[23784:23952], "r", markersize=1, label = "_nolegend_",alpha=0.8)
# plt.plot(timeseries[23784:23952], lobo[23784:23952], "r", markersize=1, label = "95% Confidence Interval",alpha=0.8)
# plt.plot(timeseries[23784:23952], test[23784:23952], "+k", markersize=5, label = "Observations",alpha=0.7)
# plt.plot(timeseries[23784:23952], robo[23784:23952], "b", markersize=4, label = "MCBA Averaged Prediction",linewidth=2.0)
# plt.title("24 Hour Predictions", fontdict={'fontsize' : 18})
# plt.xlabel("Date", fontdict={'fontsize' : 18})
# plt.ylabel("Surge Height [m]", fontdict={'fontsize' : 18})
# plt.grid(color='r', linestyle='--', linewidth=0.5)
# plt.legend(prop = {'size' : 12})
# plt.ylim([-0.5, 1.5])
# plt.xticks(ticks=np.arange(23784, 23952+24, 24).tolist(), labels=['18/09/2023',
#                                                               '19/09/2023',
#                                                               '20/09/2023',
#                                                               '21/09/2023',
#                                                               '22/09/2023',
#                                                               '23/09/2023',
#                                                               '24/09/2023',
#                                                               '25/09/2023'],
#                                                                 rotation=45)
# plt.xlim([23784,23952])
# plt.tight_layout()
# plt.show()
# fname="timeseries_2023.png"
# fig.savefig(fname, 
#         dpi=500, format=None, metadata=None,
#         bbox_inches=None, pad_inches=0.1,
#         facecolor='auto', edgecolor='auto',
#         backend=None
#         )
