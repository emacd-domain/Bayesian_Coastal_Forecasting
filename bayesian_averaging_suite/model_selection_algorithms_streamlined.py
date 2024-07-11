__author__ = "EM"

# import system functions
import os
import sys
from time import sleep
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import packages
from numpy import (zeros, round, arange, 
                   squeeze, mean, sqrt, 
                   interp, argmax, tile, transpose, sum, expand_dims, 
                   argsort, nan, repeat,  
                   reshape, digitize, random, indices, isnan, 
                   invert, nanmax, nanmin)
from tqdm import tqdm
# import function tools
from load_save_functions import load_models
from posterior_evaluation_functions_streamlined import posterior_evaluation


def initialise_evaluation_arrays(features, models):
    ### INITIALISE VARIABLES TO BE CALCULATED WITHIN LOOP
    NetP = zeros([len(features), len(models)])
    PostIntOutput = zeros([len(features), len(models)])
    ROBO = zeros([len(features)])
    LOBO = zeros([len(features)])
    UPBO = zeros([len(features)])
    return [ROBO, LOBO, UPBO, NetP, PostIntOutput]

def make_robust_predictions(algorithm, lead_time, test_features, posterior_variable_list, PosteriorVariables, mod_list, prediction_loop_vars, alpha, estimation_strategy, PosteriorSamples=None, GMList=None, mc_samples=None, feature_error_standevs=None, feature_error_means=None, feature_error_bins=None):
    
    ROBO = zeros([len(test_features), len(posterior_variable_list)])
    ROBO[:] = nan
    VAR = zeros([len(test_features), len(posterior_variable_list)])
    VAR[:] = nan
    CERT = zeros([len(test_features), len(posterior_variable_list)])
    CERT[:] = nan

    idx_nan = isnan(test_features).any(axis=(1,2))
    idx_real = invert(idx_nan)
    if sum(idx_real) == 0:
        Weighted_ROBO = zeros([len(test_features)])
        Weighted_ROBO[:] = nan
        Weighted_UPBO = zeros([len(test_features)])
        Weighted_UPBO[:] = nan
        Weighted_LOBO = zeros([len(test_features)])
        Weighted_LOBO[:] = nan
    else:
        DiVariable = []
        for variable in posterior_variable_list:
            if variable == 'Target':
                target = []
                DiVariable.append(target)
            if variable == 'PC1':
                pc1 = test_features[idx_real, lead_time-1, 0]
                DiVariable.append(pc1)
            if variable == 'PC2':
                pc2 = test_features[idx_real, lead_time-1, 1]
                DiVariable.append(pc2)
            if variable == 'PC3':
                pc3 = test_features[idx_real, lead_time-1, 2]
                DiVariable.append(pc3)            
            if variable == 'Local Pressure':
                mslp = test_features[idx_real, lead_time-1, 8]
                DiVariable.append(mslp)
        
        NetP = zeros([sum(idx_real), len(mod_list)])
        PostIntOutput = zeros([sum(idx_real), len(mod_list)])
        
        if algorithm == 'ABMS':
            
            ''' make predictions '''
            for i in range(0, len(mod_list), 1):
                model_set = mod_list[i]
                PostIntOutput[:, i] = squeeze(model_set.predict(test_features[idx_real,:,:], verbose=0))[:, lead_time-1]
                
            for var_count in range(0, len(posterior_variable_list), 1):
                posterior_variable = posterior_variable_list[var_count]
                Variable = DiVariable[var_count]
                if estimation_strategy=='Interpolation':
                    PosteriorSample = PosteriorSamples[var_count]
                    PosteriorVariable = PosteriorVariables[var_count]
                    calunix = argsort(PosteriorVariable)
                else:
                    GMMSet = GMList[var_count]
                
                for i in range(0, PostIntOutput.shape[1], 1):
                    if (estimation_strategy == 'Interpolation') & (posterior_variable == 'Target'):
                        NetP[:, i] = interp(PostIntOutput[:, i], PosteriorVariable[calunix], PosteriorSample[calunix, i])
                    elif (estimation_strategy == 'Interpolation') & (posterior_variable != 'Target'):
                        NetP[:, i] = interp(Variable, PosteriorVariable[calunix], PosteriorSample[calunix, i])
                    elif (estimation_strategy == 'GaussianMixture') & (posterior_variable == 'Target'):
                        NetP[:, i] = sum(GMMSet[i][0].predict_proba(PostIntOutput[:, i].reshape(-1,1))*GMMSet[i][0].weights_/GMMSet[i][1], axis=1)
                    elif (estimation_strategy == 'GaussianMixture') & (posterior_variable != 'Target'):
                        NetP[:, i] = sum(GMMSet[i][0].predict_proba(Variable.reshape(-1,1))*GMMSet[i][0].weights_/GMMSet[i][1], axis=1)
                    elif (estimation_strategy == 'Gaussian') & (posterior_variable == 'Target'):
                        NetP[:, i] = sum(GMMSet[i][0].predict_proba(PostIntOutput[:, i].reshape(-1,1))*GMMSet[i][0].weights_/GMMSet[i][1], axis=1)
                    elif (estimation_strategy == 'Gaussian') & (posterior_variable != 'Target'):
                        NetP[:, i] = sum(GMMSet[i][0].predict_proba(Variable.reshape(-1,1))*GMMSet[i][0].weights_/GMMSet[i][1], axis=1)
                
                Certainty = sum(NetP[:, :], axis=1)
                CERT[idx_real, var_count] = Certainty
                NetP[:, :] = NetP[:, :] / transpose(tile(Certainty, [len(mod_list), 1]))
                BestNet = argmax(NetP[:, :], axis=1)
                piOutput = zeros([sum(idx_real)])
                for q in range(0, len(BestNet), 1):
                    piOutput[q] = PostIntOutput[q, BestNet[q]]
        
                af_MU = sum(NetP * (PostIntOutput - transpose(tile(piOutput, [len(mod_list), 1]))), axis=1)
    
                var_ROBO = piOutput + af_MU
                
                af_VAR = sum(NetP * (PostIntOutput - transpose(tile(var_ROBO, [len(mod_list), 1]))) ** 2, axis=1)
                
                ROBO[idx_real, var_count] = var_ROBO
                VAR[idx_real, var_count] = af_VAR
            
            SumCERT = sum(CERT, axis=1)
            WeightedCERT = CERT / transpose(tile(SumCERT, [len(posterior_variable_list), 1]))
            Weighted_ROBO = sum(ROBO * WeightedCERT, axis=1)
            Weighted_VAR = sum(VAR * WeightedCERT, axis=1)
            Weighted_UPBO = Weighted_ROBO + (alpha * sqrt(Weighted_VAR))
            Weighted_LOBO = Weighted_ROBO - (alpha * sqrt(Weighted_VAR))
        
        if algorithm == 'ABMC':
            
            ''' RESHAPE INPUTS '''
            rep_test_features = repeat(expand_dims(test_features[idx_real,:,:], 0), mc_samples, axis=0)
            stacked_rep_test_features = reshape(rep_test_features, (rep_test_features.shape[0]*rep_test_features.shape[1], rep_test_features.shape[2], rep_test_features.shape[3]), order='F')
            del rep_test_features
            
            '''  GENERATE ERRORS BASED ON FEATURE VALUES AND APPLY TO EACH SAMPLE IN BATCH'''
            cast_standev = zeros([stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]])
            cast_standev[:] = nan
            cast_means = zeros([stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]])
            cast_means[:] = nan

            for m in range(0, feature_error_standevs.shape[1], 1):
               # digitize using error bins
               bindex = digitize(stacked_rep_test_features[:, :, m], feature_error_bins[:, m], right=False)-1
               # assign standevs to digitized values
               for i in range(0, len(feature_error_bins)-1, 1):
                   cast_standev[bindex == i] = feature_error_standevs[i, m]
                   cast_means[bindex == i] = feature_error_means[i, m]
               # gen noise using standevs expand_dims(evbest, axis=-1)
               rn = random.normal(cast_means, cast_standev, [stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]]) #repeat(expand_dims(,-1), 49, -1) #[stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]]
               #rn[:, 25:] = repeat(expand_dims(rn[:,24], -1), 24, -1)
               stacked_rep_test_features[:, :, m] += rn #[:, 24:]
            #surge_uncert = random.uniform(-0.25, 0.25, 1)
            #stacked_rep_test_features[:,24,-1] += surge_uncert 
            PostIntOutput_MC = repeat(PostIntOutput, mc_samples, axis=0)
            ''' MAKE PREDICTIONS ''' 
            for i in range(0, len(mod_list), 1):
                model_set = mod_list[i]
                PostIntOutput_MC[:, i] = squeeze(model_set.predict(stacked_rep_test_features, verbose=0)[:, lead_time-1])
            
            # remove stacked features to preserve RAM
            del stacked_rep_test_features
            
            for var_count in range(0, len(posterior_variable_list), 1):
    
                NetP_MC = repeat(NetP, mc_samples, axis=0)
                posterior_variable = posterior_variable_list[var_count]
                Variable = DiVariable[var_count]
                Variable_MC = repeat(Variable, mc_samples, axis=0)
                
                if estimation_strategy=='Interpolation':
                    PosteriorVariable = PosteriorVariables[var_count]
                    PosteriorSample = PosteriorSamples[var_count]
                    calunix = argsort(PosteriorVariable)
                else:
                    GMMSet = GMList[var_count]
                
                ''' REMAKE LOOP VARIABLE ARRAYS SO THEY ARE THE SAME DIMENSIONS MULTIPLIED BY mc_samples and reshape into input dimensions '''
                
                for i in range(0, PostIntOutput.shape[1], 1):
                    if (estimation_strategy == 'Interpolation') & (posterior_variable == 'Target'):
                        NetP_MC[:, i] = interp(PostIntOutput_MC[:, i], PosteriorVariable[calunix], PosteriorSample[calunix, i])
                    elif (estimation_strategy == 'Interpolation') & (posterior_variable != 'Target'):
                        NetP_MC[:, i] = interp(Variable_MC, PosteriorVariable[calunix], PosteriorSample[calunix, i])
                    elif ((estimation_strategy == 'Gaussian') | (estimation_strategy == 'GaussianMixture')) & (posterior_variable == 'Target'):
                        NetP_MC[:, i] = sum(GMMSet[i][0].predict_proba(PostIntOutput_MC[:, i].reshape(-1,1))*GMMSet[i][0].weights_/GMMSet[i][1], axis=1)
                    elif ((estimation_strategy == 'Gaussian') | (estimation_strategy == 'GaussianMixture')) & (posterior_variable != 'Target'):
                        NetP_MC[:, i] = sum(GMMSet[i][0].predict_proba(Variable_MC.reshape(-1,1))*GMMSet[i][0].weights_/GMMSet[i][1], axis=1)
                
                ''' RESHAPE TO ORIGINAL DIMENSIONS '''
                unstacked_NetP_MC = reshape(NetP_MC, (mc_samples, int(NetP_MC.shape[0]/mc_samples), NetP_MC.shape[1]), order='F')
                unstacked_PostIntOutput_MC = reshape(PostIntOutput_MC, (mc_samples, int(PostIntOutput_MC.shape[0]/mc_samples), PostIntOutput_MC.shape[1]), order='F')
                # reduce memory burden
                #del PostIntOutput_MC
                ''' REAVERAGE INTO STANDARD OUTPUT FORMAT '''
                
                # sum certainties to 1 over each sample of each observation
                Certainty = sum(unstacked_NetP_MC[:,:], axis=2)
                unstacked_NetP_MC[:,:]=unstacked_NetP_MC[:,:]/repeat(expand_dims(Certainty, -1), len(mod_list), axis=-1)
                # place certainty for liklihood variable
                CERT[idx_real, var_count] = sum(Certainty, axis=0)
                # find the best network per sample
                BestNet = argmax(unstacked_NetP_MC[:,:], axis=2)
                # get best net for each observation and sample using indicies
                indy = indices(BestNet.shape)
                piOutput = unstacked_PostIntOutput_MC[indy[0], indy[1], BestNet]
                # calculate adjustment factor for each sample and observation
                af_MU = sum(unstacked_NetP_MC*(unstacked_PostIntOutput_MC-repeat(expand_dims(piOutput, -1), len(mod_list), axis=-1)), axis=2)
                # calculate mean predictions of all samples
                samprobo = piOutput + af_MU
                # average across all mc samples
                evbest = mean(samprobo, axis=0)
                rep_evbest = repeat(expand_dims(evbest, axis=-1), len(mod_list), axis=-1)
                restruct_evbest = repeat(expand_dims(rep_evbest, axis=0), mc_samples, axis=0)
                # calculate the variance of deviation from adjusted prediction
                af_VAR = sum((unstacked_NetP_MC/mc_samples)*(unstacked_PostIntOutput_MC-restruct_evbest)**2, axis=(0,2))
    
                ROBO[idx_real, var_count] = evbest
                VAR[idx_real, var_count] = af_VAR
            
            SumCERT = sum(CERT, axis=1)
            WeightedCERT = CERT / transpose(tile(SumCERT, [len(posterior_variable_list), 1]))
            Weighted_ROBO = sum(ROBO * WeightedCERT, axis=1)
            Weighted_VAR = sum(VAR * WeightedCERT, axis=1)
            Weighted_UPBO = Weighted_ROBO + (alpha * sqrt(Weighted_VAR))
            Weighted_LOBO = Weighted_ROBO - (alpha * sqrt(Weighted_VAR))

    return (Weighted_LOBO, Weighted_ROBO, Weighted_UPBO)


def msa(
        msa=None,
        NNSdirectory=None,
        models=None,
        workingdirectory=None,
        features=None, 
        means=None, 
        stdevs=None, 
        alpha=None,
        lead_time=None,
        post_distro=None, 
        post_data=None, 
        estimation_strategy=None, 
        posterior_variable_list=None,
        make_plot=False,
        plot_title=None,
        plot_prefix=None,
        plot_directory=None,
        make_metric=False,
        metric_prefix=None,
        metric_directory=None,
        targets=None,
        normalised=False,
        mc_samples=None, 
        feature_error_standevs=None,
        feature_error_means=None,
        feature_error_bins=None
       ):
    
    if msa is None:
        sys.exit()
        
    if NNSdirectory is None:
        sys.exit()
    
    if features is None:
        sys.exit()
        
    if means is None:
        sys.exit()

    if stdevs is None:
        sys.exit()

    if alpha is None:
        alpha = 1.96

    if lead_time is None:
        lead_time = 24

    if estimation_strategy is None:
        estimation_strategy = 'interpolate'

    if post_distro is not None and post_data is None:
        GMList = post_distro
    elif post_distro is None and post_data is None:
        sys.exit()
    elif post_distro is not None and post_data is not None:
        post_data = None
        GMList = post_distro

    if (mc_samples is None) & (msa == 'ABMC'):
        mc_samples = 25        
        
    if (feature_error_bins is None) & (msa == 'ABMC'):
        sys.exit()
    
    if (models is None) & (NNSdirectory is not None):
        mod_list = load_models(NNSdirectory)
    elif (models is not None):
        mod_list = models.copy()
        del models

    if post_distro is None:
        GMList, PosteriorSamples, PosteriorVariables = posterior_evaluation(post_data=post_data, posterior_variable_list=posterior_variable_list, lead_time=lead_time, mod_list=mod_list, estimation_strategy=estimation_strategy)
    else:
        PosteriorSamples = []
        PosteriorVariables=[]
        
    prediction_loop_vars = initialise_evaluation_arrays(features, mod_list)

    if msa == 'ABMS':
        robust_prediction = make_robust_predictions(
                                                    algorithm='ABMS',
                                                    lead_time=lead_time,
                                                    test_features=features,
                                                    posterior_variable_list=posterior_variable_list,
                                                    PosteriorVariables=PosteriorVariables,
                                                    mod_list=mod_list,
                                                    prediction_loop_vars=prediction_loop_vars,
                                                    alpha=alpha,
                                                    estimation_strategy=estimation_strategy,
                                                    PosteriorSamples = PosteriorSamples,
                                                    GMList=GMList
                                                   )
    elif msa == 'ABMC':
        robust_prediction = make_robust_predictions(
                                                    algorithm='ABMC',
                                                    lead_time=lead_time,
                                                    test_features=features,
                                                    posterior_variable_list=posterior_variable_list,
                                                    PosteriorVariables=PosteriorVariables,
                                                    mod_list=mod_list,
                                                    prediction_loop_vars=prediction_loop_vars,
                                                    alpha=alpha,
                                                    estimation_strategy=estimation_strategy,
                                                    PosteriorSamples = PosteriorSamples,
                                                    GMList=GMList,
                                                    mc_samples=mc_samples,
                                                    feature_error_standevs=feature_error_standevs,
                                                    feature_error_means=feature_error_means,
                                                    feature_error_bins=feature_error_bins
                                                   )
        
    test = targets[:, lead_time-1]

    if not normalised:
        
        robust_prediction = ((robust_prediction[0] * stdevs[0, -1]) + means[0, -1],
                             (robust_prediction[1] * stdevs[0, -1]) + means[0, -1],
                             (robust_prediction[2] * stdevs[0, -1]) + means[0, -1])
        test = (test * stdevs[0, -1]) + means[0, -1]
        
    
    return robust_prediction, test




