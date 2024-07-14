__author__ = "EM"


'''

model_selection_algorithms.py contains the model selection algorithms for 
the Bayesian Coastal Forecasting process.

Primary function is BMF (Bayesian Model Forecast) 

INPUTS:

    msa                    : model selection algorithm. 
                             select either 'ABMS' or 'MCBA'
    streamlined            : fast option that supresses all console outputs
                             True or False. When setting up the model, 
                             streamline == False is advised so that input 
                             issues can easily be identified 
    NNSdirectory           : directory of tensorflow models, can be 'None' if 
                             user defines <models> input
    models                 : list of tensorflow models, can be 'None' if user
                             defines <NNSdirectory> input
    workingdirectory       : directory of run script
    
    features               : 3D normalised input array with dimensions 
                             [observations, timesteps, features] 
    means                  : normalisation means for real scaled outputs if 
                             <normalised> == False 
    stdevs                 : normalisation standard deviations for real scaled 
                             outputs if <normalised> == False 
    alpha                  : z-score for confidence interval, 
                             default = 1.96 for 95%
    lead_time              : leadtime of surge forecast, must be integer value 
                             in range 1 to 24 hours, default 24
    post_distro            : list of sklearn Gaussian Models, can be None if 
                             user provides <post_data> 
    post_data              : tuple of posterior probability evaluation data in
                             format (features, targets), can be None if user
                             provides <post_distro> input
    estimation_strategy    : method for inferring posterior probabilities for 
                             predictions. select either 
                             'Interpolation' to interpolate between <post_data> 
                             'Gaussian' to predict from Guassian model provided
                             or fitted to <post_data>
                             'GaussianMixture' to predict from Gaussian Mixture
                             Model provided or fitted to <post_data>
    posterior_variable_list: Experimental Option. Remain as 'Target'
    
    normalised             : real scaled outputs if False
                             normalised outputs if True
    mc_samples             : Number of MonteCarlo samples for MCBA method
                             default is 25 samples
    feature_error_standevs : forecast error standard deviations, must be 
                             provided if <msa> == 'MCBA'
    feature_error_means    : forecast error means, must be 
                             provided if <msa> == 'MCBA'
    feature_error_bins     : forecast error percentile bins, must be 
                             provided if <msa> == 'MCBA'

OUTPUTS:
    
    robust_prediction = (lower bound, averaged prediction, upper bound)
    
BMF uses sub-processses:
    
    initialise_evaluation_arrays
    make_robust_predictions
    posterior_evaluation_functions\posterior_evaluation

'''

# import system functions
import os
import sys
from time import sleep
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import numpy packages
from numpy import (zeros, squeeze, mean, sqrt, interp, argmax, tile, transpose,
                   sum, expand_dims, argsort, nan, repeat, reshape, digitize, 
                   random, indices, isnan, invert)
from tqdm import tqdm

# import function tools
from load_save_functions import load_models
from posterior_evaluation_functions import posterior_evaluation


def BMF(
        msa=None,
        streamlined=False,
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
        normalised=False,
        mc_samples=None, 
        feature_error_standevs=None,
        feature_error_means=None,
        feature_error_bins=None
       ):
    
    # state model selection algorithm
    if (msa is None) & (streamlined == False):
        print("\nPlease select ABMS or MCBA algorithm")
        sys.exit()
    elif (msa == 'ABMS') & (streamlined == False):
        print("\n")
        print("###############################################################")
        print("######### ADAPTIVE BAYESIAN MODEL SELECTION ALGORITHM #########")
        print("###############################################################")
        sleep(0.5)
    elif (msa == 'MCBA') & (streamlined == False):
        print("\n")
        print("###########################################################")
        print("######### MONTECARLO BAYESIAN AVERAGING ALGORITHM #########")
        print("###########################################################")
    elif (msa is None) & (streamlined == True):
        sys.exit()
        
    # state model set
    if (NNSdirectory is None) & (models is None) & (streamlined == False):
        print("\nNo Model Directory or Models Provided")
        sleep(0.5)
        sys.exit()
    elif (NNSdirectory is None) & (models is None) & (streamlined == True):
        sys.exit()
    elif (models is None) & (NNSdirectory is not None):
        mod_list = load_models(NNSdirectory)
    elif (models is not None):
        mod_list = models.copy()
        del models
    
    # state features
    if (features is None) & (streamlined == False):
        print("\nNo Data Provided!")
        sleep(0.5)
        sys.exit()
    elif (features is None) & (streamlined == True):
        sys.exit()
    elif (features is not None) & (streamlined == False):
        print(f"\nfound feature input with dimensions {features.shape}")
        sleep(0.5)

    # state feature means        
    if (means is None) & (normalised is False) & (streamlined == False):
        print("No Means Provided for Features, either provide means or set normalised to True")
        sleep(0.5)
        sys.exit()
    elif (means is None) & (normalised is False) & (streamlined == True):
        sys.exit()
    elif (means is not None) & (normalised is False) & (streamlined == False):
        print(f"found means with dimensions {means.shape}")
        sleep(0.5)

    # state feature standard deviations
    if (stdevs is None) & (normalised is False) & (streamlined == False):
        print("No Standard Deviations Provided for Features, either provide Standard Deviations or set normalised to True")
        sleep(0.5)
        sys.exit()
    elif (stdevs is None) & (normalised is False) & (streamlined == True):
        sys.exit()
    elif (stdevs is not None) & (normalised is False) & (streamlined == False):
        print(f"found standard deviations with dimensions {stdevs.shape}")
        sleep(0.5)

    # set p-value for confidence interval
    if (alpha is None) & (streamlined == False):
        print("\nNo alpha provided, 95% confidence interval alpha=1.96 is assumed")
        sleep(0.5)
        alpha = 1.96
    elif (alpha is None) & (streamlined == True):
        alpha = 1.96

    # set leadtime for evaluations
    if (lead_time is None) & (streamlined == False):
        print("\nNo lead time provided, 24 hour lead time is assumed")
        sleep(0.5)
        lead_time = 24
    if (lead_time is None) & (streamlined == True):
        lead_time = 24
    #elif len(lead_time) > 1:
    #    print('\nPlease provide a single lead time')

    # Check Inference Model or Data Provided
    if (post_distro is not None) & (post_data is None) & (streamlined == False):
        GMList = post_distro
        print(f"\n{len(GMList[0])} posterior distributions provided")
        for i in range (0, len(GMList[0]), 1):
            print(f'distribution {i+1}: {GMList[0][i][0]}')
    elif (post_distro is not None) & (post_data is None) & (streamlined == True):
        GMList = post_distro
    elif (post_distro is None) & (post_data is None) & (streamlined == False):
        print("\nPlease provide model selection distribution or data to construct model selection distribution")
        sleep(0.5)
        sys.exit()
    elif (post_distro is None) & (post_data is None) & (streamlined == True):
        sys.exit()
    elif (post_distro is not None) & (post_data is not None) & (streamlined == False):
        print("Both model selection distribution and data provided, \n abms method will use selection distribution")
        sleep(0.5)
        post_data = None
        GMList = post_distro
    elif (post_distro is not None) & (post_data is not None) & (streamlined == True):
        post_data = None
        GMList = post_distro
    elif (post_distro is None) & (post_data is not None) & (streamlined == False):
        print(f"found likelihood evaluation data with feature dimensions {post_data[0].shape} and target dimensions {post_data[1].shape}")
        sleep(0.5)

    # Posterior Inference Strategy
    if (estimation_strategy is None) & (post_data is not None) & (post_distro is None) & (streamlined == False):
        print("\n No estimation strategy provided, interpolation is assumed")
        sleep(0.5)
        estimation_strategy = 'interpolate'
    elif (estimation_strategy is None) & (post_data is not None) & (post_distro is None) & (streamlined == True):
        estimation_strategy = 'interpolate'
    elif (estimation_strategy is None) & (post_data is None) & (post_distro is not None) & (streamlined == False):
        print("\n No estimation strategy provided, Gaussian Mixture is assumed is assumed")
        sleep(0.5)
        estimation_strategy = 'GaussianMixture'
    elif (estimation_strategy is None) & (post_data is None) & (post_distro is not None) & (streamlined == True):
        estimation_strategy = 'GaussianMixture'
    elif (estimation_strategy is not None) & (streamlined == False):
        print(f"\n {estimation_strategy} strategy selected")
        sleep(0.5)

    # check mc_samples
    if (mc_samples is None) & (msa == 'MCBA') & (streamlined == False):
        mc_samples = 25
        print(f"\nmc_samples not defined, setting default value {mc_samples}")
        sleep(0.5)
    elif (mc_samples is not None) & (msa == 'MCBA') & (streamlined == False):
        print(f"\nmc_samples set at {mc_samples}")
        sleep(0.5)
    elif (mc_samples is None) & (msa == 'MCBA') & (streamlined == True):
        mc_samples = 25
        
    # check error means
    if (feature_error_means is None) & (msa == 'MCBA') & (streamlined == False):
        print("\nPlease provide feature error means or select ABMS if none are available")
        sleep(0.5)
        sys.exit()
    elif (feature_error_means is None) & (msa == 'MCBA') & (streamlined == True):
        sys.exit()
    elif (feature_error_means is not None) & (msa == 'MCBA') & (streamlined == False):
        print(f"found feature error means with dimensions {feature_error_means.shape}")
        sleep(0.5)
        
    # check error standard deviations
    if (feature_error_standevs is None) & (msa == 'MCBA') & (streamlined == False):
        print("\nPlease provide feature error standard deviations or select ABMS if none are available")
        sleep(0.5)
        sys.exit()
    elif (feature_error_standevs is None) & (msa == 'MCBA') & (streamlined == True):
        sys.exit()
    elif (feature_error_standevs is not None) & (msa == 'MCBA') & (streamlined == False):
        print(f"found feature error standard deviations with dimensions {feature_error_standevs.shape}")
        sleep(0.5)
    
    # check error percentile bins
    if (feature_error_bins is None) & (msa == 'MCBA') & (streamlined == False):
        print("\nPlease provide feature error percentile bins or select ABMS if none are available")
        sleep(0.5)
        sys.exit()
    elif (feature_error_bins is None) & (msa == 'MCBA') & (streamlined == True):
        sys.exit()
    elif (feature_error_bins is not None) & (msa == 'MCBA') & (streamlined == False):
        print(f"found feature error percentile bins with dimensions {feature_error_bins.shape}")
        sleep(0.5)
    
    # enable posterior inference strategy
    if post_distro is None:
        GMList, PosteriorSamples, PosteriorVariables = posterior_evaluation(streamlined=streamlined, post_data=post_data, posterior_variable_list=posterior_variable_list, lead_time=lead_time, mod_list=mod_list, estimation_strategy=estimation_strategy)
    else:
        PosteriorSamples = []
        PosteriorVariables=[]
    
    # initialise arrays
    prediction_loop_vars = initialise_evaluation_arrays(features, mod_list)

    if msa == 'ABMS':
        robust_prediction = make_robust_predictions(
                                                    streamlined=streamlined,
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
    elif msa == 'MCBA':
        robust_prediction = make_robust_predictions(
                                                    streamlined=streamlined,
                                                    algorithm='MCBA',
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

    if not normalised:
        
        robust_prediction = ((robust_prediction[0] * stdevs[0, -1]) + means[0, -1],
                             (robust_prediction[1] * stdevs[0, -1]) + means[0, -1],
                             (robust_prediction[2] * stdevs[0, -1]) + means[0, -1])
    
    return robust_prediction


def initialise_evaluation_arrays(features, models):
    ### INITIALISE VARIABLES TO BE CALCULATED WITHIN LOOP
    NetP = zeros([len(features), len(models)])
    PostIntOutput = zeros([len(features), len(models)])
    ROBO = zeros([len(features)])
    LOBO = zeros([len(features)])
    UPBO = zeros([len(features)])
    return [ROBO, LOBO, UPBO, NetP, PostIntOutput]

def make_robust_predictions(streamlined, algorithm, lead_time, test_features, posterior_variable_list, PosteriorVariables, mod_list, prediction_loop_vars, alpha, estimation_strategy, PosteriorSamples=None, GMList=None, mc_samples=None, feature_error_standevs=None, feature_error_bins=None):
    
    ROBO = zeros([len(test_features), len(posterior_variable_list)])
    ROBO[:] = nan
    VAR = zeros([len(test_features), len(posterior_variable_list)])
    VAR[:] = nan
    CERT = zeros([len(test_features), len(posterior_variable_list)])
    CERT[:] = nan

    idx_nan = isnan(test_features).any(axis=(1,2))
    idx_real = invert(idx_nan)

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
        if streamlined == False:
            """ make ABMS robust predictions using posterior data """
            print("\n######################################################################")
            print("\nMake ABMS Robust Prediction Using "+estimation_strategy+" Estimation Strategy")
        
            ''' make predictions '''
            print("Making Network Predictions")
            for i in tqdm(range(0, len(mod_list), 1)):
                model_set = mod_list[i]
                PostIntOutput[:, i] = squeeze(model_set.predict(test_features[idx_real,:,:], verbose=0))[:, lead_time-1]
        else:
            for i in range(0, len(mod_list), 1):
                model_set = mod_list[i]
                PostIntOutput[:, i] = squeeze(model_set.predict(test_features[idx_real,:,:], verbose=0))[:, lead_time-1]            
        for var_count in range(0, len(posterior_variable_list), 1):
            if streamlined == False:
                print(f'\nMaking Averaged Predictions Using {posterior_variable_list[var_count]} Posteriorss')
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
                elif ((estimation_strategy == 'Gaussian') | (estimation_strategy == 'GaussianMixture')) & (posterior_variable == 'Target'):
                    NetP[:, i] = sum(GMMSet[i][0].predict_proba(PostIntOutput[:, i].reshape(-1,1))*GMMSet[i][0].weights_/GMMSet[i][1], axis=1)
                elif ((estimation_strategy == 'Gaussian') | (estimation_strategy == 'GaussianMixture')) & (posterior_variable != 'Target'):
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
    
    if algorithm == 'MCBA':
        
        if streamlined == False:
            """ make MCBA robust predictions using posterior data """
            print("\n######################################################################")
            print("\nMCBA Robust Prediction Using "+estimation_strategy+" Estimation Strategy")
        
        ''' RESHAPE INPUTS '''
        rep_test_features = repeat(expand_dims(test_features[idx_real,:,:], 0), mc_samples, axis=0)
        if streamlined == False:
            print('\nreshape features to accommodate MC method')
        stacked_rep_test_features = reshape(rep_test_features, (rep_test_features.shape[0]*rep_test_features.shape[1], rep_test_features.shape[2], rep_test_features.shape[3]), order='F')
        del rep_test_features
        
        '''  GENERATE ERRORS BASED ON FEATURE VALUES AND APPLY TO EACH SAMPLE IN BATCH'''
        if streamlined == False:
            print('\napplying errors to features:')
            sleep(0.5)
        cast_standev = zeros([stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]])
        cast_standev[:] = nan
        if streamlined == False:
            for m in tqdm(range(0,feature_error_standevs.shape[1],1)):
               # digitize using error bins
               bindex = digitize(stacked_rep_test_features[:,:,m], feature_error_bins[:,m], right=False)-1
               # assign standevs to digitized values
               for i in range(0, len(feature_error_bins)-1, 1):
                   cast_standev[bindex == i] = feature_error_standevs[i, m]
               # gen noise using standevs expand_dims(evbest, axis=-1)
               rn = random.normal(0, cast_standev, [stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]]) #repeat(expand_dims(,-1), 49, -1) #[stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]]
               stacked_rep_test_features[:,:,m] += rn
        else:
            for m in range(0,feature_error_standevs.shape[1],1):
               # digitize using error bins
               bindex = digitize(stacked_rep_test_features[:,:,m], feature_error_bins[:,m], right=False)-1
               # assign standevs to digitized values
               for i in range(0, len(feature_error_bins)-1, 1):
                   cast_standev[bindex == i] = feature_error_standevs[i, m]
               # gen noise using standevs expand_dims(evbest, axis=-1)
               rn = random.normal(0, cast_standev, [stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]]) #repeat(expand_dims(,-1), 49, -1) #[stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]]
               stacked_rep_test_features[:,:,m] += rn
        PostIntOutput_MC = repeat(PostIntOutput, mc_samples, axis=0)
        ''' MAKE PREDICTIONS ''' 
        if streamlined == False:
            print('\nMCBA predictions:')                   
            for i in range(0, len(mod_list), 1):
                model_set = mod_list[i]
                print(f'model {i+1} of {len(mod_list)}')
                sleep(0.5)
                PostIntOutput_MC[:, i] = squeeze(model_set.predict(stacked_rep_test_features, verbose=1)[:, lead_time-1])
        else:
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
            if streamlined == False:
                print('\nextending and reshaping likelihood estimate array')
                sleep(0.5)
            
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
            if streamlined == False:
                print('\nreshaping to original dimensions')
                sleep(0.5)
            unstacked_NetP_MC = reshape(NetP_MC, (mc_samples, int(NetP_MC.shape[0]/mc_samples), NetP_MC.shape[1]), order='F')
            unstacked_PostIntOutput_MC = reshape(PostIntOutput_MC, (mc_samples, int(PostIntOutput_MC.shape[0]/mc_samples), PostIntOutput_MC.shape[1]), order='F')
            # reduce memory burden
            del PostIntOutput_MC
            
            ''' REAVERAGE INTO STANDARD OUTPUT FORMAT '''
            if streamlined == False:
                print('\nABMC model averaging')
                sleep(0.5)
            
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

    if (algorithm != 'ABMS') & (algorithm != 'MCBA') & (streamlined == False):
        print("please select abms algorithm or abmc alogrithm")
    return (Weighted_LOBO, Weighted_ROBO, Weighted_UPBO)







