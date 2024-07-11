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
from posterior_evaluation_functions import posterior_evaluation


def initialise_evaluation_arrays(features, models):
    ### INITIALISE VARIABLES TO BE CALCULATED WITHIN LOOP
    NetP = zeros([len(features), len(models)])
    PostIntOutput = zeros([len(features), len(models)])
    ROBO = zeros([len(features)])
    LOBO = zeros([len(features)])
    UPBO = zeros([len(features)])
    return [ROBO, LOBO, UPBO, NetP, PostIntOutput]

def make_robust_predictions(algorithm, lead_time, test_features, posterior_variable_list, PosteriorVariables, mod_list, prediction_loop_vars, alpha, estimation_strategy, PosteriorSamples=None, GMList=None, mc_samples=None, feature_error_standevs=None, feature_error_bins=None):
    
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
        """ make ABMS robust predictions using posterior data """
        print("\n######################################################################")
        print("\nMake ABMS Robust Prediction Using "+estimation_strategy+" Estimation Strategy")
        
        ''' make predictions '''
        print("Making Network Predictions")
        for i in tqdm(range(0, len(mod_list), 1)):
            model_set = mod_list[i]
            PostIntOutput[:, i] = squeeze(model_set.predict(test_features[idx_real,:,:], verbose=0))[:, lead_time-1]
            
        for var_count in range(0, len(posterior_variable_list), 1):
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
    
    if algorithm == 'ABMC':
        
        """ make ABMC robust predictions using posterior data """
        print("\n######################################################################")
        print("\nABMC Robust Prediction Using "+estimation_strategy+" Estimation Strategy")
        
        ''' RESHAPE INPUTS '''
        rep_test_features = repeat(expand_dims(test_features[idx_real,:,:], 0), mc_samples, axis=0)
        print('\nreshape features to accommodate MC method')
        stacked_rep_test_features = reshape(rep_test_features, (rep_test_features.shape[0]*rep_test_features.shape[1], rep_test_features.shape[2], rep_test_features.shape[3]), order='F')
        del rep_test_features
        
        '''  GENERATE ERRORS BASED ON FEATURE VALUES AND APPLY TO EACH SAMPLE IN BATCH'''
        print('\napplying errors to features:')
        sleep(0.5)
        cast_standev = zeros([stacked_rep_test_features.shape[0], stacked_rep_test_features.shape[1]])
        cast_standev[:] = nan
        for m in tqdm(range(0,feature_error_standevs.shape[1],1)):
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
        print('\nABMC predictions:')                   
        for i in range(0, len(mod_list), 1):
            model_set = mod_list[i]
            print(f'model {i+1} of {len(mod_list)}')
            sleep(0.5)
            PostIntOutput_MC[:, i] = squeeze(model_set.predict(stacked_rep_test_features, verbose=1)[:, lead_time-1])
        
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
            print('\nreshaping to original dimensions')
            sleep(0.5)
            unstacked_NetP_MC = reshape(NetP_MC, (mc_samples, int(NetP_MC.shape[0]/mc_samples), NetP_MC.shape[1]), order='F')
            unstacked_PostIntOutput_MC = reshape(PostIntOutput_MC, (mc_samples, int(PostIntOutput_MC.shape[0]/mc_samples), PostIntOutput_MC.shape[1]), order='F')
            # reduce memory burden
            del PostIntOutput_MC
            ''' REAVERAGE INTO STANDARD OUTPUT FORMAT '''
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

    if (algorithm != 'ABMS') & (algorithm != 'ABMC'):
        print("please select abms algorithm or abmc alogrithm")
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
        feature_error_bins=None
       ):
    
    if msa is None:
        print("\nPlease select ABMS or ABMC algorithm")
        sys.exit()
    elif msa == 'ABMS':
        print("\n")
        print("###############################################################")
        print("######### ADAPTIVE BAYESIAN MODEL SELECTION ALGORITHM #########")
        print("###############################################################")
        sleep(0.5)
    elif msa == 'ABMC':
        print("\n")
        print("##########################################################################")
        print("######### MONTECARLO ADAPTIVE BAYESIAN MODEL SELECTION ALGORITHM #########")
        print("##########################################################################")
        
    if NNSdirectory is None:
        print("\nNo Model Directory Provided!")
        sleep(0.5)
        sys.exit()
    
    if features is None:
        print("\nNo Data Provided!")
        sleep(0.5)
        sys.exit()
    else:
        print(f"\nfound feature input with dimensions {features.shape}")
        sleep(0.5)
        
    if means is None:
        print("No Means Provided for Features!")
        sleep(0.5)
        sys.exit()
    else:
        print(f"found means with dimensions {means.shape}")
        sleep(0.5)

    if stdevs is None:
        print("\nNo Standard Deviations Provided for Features!")
        sleep(0.5)
        sys.exit()
    else:
        print(f"found standard deviations with dimensions {stdevs.shape}")
        sleep(0.5)

    # set p-value for confidence interval
    if alpha is None:
        print("\nNo alpha provided, 95% confidence interval alpha=1.96 is assumed")
        sleep(0.5)
        alpha = 1.96

    # set p-value for confidence interval
    if lead_time is None:
        print("\nNo lead time provided, 24 hour lead time is assumed")
        sleep(0.5)
        lead_time = 24
    #elif len(lead_time) > 1:
    #    print('\nPlease provide a single lead time')

    # set abms estimation strategy
    if estimation_strategy is None:
        print("\n No estimation strategy provided, interpolation is assumed")
        sleep(0.5)
        estimation_strategy = 'interpolate'

    # check whether distribution is provided for abms or
    if post_distro is not None and post_data is None:
        GMList = post_distro
        print(f"\n{len(GMList[0])} posterior distributions provided")
        for i in range (0, len(GMList[0]), 1):
            print(f'distribution {i+1}: {GMList[0][i][0]}')
    elif post_distro is None and post_data is None:
        print("\nPlease provide model selection distribution or data to construct model selection distribution")
        sleep(0.5)
        sys.exit()
    elif post_distro is not None and post_data is not None:
        print("Both model selection distribution and data provided, \n abms method will use selection distribution")
        sleep(0.5)
        post_data = None
        GMList = post_distro
    elif post_distro is None and post_data is not None:
        print(f"found likelihood evaluation data with feature dimensions {post_data[0].shape} and target dimensions {post_data[1].shape}")
        sleep(0.5)

    # set n_samples if abmc is selected and mc_samples is undefined
    if (mc_samples is None) & (msa == 'ABMC'):
        mc_samples = 25
        print(f"\nmc_samples not defined, setting default value {mc_samples}")
        sleep(0.5)
    elif (mc_samples is not None) & (msa == 'ABMC'):
        print(f"\nmc_samples set at {mc_samples}")
        sleep(0.5)
        
    if (feature_error_standevs is None) & (msa == 'ABMC'):
        print("\nPlease provide feature error standard deviations or select ABMS if none are available")
        sleep(0.5)
        sys.exit()
    elif (feature_error_standevs is not None) & (msa == 'ABMC'):
        print(f"found feature error standard deviations with dimensions {feature_error_standevs.shape}")
        sleep(0.5)
        
    if (feature_error_bins is None) & (msa == 'ABMC'):
        print("\nPlease provide feature error standard deviation bins for error standard deviations or select ABMS if none are available")
        sleep(0.5)
        sys.exit()
    elif (feature_error_bins is not None) & (msa == 'ABMC'):
        print(f"found feature error standard deviation bins with dimensions {feature_error_bins.shape}")
        sleep(0.5)
    
    if (models is None) & (NNSdirectory is not None):
        # load models
        print("model directory provided")
        sleep(0.5)
        mod_list = load_models(NNSdirectory)
    elif (models is not None):
        print(f"model list provided containing {len(models)} models")
        mod_list = models.copy()
        del models
        sleep(0.5)
    elif (models is None) & (NNSdirectory is None): 
        print("PLease provide either model list or model directory")

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
                                                    feature_error_bins=feature_error_bins
                                                   )
        
    test = targets[:, lead_time-1]

    if not normalised:
        
        robust_prediction = ((robust_prediction[0] * stdevs[0, -1]) + means[0, -1],
                             (robust_prediction[1] * stdevs[0, -1]) + means[0, -1],
                             (robust_prediction[2] * stdevs[0, -1]) + means[0, -1])
        test = (test * stdevs[0, -1]) + means[0, -1]
        
    if make_plot:

        import matplotlib.pyplot as plt
        from  matplotlib import colors
        from numpy import sum
        
        lobo = robust_prediction[0]
        robo = robust_prediction[1]
        upbo = robust_prediction[2]
        
        # sort values in descending order for plot
        sort = (-robo).argsort()
        qsidx = sum(~isnan(robo)*1, axis=0)
        lobo=lobo[sort]
        robo=robo[sort]
        upbo=upbo[sort]
        test=test[sort]
        x = arange(0, len(test), 1) 
        
        if plot_prefix is None:
            plot_prefix = 'bayesian_model_selection_suite_'
    
        if plot_directory is None:
            plot_directory = workingdirectory
    
        #idx = (test <= upbo) & (test > lobo)
        # full series plot    
        plt.rc('font',family='Times New Roman')
        plt.show()
        fig=plt.figure()
        plt.plot(x, upbo, ".r", markersize=5, label = "_nolegend_",alpha=0.3)
        plt.plot(x, lobo, ".r", markersize=5, label = "95% Confidence Interval",alpha=0.3)
        plt.plot(x, test, ".k",  markersize=5, label = "Observations",alpha=0.3)
        if msa == 'ABMS':
            plt.plot(x, robo, ".b", markersize=7, label = "ABMS Averaged Prediction",linewidth=3.0)
        elif msa == 'ABMC':
            plt.plot(x, robo, ".b", markersize=7, label = "ABMC Averaged Prediction",linewidth=3.0)
        if plot_title is not None:
            plt.title(f"{plot_title}\n{lead_time} Hour Predictions", fontdict={'fontsize' : 18})
        else:
            plt.title(f"{lead_time} Hour Predictions", fontdict={'fontsize' : 18})
        plt.xlabel("Prediction Number Sorted by Descending Magnitude", fontdict={'fontsize' : 18})
        if normalised:
            plt.ylabel("Normalized Surge Height", fontdict={'fontsize' : 18})
        else:
            plt.ylabel("Surge Height [m]", fontdict={'fontsize' : 18})
        plt.grid(color='r', linestyle='--', linewidth=0.5)
        lgnd=plt.legend(prop = {'size' : 14})
        lgnd.legendHandles[0]._legmarker.set_markersize(7)
        lgnd.legendHandles[1]._legmarker.set_markersize(7)
        plt.xlim([0, qsidx])
        plt.tight_layout()
        plt.show()
        fname=plot_prefix+f"leadtime_{lead_time}_{msa}_descending_robust.png"
        fig.savefig(plot_directory+'\\'+fname, 
                    dpi=300, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None
                    )
    
        # extreme plot
        fig=plt.figure()
        plt.plot(x[:200], upbo[:200], "r", linewidth=2, label = "_nolegend_",alpha=0.8)
        plt.plot(x[:200], lobo[:200], "r", linewidth=2, label = "95% Confidence Interval",alpha=0.8)
        plt.plot(x[:200], test[:200], ".k", markersize=7, label = "Observations",alpha=0.8)
        if msa == 'ABMS':
            plt.plot(x[:200], robo[:200], ".b", markersize=7, label = "ABMS Averaged Prediction",linewidth=2.0)
        elif msa == 'ABMC':
            plt.plot(x[:200], robo[:200], ".b", markersize=7, label = "ABMC Averaged Prediction",linewidth=2.0)
        if plot_title is not None:
            plt.title(f"{plot_title}\n{lead_time} Hour Predictions", fontdict={'fontsize' : 18})
        else:
            plt.title(f"{lead_time} Hour Predictions", fontdict={'fontsize' : 18})
        plt.xlabel("Prediction Number Sorted by Descending Magnitude", fontdict={'fontsize' : 18})
        if normalised:
            plt.ylabel("Normalized Surge Height", fontdict={'fontsize' : 18})
        else:
            plt.ylabel("Surge Height [m]", fontdict={'fontsize' : 18})
        plt.grid(color='r', linestyle='--', linewidth=0.5)
        lgnd=plt.legend( prop = {'size' : 14})
        lgnd.legendHandles[1]._legmarker.set_markersize(7)
        plt.tight_layout()
        plt.xlim([0,200])
        #plt.ylim([0,1.4])
        plt.show()
        fname=plot_prefix+f"leadtime_{lead_time}_{msa}_descending_robust_extreme.png"
        fig.savefig(plot_directory+'\\'+fname, 
                  dpi=300, format=None, metadata=None,
                  bbox_inches=None, pad_inches=0.1,
                  facecolor='auto', edgecolor='auto',
                  backend=None
                  )
    
        # plot pred vs observed
        fig=plt.figure()
        ul = nanmax([round(test)+0.5, round(robo)+0.5])
        ll = nanmin([round(test)-0.5, round(robo)-0.5])
        if normalised:
            plt.hist2d(test, robo, bins=arange(ll,ul,0.5),cmap='magma',norm=colors.LogNorm())  
        else:
            plt.scatter(test, robo, c='black', alpha=0.15)
            plt.plot([-1.5,2.0],[-1.5,2.0],'r-.',linewidth=1)
            plt.plot([-1.5,2.0],[-1.2,2.3],'r:',linewidth=1, label='+/- 0.3m')
            plt.plot([-1.5,2.0],[-1.8,1.7],'r:',linewidth=1)
            plt.grid(color='r', linestyle=':', linewidth=0.5)
            plt.xlim([-1.5,2.0])
            plt.ylim([-1.5,2.0])
            plt.xticks(ticks=arange(-1.5,2.5,0.5).tolist())
            plt.yticks(ticks=arange(-1.5,2.5,0.5).tolist())
            plt.legend()
            plt.gca().set_aspect('equal')
            
        
        plt.plot([-10,10], [-10,10],'r:')
        plt.grid(color='r', linestyle=':', linewidth=0.5) 

        if plot_title is not None:
            plt.title(f"{plot_title}\n{lead_time} Hour Predictions", fontdict={'fontsize' : 18})
        else:
            plt.title(f"{lead_time} Hour Forecast", fontdict={'fontsize' : 18})
        if normalised:
            plt.xlabel("Normalized Observations", fontdict={'fontsize' : 18})
            plt.ylabel(f"Normalized Averaged Predictions \n{msa} Algorithm ", fontdict={'fontsize' : 18})
        else:
            plt.xlabel("Observed [m]", fontdict={'fontsize' : 18})
            plt.ylabel(f"Averaged {msa} [m]", fontdict={'fontsize' : 18})
        plt.grid(color='r', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        fname=plot_prefix+f"leadtime_{lead_time}_{msa}_measured_vs_forecast.png"
        fig.savefig(plot_directory+'\\'+fname, 
                dpi=300, format=None, metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None
                )

    if make_metric:
    
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from numpy import sum
        from pandas import DataFrame
        
        qsidx = ~isnan(robo) & ~isnan(test)
        robo = robo[qsidx]
        test = test[qsidx]
        
        if not normalised:
            thres1 = 0.75
            thres2 = 1.25
        else:
            thres1 = (0.75 - means[0, -1])/stdevs[0, -1]
            thres2 = (1.25 - means[0, -1])/stdevs[0, -1]

        
        if metric_prefix is None:
            metric_prefix = 'bayesian_model_selection_suite_'
        if metric_directory is None:
            metric_directory = workingdirectory
        iy = test>thres1
        iz = test>thres2
        errormetrics=zeros([5,3])
        ### R2
        errormetrics[0,0] = r2_score(robo,test)
        ### RMSE
        errormetrics[1,0] = sqrt(mean_squared_error(robo,test))
        ### MAE
        errormetrics[2,0] = mean_absolute_error(robo,test)
        ### BIAS
        errormetrics[3,0] = mean(robo-test)
        ### SI
        errormetrics[4,0]  = sqrt((sum(((test-mean(test))-(robo-mean(robo)))**2))/(sum(test*robo)))
        if sum(iy) > 10:
            errormetrics[0,1] = r2_score(robo[iy],test[iy])
            errormetrics[1,1] = sqrt(mean_squared_error(robo[iy],test[iy]))
            errormetrics[2,1] = mean_absolute_error(robo[iy],test[iy])
            errormetrics[3,1] = mean(robo[iy]-test[iy])
            errormetrics[4,1] = sqrt((sum(((test[iy]-mean(test[iy]))-(robo[iy]-mean(robo[iy])))**2))/(sum(test[iy]*robo[iy])))
        else:
            errormetrics[0,1] = nan
            errormetrics[1,1] = nan
            errormetrics[2,1] = nan
            errormetrics[3,1] = nan
            errormetrics[4,1] = nan
        if sum(iz) > 10:
            errormetrics[0,2] = r2_score(robo[iz],test[iz])
            errormetrics[1,2] = sqrt(mean_squared_error(robo[iz],test[iz]))
            errormetrics[2,2] = mean_absolute_error(robo[iz],test[iz])
            errormetrics[3,2] = mean(robo[iz]-test[iz])
            errormetrics[4,2] = sqrt((sum(((test[iz]-mean(test[iz]))-(robo[iz]-mean(robo[iz])))**2))/(sum(test[iz]*robo[iz])))
        else:
            errormetrics[0,2] = nan
            errormetrics[1,2] = nan
            errormetrics[2,2] = nan
            errormetrics[3,2] = nan
            errormetrics[4,2] = nan
        # ### HH
        # errormetrics[0,5]  = np.sqrt((np.sum((test-robo)**2))/(np.sum(test*robo)))
        # errormetrics[1,5]  = np.sqrt((np.sum((test[iy]-robo[iy])**2))/(np.sum(test[iy]*robo[iy])))
        # ### CC
        # errormetrics[0,6]  = np.sum((robo-np.mean(robo))*(test-np.mean(test)))/np.sqrt(np.sum((robo-np.mean(robo))**2)*np.sum((test-np.mean(test))**2))
        # errormetrics[1,6]  = np.sum((robo[iy]-np.mean(robo[iy]))*(test[iy]-np.mean(test[iy])))/np.sqrt(np.sum((robo[iy]-np.mean(robo[iy]))**2)*np.sum((test[iy]-np.mean(test[iy]))**2))
        fname=metric_prefix+f"leadtime_{lead_time}_{msa}_metrics.csv"
        df = DataFrame(data=errormetrics, index=['R2','RMSE','MAE','BIAS','SI'], columns=['Full Series','Threshold','Extreme'])
        df.to_csv(metric_directory+'\\'+fname)
    return robust_prediction




