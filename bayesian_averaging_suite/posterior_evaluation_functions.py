__author__ = "EM"

# import system functions
import os
from time import sleep
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import packages
from numpy import (zeros, abs, round, percentile, arange, 
                   squeeze, array_split, mean, e, sqrt, pi, bincount,
                   argsort, unique, nan, reshape, asarray, where, roll, sum)

from tqdm import tqdm

def initialise_posterior_arrays(streamlined, posttargetgroups, posttargets, models, PosteriorVariables):
    if streamlined == False:
        print("Initialise Variables")
        sleep(0.5)
    post_loop_vars=[]
    Mags = unique(posttargetgroups)
    Var = zeros([len(Mags), len(models)])
    Likelihood = zeros([len(posttargets), len(models)])
    Pnorm = zeros([len(posttargets), len(models)])
    SquError = zeros([len(posttargets), len(models)])
    PosteriorSamples = zeros([len(posttargets), len(models)])
    for i in range (0, len(PosteriorVariables)):
        post_loop_vars.append([Mags, Var, Likelihood, Pnorm, SquError, PosteriorSamples])
    return post_loop_vars


def set_closest_value(array1, array2):
    closest_indices = abs(array1[:, None] - array2).argmin(axis=1)
    closest_values = round(array2[closest_indices], decimals=2)
    return closest_values


def posterior_target_groups(streamlined, posttargets, variable_name):
    """ function to ro """
    if streamlined == False:
        print("\nSorting Posterior Target Groups for "+variable_name)
        sleep(0.5)
    range_percentiles = percentile(posttargets, arange(0,101,2))
    posttargetgrouped = set_closest_value(posttargets, range_percentiles)
    posttargetgroups = unique(posttargetgrouped)
    return posttargetgroups, posttargetgrouped


def make_predictions(streamlined, post_features, mod_list, lead_time):
    if streamlined == False:
        print("\nMaking Predictions for Posterior Calculations")
        sleep(0.5)
        pred = zeros([len(post_features), len(mod_list)])
        count = 0
        for model_set in tqdm(mod_list):
            pred[:, count] = squeeze(model_set.predict(post_features, verbose=0)[:, lead_time-1])
            count += 1
    else:
        pred = zeros([len(post_features), len(mod_list)])
        count = 0
        for model_set in mod_list:
            pred[:, count] = squeeze(model_set.predict(post_features, verbose=0)[:, lead_time-1])
            count += 1
    return pred


def calculate_posteriors(streamlined, post_loop_vars, pred, VarSortIndex, posterior_variable, post_targets, mod_list, uniqueidx, idx, variable_name, lead_time):
    if streamlined == False:
        print("Calculating Posterior Probabilities for "+variable_name)
        sleep(0.5)
    # split predictions into groups and calculate variance across each group of predictions

    # Mags, Var, Likelihood, Pnorm, SquError, RobustOut, PostDist, PosteriorSamples, Pred

    Mags = post_loop_vars[0]
    Var = post_loop_vars[1]
    Likelihood = post_loop_vars[2]
    Evidence = post_loop_vars[3]
    SquError = post_loop_vars[4]
    PosteriorSamples = post_loop_vars[5]

    for i in range(0, len(mod_list), 1):
        splitx = array_split(pred[VarSortIndex, i], idx)
        splitx = splitx[1:]
        splitt = array_split(post_targets[VarSortIndex, lead_time-1], idx)
        splitt = splitt[1:]
        v = zeros([len(Mags)])
        for z in range(0, len(Mags), 1):
            v[z] = mean((splitx[z] - mean(splitt[z])) ** 2)  # (posttarget[VarSortIndex]-pred[:,k])**2)
        ### SQU ERROR
        SquError[:, i] = (post_targets[VarSortIndex, lead_time-1] - pred[VarSortIndex, i]) ** 2
        Var[:, i] = v
        ### UNIFORM PRIORS
        Prior = 1 / (len(mod_list))
        ### MAXIMUM LIKLEHOOD EQUATION
        Likelihood[:, i] = (1 / (sqrt(2 * pi * Var[uniqueidx, i]))) * (e ** (-SquError[:, i] / (2 * Var[uniqueidx, i])))  #
        ### tempP
        binevidence = bincount(uniqueidx, Likelihood[:, i]*Prior)
        Evidence[:,i] = binevidence[uniqueidx]
        #Pnorm[:, i] = yhat[uniqueidx]
        PosteriorSamples = Likelihood * Prior / Evidence
        
    return PosteriorSamples


def fit_distribution(streamlined, PosteriorSamples, idx, posterior_variable_list, PosteriorVariables, VarSortIndexes, post_loop_vars, FitType=None, make_plot=False, plot_title=None, plot_prefix=None, plot_directory=None):
    # get distribution packages
    if streamlined == False:
        print('\n###############################################################')
        print('\nFitting Distribution To Posterior Probabilities')
    from sklearn.mixture import GaussianMixture
    from numpy import (random, array_split, average,
                       concatenate, argmin, diff, full_like, linspace)
    
    if make_plot:
        import matplotlib.pyplot as plt
    PostGMList = []
    for i in range(0, len(PosteriorSamples)):
        if streamlined == False:
            print(f'\nOptimizing and Fitting Gaussian Mixture Models For Variable {posterior_variable_list[i]}')
        Mags = post_loop_vars[i][0]
        GMList=[]
        Y = PosteriorSamples[i]
        X = PosteriorVariables[i]
        VarSortIndex = VarSortIndexes[i]
        
        if (FitType == 'GaussianMixture') & (streamlined==False):
            for j in tqdm(range(0, Y.shape[1], 1)):
                y = Y[:, j].reshape(-1,1)
                x = X.reshape(-1,1)

                splitx = array_split(x[VarSortIndex], idx)
                splitx = splitx[1:]
                splitt = array_split(y[VarSortIndex], idx)
                splitt = splitt[1:]

                norm_params = zeros([len(Mags),2])
                norm_gen = []
                for q in range(0, len(Mags), 1):
                    yy = asarray(splitt[q].flatten())
                    xx = asarray(splitx[q].flatten())
                    weighted_mean = average(xx, weights=yy)
                    weighted_variance = average((xx - weighted_mean)**2, weights=yy)
                    norm_params[q, 0] = weighted_mean.copy()
                    norm_params[q, 1] = sqrt(weighted_variance).copy()
                    norm_gen.append(random.normal(weighted_mean, sqrt(weighted_variance), 1000))
                
                all_data = concatenate([norm_gen], axis=0)
                all_data = reshape(all_data, [all_data.shape[0]*all_data.shape[1]])
                xs = linspace(1.1*min(x), 1.1*max(x), 200)
                
                n_components = arange(1, 22, 3)

                # Initialize lists to store BIC and AIC values
                bics = []
                aics = []

                # Fit GMMs for each number of components and compute BIC/AIC
                for n in n_components:
                    gmm = GaussianMixture(n_components=n, random_state=0)
                    gmm.fit(all_data.reshape(-1,1), y=None)
                    bics.append(gmm.bic(all_data.reshape(-1,1)))
                    aics.append(gmm.aic(all_data.reshape(-1,1)))
                
                optimal_n_components_bic = n_components[argmin(bics)]
                
                # fit optimal mode
                gmm = GaussianMixture(optimal_n_components_bic, random_state=0)
                gmm.fit(all_data.reshape(-1,1), y=None)
                
                # calculate "bin widths by getting the spacing between model means"
                best_means = squeeze(gmm.means_.astype(float))
                order_means = argsort(best_means)
                reverse_order_means = argsort(order_means)
                
                best_means = best_means[order_means]
                differences = diff(best_means)

                # Create arrays for left and right distances
                left_distances = full_like(best_means, nan)
                right_distances = full_like(best_means, nan)

                # Assign differences to left and right arrays
                left_distances[1:] = differences
                right_distances[:-1] = differences
                # make left end equal to right end and vice versa
                left_distances[0] = right_distances[0]
                right_distances[-1] = left_distances[-1]
                distances = (left_distances/2) + (right_distances/2)
                distances = distances[reverse_order_means]
                
                GMList.append([gmm, distances])
                
                ms = gmm.predict_proba(xs.reshape(-1,1))
                mss = sum(ms*gmm.weights_/distances, axis=1)
                del gmm
            
                if make_plot:
                    plt.rc('font',family='Times New Roman')
                    fig=plt.figure()
                    plt.plot(xs, mss)
                    plt.hist(all_data, density=True, bins=len(Mags)) #
                    plt.xlabel("x", fontdict={'fontsize' : 18})
                    plt.ylabel("f(x)", fontdict={'fontsize' : 18})
                    plt.xticks(size=14)
                    plt.title(f'GMM of {posterior_variable_list[i]} Posterior Likelihoods \n for Model {j}')
                    plt.yticks(size=14)
                    plt.tight_layout()
                    plt.show()
                    fname=f"GMM of {posterior_variable_list[i]} Posterior Likelihoods for Model {j}.png"
                    fig.savefig(plot_directory+'\\'+fname, 
                               dpi=300, format=None, metadata=None,
                               bbox_inches=None, pad_inches=0.1,
                               facecolor='auto', edgecolor='auto',
                               backend=None
                               )
        elif (FitType == 'GaussianMixture') & (streamlined==True):
            for j in range(0, Y.shape[1], 1):
                y = Y[:, j].reshape(-1,1)
                x = X.reshape(-1,1)

                splitx = array_split(x[VarSortIndex], idx)
                splitx = splitx[1:]
                splitt = array_split(y[VarSortIndex], idx)
                splitt = splitt[1:]

                norm_params = zeros([len(Mags),2])
                norm_gen = []
                for q in range(0, len(Mags), 1):
                    yy = asarray(splitt[q].flatten())
                    xx = asarray(splitx[q].flatten())
                    weighted_mean = average(xx, weights=yy)
                    weighted_variance = average((xx - weighted_mean)**2, weights=yy)
                    norm_params[q, 0] = weighted_mean.copy()
                    norm_params[q, 1] = sqrt(weighted_variance).copy()
                    norm_gen.append(random.normal(weighted_mean, sqrt(weighted_variance), 1000))
                
                all_data = concatenate([norm_gen], axis=0)
                all_data = reshape(all_data, [all_data.shape[0]*all_data.shape[1]])
                xs = linspace(1.1*min(x), 1.1*max(x), 200)
                
                n_components = arange(1, 22, 3)

                # Initialize lists to store BIC and AIC values
                bics = []
                aics = []

                # Fit GMMs for each number of components and compute BIC/AIC
                for n in n_components:
                    gmm = GaussianMixture(n_components=n, random_state=0)
                    gmm.fit(all_data.reshape(-1,1), y=None)
                    bics.append(gmm.bic(all_data.reshape(-1,1)))
                    aics.append(gmm.aic(all_data.reshape(-1,1)))
                
                optimal_n_components_bic = n_components[argmin(bics)]
                
                # fit optimal mode
                gmm = GaussianMixture(optimal_n_components_bic, random_state=0)
                gmm.fit(all_data.reshape(-1,1), y=None)
                
                # calculate "bin widths by getting the spacing between model means"
                best_means = squeeze(gmm.means_.astype(float))
                order_means = argsort(best_means)
                reverse_order_means = argsort(order_means)
                
                best_means = best_means[order_means]
                differences = diff(best_means)

                # Create arrays for left and right distances
                left_distances = full_like(best_means, nan)
                right_distances = full_like(best_means, nan)

                # Assign differences to left and right arrays
                left_distances[1:] = differences
                right_distances[:-1] = differences
                # make left end equal to right end and vice versa
                left_distances[0] = right_distances[0]
                right_distances[-1] = left_distances[-1]
                distances = (left_distances/2) + (right_distances/2)
                distances = distances[reverse_order_means]
                
                GMList.append([gmm, distances])
                
                ms = gmm.predict_proba(xs.reshape(-1,1))
                mss = sum(ms*gmm.weights_/distances, axis=1)
                del gmm
            
                if make_plot:
                    plt.rc('font',family='Times New Roman')
                    fig=plt.figure()
                    plt.plot(xs, mss)
                    plt.hist(all_data, density=True, bins=len(Mags)) #
                    plt.xlabel("x", fontdict={'fontsize' : 18})
                    plt.ylabel("f(x)", fontdict={'fontsize' : 18})
                    plt.xticks(size=14)
                    plt.title(f'GMM of {posterior_variable_list[i]} Posterior Likelihoods \n for Model {j}')
                    plt.yticks(size=14)
                    plt.tight_layout()
                    plt.show()
                    fname=f"GMM of {posterior_variable_list[i]} Posterior Likelihoods for Model {j}.png"
                    fig.savefig(plot_directory+'\\'+fname, 
                               dpi=300, format=None, metadata=None,
                               bbox_inches=None, pad_inches=0.1,
                               facecolor='auto', edgecolor='auto',
                               backend=None
                               )
        PostGMList.append(GMList)
    return PostGMList


def posterior_evaluation(streamlined=False, post_data=None, posterior_variable_list=['Target'], lead_time=24, mod_list=None, estimation_strategy='Interpolation', 
                         make_plot=False, plot_title=None, plot_prefix=None, plot_directory=None):
    if post_data is not None:
        ''' build abms approach from data and estimation strategy '''
        post_features, post_targets = post_data

        # establish posterior target groups
        # write function here

        PosteriorVariables = []

        if posterior_variable_list is not None:
            for variable in posterior_variable_list:
                if variable == 'Target':
                    target = post_targets[:, lead_time-1]
                    PosteriorVariables.append(target)
                if variable == 'PC1':
                    pc1 = post_features[:, lead_time-1, 0]
                    PosteriorVariables.append(pc1)
                if variable == 'PC2':
                    pc2 = post_features[:, lead_time-1, 1]
                    PosteriorVariables.append(pc2)
                if variable == 'PC3':
                    pc3 = post_features[:, lead_time-1, 2]
                    PosteriorVariables.append(pc3)                
                if variable == 'Local Pressure':
                    mslp = post_features[:, lead_time-1, 0]
                    PosteriorVariables.append(mslp)
        if streamlined==False:
            print("\n######################################################################")
        # make predictions and update post loop variables
        pred = make_predictions(streamlined, post_features, mod_list, lead_time)

        PosteriorSamples = []
        VarSortIndexes = []
        for var_count in range(0, len(posterior_variable_list), 1):
            
            PosteriorVariable = PosteriorVariables[var_count]
            variable_name = posterior_variable_list[var_count]
            # create target groups
            posttargetgroups, posttargetgrouped = posterior_target_groups(streamlined, PosteriorVariable, variable_name)
            # sort posterior targets
            VarSortIndex = argsort(PosteriorVariable)
            # post_loop_vars = (Mags, Var, Likelihood, Pnorm, SquError, RobustOut, PostDist, PosteriorSamples)
            post_loop_vars = initialise_posterior_arrays(streamlined, posttargetgroups, post_targets, mod_list, PosteriorVariables)
            ### sort binned targets for varying model weights with magnitude
            VarSort = posttargetgrouped[VarSortIndex]
            ### unique indexes for reconstruction
            _, _, uniqueidx = unique(VarSort, return_index=True, return_inverse=True)
            # roll unique values to sort into bins
            idx = where(roll(uniqueidx, 1) != uniqueidx)[0]
            # calculate posterior samples for variable
            PosteriorVariableSamples = calculate_posteriors(streamlined, post_loop_vars[var_count], pred, VarSortIndex, PosteriorVariable, post_targets, mod_list, uniqueidx, idx, variable_name, lead_time)
            # append arrays in loop
            PosteriorSamples.append(PosteriorVariableSamples)
            VarSortIndexes.append(VarSortIndex)
    
    if (estimation_strategy == 'GaussianMixture'):
        GMList = fit_distribution(streamlined, PosteriorSamples, idx, posterior_variable_list, PosteriorVariables, VarSortIndexes, post_loop_vars, FitType='GaussianMixture', make_plot=make_plot, plot_title=plot_title, plot_prefix=plot_prefix, plot_directory=plot_directory)
    elif (estimation_strategy == 'Interpolation'):
        GMList = []
    return GMList, PosteriorSamples, PosteriorVariables