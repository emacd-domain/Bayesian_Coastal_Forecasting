# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:33:33 2024

@author: alphonse
"""
import os


def make_plots(robust_prediction=None, test=None, normalised=None, lead_time=None, msa=None,
               plot_title=None, plot_prefix=None, plot_directory=None):

    import matplotlib.pyplot as plt
    from  matplotlib import colors
    from numpy import sum, isnan, arange, nanmax, nanmin, round

    
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
        plot_directory = os.getcwd()

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
    elif msa == 'MCBA':
        plt.plot(x, robo, ".b", markersize=7, label = "MCBA Averaged Prediction",linewidth=3.0)
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
    plt.plot(x[:400], upbo[:400], "r", linewidth=2, label = "_nolegend_",alpha=0.8)
    plt.plot(x[:400], lobo[:400], "r", linewidth=2, label = "95% Confidence Interval",alpha=0.8)
    plt.plot(x[:400], test[:400], ".k", markersize=7, label = "Observations",alpha=0.8)
    if msa == 'ABMS':
        plt.plot(x[:400], robo[:400], ".b", markersize=7, label = "ABMS Averaged Prediction",linewidth=2.0)
    elif msa == 'MCBA':
        plt.plot(x[:400], robo[:400], ".b", markersize=7, label = "MCBA Averaged Prediction",linewidth=2.0)
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
    plt.xlim([0,400])
    plt.ylim([0,1.6])
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
        plt.plot([-1.0,1.5],[-1.0,1.5],'r-.',linewidth=1)
        plt.plot([-1.0,1.5],[-0.7,1.8],'r:',linewidth=1, label='+/- 0.3m')
        plt.plot([-1.0,1.5],[-1.3,1.2],'r:',linewidth=1)
        plt.grid(color='r', linestyle=':', linewidth=0.5)
        plt.xlim([-1.0,1.5])
        plt.ylim([-1.0,1.5])
        plt.xticks(ticks=arange(-1.0,2.0,0.5).tolist())
        plt.yticks(ticks=arange(-1.0,2.0,0.5).tolist())
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

def make_metrics(robust_prediction=None, test=None, normalised=None, 
                 metric_prefix=None, metric_directory=None, thres1=None, thres2=None,
                 lead_time=None, msa=None):

    lobo = robust_prediction[0]
    robo = robust_prediction[1]
    upbo = robust_prediction[2]
    #test = robust_prediction[3]

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from numpy import sum, isnan, zeros, sqrt, mean, nan, floor, ceil
    from pandas import DataFrame
    
    qsidx = ~isnan(robo) & ~isnan(test) & (robo!=0)
    robo = robo[qsidx]
    lobo = lobo[qsidx]
    upbo = upbo[qsidx]
    
    test = test[qsidx]
        
    if metric_prefix is None:
        metric_prefix = 'bayesian_model_selection_suite_'
    if metric_directory is None:
        metric_directory = os.getcwd()
    iy = test>thres1
    iz = test>thres2
    errormetrics=zeros([7,3])
    ### R2
    errormetrics[0,0] = r2_score(robo,test)
    ### RMSE
    errormetrics[1,0] = sqrt(mean_squared_error(test,robo))
    ### MAE
    errormetrics[2,0] = mean_absolute_error(robo,test)
    ### BIAS
    errormetrics[3,0] = mean(robo-test)
    ### SI
    errormetrics[4,0]  = sqrt((sum(((test-mean(test))-(robo-mean(robo)))**2))/(sum(test*robo)))
    ### PRCT
    errormetrics[5,0]  = 100-100*(sum(test>upbo)+sum(test<lobo))/len(robo)
    ### PINTER WIDTH
    errormetrics[6,0]  = mean(upbo-lobo)
    
    if sum(iy) > 10:
        errormetrics[0,1] = r2_score(test[iy],robo[iy])
        errormetrics[1,1] = sqrt(mean_squared_error(robo[iy],test[iy]))
        errormetrics[2,1] = mean_absolute_error(robo[iy],test[iy])
        errormetrics[3,1] = mean(robo[iy]-test[iy])
        errormetrics[4,1] = sqrt((sum(((test[iy]-mean(test[iy]))-(robo[iy]-mean(robo[iy])))**2))/(sum(test[iy]*robo[iy])))
        errormetrics[5,1] = 100-100*(sum(test[iy]>upbo[iy])+sum(test[iy]<lobo[iy]))/len(robo[iy])
        errormetrics[6,1] = mean(upbo[iy]-lobo[iy])
    else:
        errormetrics[0,1] = nan
        errormetrics[1,1] = nan
        errormetrics[2,1] = nan
        errormetrics[3,1] = nan
        errormetrics[4,1] = nan
        errormetrics[5,1] = nan
        errormetrics[6,1] = nan
    if sum(iz) > 10:
        errormetrics[0,2] = r2_score(test[iz], robo[iz])
        errormetrics[1,2] = sqrt(mean_squared_error(robo[iz],test[iz]))
        errormetrics[2,2] = mean_absolute_error(robo[iz],test[iz])
        errormetrics[3,2] = mean(robo[iz]-test[iz])
        errormetrics[4,2] = sqrt((sum(((test[iz]-mean(test[iz]))-(robo[iz]-mean(robo[iz])))**2))/(sum(test[iz]*robo[iz])))
        errormetrics[5,2]  = 100-100*(sum(test[iz]>upbo[iz])+sum(test[iz]<lobo[iz]))/len(robo[iz])
        errormetrics[6,2] = mean(upbo[iz]-lobo[iz])
    else:
        errormetrics[0,2] = nan
        errormetrics[1,2] = nan
        errormetrics[2,2] = nan
        errormetrics[3,2] = nan
        errormetrics[4,2] = nan
        errormetrics[5,2] = nan
        errormetrics[6,2] = nan

    # ### HH
    # errormetrics[0,5]  = np.sqrt((np.sum((test-robo)**2))/(np.sum(test*robo)))
    # errormetrics[1,5]  = np.sqrt((np.sum((test[iy]-robo[iy])**2))/(np.sum(test[iy]*robo[iy])))
    # ### CC
    # errormetrics[0,6]  = np.sum((robo-np.mean(robo))*(test-np.mean(test)))/np.sqrt(np.sum((robo-np.mean(robo))**2)*np.sum((test-np.mean(test))**2))
    # errormetrics[1,6]  = np.sum((robo[iy]-np.mean(robo[iy]))*(test[iy]-np.mean(test[iy])))/np.sqrt(np.sum((robo[iy]-np.mean(robo[iy]))**2)*np.sum((test[iy]-np.mean(test[iy]))**2))
    fname=metric_prefix+f"leadtime_{lead_time}_{msa}_metrics.csv"
    df = DataFrame(data=errormetrics, index=['R2','RMSE','MAE','BIAS','SI','PER','BND_SIZE'], columns=['Full Series','Threshold','Extreme'])
    df.to_csv(metric_directory+'\\'+fname)
    
    def plot_timeline():
        return
    
