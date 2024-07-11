import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from keras import Model, Sequential, regularizers
from keras.layers import Input, Dense, LSTM2D, Dropout, Bidirectional, ConvLSTM1D, Flatten, RepeatVector, TimeDistributed, BatchNormalization
import scipy.io
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from  matplotlib import colors

from repnoise import repnoise

#####################
##### LOAD DATA #####
#####################

dm = scipy.io.loadmat("..\O5_designmatrix\dm_normalised.mat")
dm = dm['dm_normalised']

targets_train = dm[:-25000,-1,25:].copy()
features_train = dm[:-25000,:,:].copy()
features_train[:, -1, 25:] = 0
features_train = np.swapaxes(features_train, 1, 2)

#targets_valid = dm[150000:-25000,-1,25:].copy()
#features_valid = dm[150000:-25000,:,:].copy()
#features_valid[:, -1, 25:] = 0
#features_valid = np.swapaxes(features_valid, 1, 2)

#features_pc_abms = dm[194528:204528,:-1,:] 
#features_surge_abms = np.expand_dims(dm[194528:204528,-1,:25], 1)
#targets_surge_abms = dm[194528:204528,-1,25:] 

targets_test = dm[-25000:,-1,25:].copy()
features_test = dm[-25000:,:,:].copy()
features_test[:, -1, 25:] = 0
features_test = np.swapaxes(features_test, 1, 2)

del dm

###################
### WEIGHT CALC ###
###################

# make the new training array with balanced inputs
#weights = np.zeros([len(targets_train)])
#t=np.round(targets_train[:,-1], decimals =1)

#uni, count = np.unique(t, return_counts=True)

#for i in range(0, len(weights), 1):
#    arg = t[i] == uni
#    weights[i] = 1/count[arg]

#weights = weights/np.mean(weights)

t = np.max(targets, axis=1)*0.2
idx = targets*0.2>0.6

targets_skew = targets.copy()
targets_skew[idx] += 0.5

#######################
##### BUILD MODEL #####
#######################
#os.mkdir('surge_models')

batch = 32
max_epochs = 50

i=1
while i<9:
    
    os.chdir('residual_mod')
    
    input2 = Input(shape=(49, 20, 1), dtype='float32', name='era_pc_input')
    x2 =ConvLSTM1D(20, kernel_size = 6, return_sequences=True)(input2)
    x2=Flatten()(x2)
    x2=Dense(24)(x2)
    x2=BatchNormalization()(x2)
    output = Dense(24, name='surge_output')(x2)
   
    model=Model(inputs=input2,
                outputs=output)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, mode='min', restore_best_weights=True)
    
   
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    history = model.fit({'era_pc_input': np.expand_dims(features_train, -1)},
                        {'surge_output': targets_train},
                        batch_size=batch,
                        #sample_weight=weights,
                        validation_split=0.2,
                        #validation_data=({'era_pc_input':features_valid},
                        #                 {'surge_output':targets_valid}),
                        epochs=max_epochs, 
                        callbacks=[early_stopping],
                        shuffle=True
                        )

    ''' test model '''
    pred = np.squeeze(model.predict({'era_pc_input':features_test}))
    pred=np.squeeze(pred[:,-1])*0.206
    test = np.squeeze(targets_test)
    test = test[:,-1]*0.206
    
    thres = 0.70 #np.percentile(target,95)
    indexy = np.argwhere(test>thres)
    indexn = np.argwhere(test<=thres)
    errormetrics=np.zeros([1,15])
    ### R2
    errormetrics[0,0] = round(r2_score(test, pred), 3)
    errormetrics[0,1] = round(r2_score(test[indexy],pred[indexy]), 3)
    errormetrics[0,2] = round(r2_score(test[indexn],pred[indexn]), 3)
    ### RMSE
    errormetrics[0,3] = round(np.sqrt(mean_squared_error(pred,test)), 2)
    errormetrics[0,4] = round(np.sqrt(mean_squared_error(pred[indexy],test[indexy])), 2)
    errormetrics[0,5] = round(np.sqrt(mean_squared_error(pred[indexn],test[indexn])), 2)
    ### MAE
    errormetrics[0,6] = round(mean_absolute_error(pred,test), 2)
    errormetrics[0,7] = round(mean_absolute_error(pred[indexy],test[indexy]), 2)
    errormetrics[0,8] = round(mean_absolute_error(pred[indexn],test[indexn]), 2)
    ### BIAS
    errormetrics[0,9] =  round(np.mean(test)/np.mean(pred), 3)
    errormetrics[0,10] = round(np.mean(test[indexy])/np.mean(pred[indexy]), 3)
    errormetrics[0,11] = round(np.mean(test[indexn])/np.mean(pred[indexn]), 3)
    ### SI
    errormetrics[0,12]  = round(np.sqrt(np.sum(((pred-np.mean(pred))-(test-np.mean(test)))**2)/np.sum(test**2)), 3)
    errormetrics[0,13]  = round(np.sqrt(np.sum(((pred[indexy]-np.mean(pred[indexy]))-(test[indexy]-np.mean(test[indexy])))**2)/np.sum(test[indexy]**2)), 3)
    errormetrics[0,14]  = round(np.sqrt(np.sum(((pred[indexn]-np.mean(pred[indexn]))-(test[indexn]-np.mean(test[indexn])))**2)/np.sum(test[indexn]**2)), 3)
    
    ''' save model '''
    #if abs(errormetrics[0,10]-1) < 0.05:
    
    MODELNAME = "FOC_SURGE_24HOUR_FORECAST_MODEL_"+str(i)

    #model.save(MODELNAME+'_lstm.h5')    
    
    fig=plt.figure()
    plt.rc('font',family='Times New Roman')
    plt.hist2d(test,pred,bins=np.arange(-2,2.5,0.1),cmap='magma',norm=colors.LogNorm())
    #plt.plot(pred,test,'.k',alpha=0.1, markersize=7)
    plt.plot([-1.5,2.5],[-1.5,2.5],'r:',linewidth=3)
    plt.grid(color='r', linestyle=':', linewidth=0.5)
    plt.xlim([-1.5,2.5])
    plt.ylim([-1.5,2.5])
    plt.title("Network "+str(i), fontdict={'fontsize' : 18})
    plt.xlabel("Observed Surge [m]", fontdict={'fontsize' : 18})
    plt.ylabel("Predicted Surge [m]", fontdict={'fontsize' : 18})
    fname="Network"+str(i)+".png"
    
    fig.savefig(fname, 
             dpi=500, format=None, metadata=None,
             bbox_inches=None, pad_inches=0.1,
             facecolor='auto', edgecolor='auto',
             backend=None
             )

    np.savetxt("errormetrics_"+str(i)+".csv", errormetrics, delimiter=",")
    os.chdir('..')
        i=i+1
    else:
        os.chdir('..')
    del model
    