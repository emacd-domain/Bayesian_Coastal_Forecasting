import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from keras import Model, Sequential, regularizers
from keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, Flatten, RepeatVector, TimeDistributed, BatchNormalization
import scipy.io
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from  matplotlib import colors

from repnoise import repnoise

#####################
##### LOAD DATA #####
#####################

dm = scipy.io.loadmat("..\O5_designmatrix\dm_normalised.mat")
dm = dm['dm_normalised']

targets = dm[:-40000,-1,25:].copy()
features = dm[:-40000,:,:].copy()
features[:, -1, 25:] = 0
features = np.swapaxes(features, 1, 2)

targets_test = dm[-40000:,-1,25:].copy()
features_test = dm[-40000:,:,:].copy()
features_test[:, -1, 25:] = 0
features_test = np.swapaxes(features_test, 1, 2)

del dm

batch = 32
max_epochs = 100

###################
### WEIGHT CALC ###
###################

seq = np.arange(0,len(features), 1)
np.random.shuffle(seq)

cut = int(np.floor(0.8*len(features)))

features_train = features#[:cut, :, :] 
targets_train = targets#[:cut, :]

del features, targets

# make the new training array with balanced inputs
weights = np.zeros([len(targets_train)])
t=np.round(targets_train[:,-1], decimals =1)

uni, count = np.unique(t, return_counts=True)

x=500

all_feat=[]
all_targ=[]
for i in range(0, len(uni), 1):
    print(i)
    arg = t == uni[i]
    if sum(arg) < x:
        rep_feat, rep_targ = repnoise(features_train[arg,:,:], targets_train[arg,:], np.ceil(x/sum(arg)))
        all_feat.append(rep_feat[:x,:,:])
        all_targ.append(rep_targ[:x,:])
    if sum(arg) >= x:
        rep_feat = features_train[arg,:,:]
        rep_targ = targets_train[arg,:]
        ser = np.arange(0, len(rep_feat), 1)
        np.random.shuffle(ser)
        all_feat.append(rep_feat[ser[:x],:,:])
        all_targ.append(rep_targ[ser[:x],:])
        
features_train = np.concatenate(all_feat, axis=0)
targets_train = np.concatenate(all_targ, axis=0)

# for i in range(0, len(weights), 1):
#     arg = t[i] == uni
#     weights[i] = 1/count[arg]

# weights = weights/np.mean(weights)

#shuffle inputs
a = np.arange(len(targets_train))
np.random.shuffle(a)

# weights = weights[a]
features = features_train[a,:,:]
targets = targets_train[a,:]

#t = np.max(targets, axis=1)*0.2
#idx = targets>0

#targets_skew = targets.copy()
#targets_skew = targets_skew + 0.4

#######################
##### BUILD MODEL #####
#######################
#os.mkdir('surge_models')
i=1
while i<12:
    
    os.chdir('surge_models')
    
    input2 = Input(shape=(49, 22), dtype='float32', name='era_pc_input')
    x2=Bidirectional(LSTM(20, return_sequences=False))(input2)
    #x2=Bidirectional(LSTM(1))(x2)
    x2=Flatten()(input2)
    x2=Dense(50)(x2)
    output = Dense(24, name='surge_output')(x2)
    
    model=Model(inputs=input2,
                outputs=output)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min', restore_best_weights=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    history = model.fit({'era_pc_input': features},
                        {'surge_output': targets},
                        batch_size=batch,
                        #validation_data=(features_valid, targets_valid),
                        validation_split=0.2,
                        epochs=10, 
                        callbacks=[early_stopping],
                        shuffle=True
                        )
    
    ''' test model '''
    pred = np.squeeze(model.predict({'era_pc_input':features_test}))
    pred=np.squeeze(pred[:,-1])*0.206
    test = np.squeeze(targets_test)
    test = test[:,-1]*0.206
    
    thres1 = 0.75
    thres2 = 1.25

    #np.percentile(target,95)
    indexy = np.argwhere(test>thres1)
    indexn = np.argwhere(test>thres2)
    errormetrics=np.zeros([1,15])
    errormetrics=np.zeros([1,15])
    ### R2
    errormetrics[0,0] = round(r2_score(test,pred), 3)
    errormetrics[0,1] = round(r2_score(pred[indexy],test[indexy]), 3)
    errormetrics[0,2] = round(r2_score(pred[indexn],test[indexn]), 3)
    ### RMSE
    errormetrics[0,3] = round(np.sqrt(mean_squared_error(pred,test)), 2)
    errormetrics[0,4] = round(np.sqrt(mean_squared_error(pred[indexy],test[indexy])), 2)
    errormetrics[0,5] = round(np.sqrt(mean_squared_error(pred[indexn],test[indexn])), 2)
    ### MAE
    errormetrics[0,6] = round(mean_absolute_error(pred,test), 2)
    errormetrics[0,7] = round(mean_absolute_error(pred[indexy],test[indexy]), 2)
    errormetrics[0,8] = round(mean_absolute_error(pred[indexn],test[indexn]), 2)
    ### BIAS
    errormetrics[0,9] =  round(np.mean(pred)-np.mean(test), 2)
    errormetrics[0,10] = round(np.mean(pred[indexy])-np.mean(test[indexy]), 2)
    errormetrics[0,11] = round(np.mean(pred[indexn])-np.mean(test[indexn]), 2)
    ### SI
    errormetrics[0,12]  = round(np.sqrt(np.sum(((pred-np.mean(pred))-(test-np.mean(test)))**2)/np.sum(test**2)), 3)
    errormetrics[0,13]  = round(np.sqrt(np.sum(((pred[indexy]-np.mean(pred[indexy]))-(test[indexy]-np.mean(test[indexy])))**2)/np.sum(test[indexy]**2)), 3)
    errormetrics[0,14]  = round(np.sqrt(np.sum(((pred[indexn]-np.mean(pred[indexn]))-(test[indexn]-np.mean(test[indexn])))**2)/np.sum(test[indexn]**2)), 3)    
     
    ''' save model '''
    #if abs(errormetrics[0,10]) < 0.15:
        
    fig=plt.figure()
    plt.rc('font',family='Times New Roman')
    #plt.hist2d(test,pred,bins=np.arange(-2,2.5,0.1),cmap='magma',norm=colors.LogNorm())
    #plt.plot(pred,test,'.k',alpha=0.1, markersize=7)
    plt.scatter(test, pred, c='black', alpha=0.15)
    plt.plot([-1.5,2.0],[-1.5,2.0],'r-.',linewidth=1)
    plt.plot([-1.5,2.0],[-1.2,2.3],'r:',linewidth=1.5, label='+/- 0.3m')
    plt.plot([-1.5,2.0],[-1.8,1.7],'r:',linewidth=1.5)
    plt.grid(color='r', linestyle=':', linewidth=0.5)
    plt.xlim([-1.5,2.0])
    plt.ylim([-1.5,2.0])
    plt.title("Network "+str(i), fontdict={'fontsize' : 18})
    plt.xlabel("Observed Surge [m]", fontdict={'fontsize' : 18})
    plt.ylabel("Predicted Surge [m]", fontdict={'fontsize' : 18})
    plt.legend()
    plt.gca().set_aspect('equal')
    fname="Network"+str(i)+".png"
        
    fig.savefig(fname, 
             dpi=500, format=None, metadata=None,
             bbox_inches=None, pad_inches=0.1,
             facecolor='auto', edgecolor='auto',
             backend=None
             )

    MODELNAME = "FOC_SURGE_24HOUR_FORECAST_MODEL_"+str(i)
    
    #model.save(MODELNAME+'_lstm.h5')  

    np.savetxt("errormetrics_"+str(i)+".csv", errormetrics, delimiter=",")
    os.chdir('..')
    i=i+1
    #else:
    #    os.chdir('..')
    del model
    