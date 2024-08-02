import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from keras import Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Flatten

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from repnoise import repnoise

#####################
##### LOAD DATA #####
#####################

import scipy.io
import mat73

dm = mat73.loadmat("..\O6_designmatrix\dm_normalised.mat")
dm = dm['dm_normalised']

means = scipy.io.loadmat("..\O6_designmatrix\dm_mean.mat")
means = means['dm_mean']

stds = scipy.io.loadmat("..\O6_designmatrix\dm_std.mat")
stds = stds['dm_std']

target_id = -3
n_feat = 44

targets = dm[:-40000,target_id,:].copy()
features = dm[:-40000,:n_feat,:].copy()
features = np.swapaxes(features, 1, 2)

targets_test = dm[-40000:,target_id,:].copy()
features_test = dm[-40000:,:44,:].copy()
features_test = np.swapaxes(features_test, 1, 2)

targets_ext = (dm[:-40000,-4,-1].copy()*stds[0,-4]) + means[0,-4]
targets_test_ext = (dm[-40000:,-4,-1].copy()*stds[0,-4]) + means[0,-4]

idx_ext = targets_ext > 0.10
idx_test_ext = targets_test_ext > 0.10

targets = targets[idx_ext, :]
features = features[idx_ext, :, :]

targets_test = targets_test[idx_test_ext, :]
features_test = features_test[idx_test_ext, :, :]

targets_test_ext = targets_test_ext[idx_test_ext]

target_mean = means[0, target_id]           
target_std = stds[0, target_id]

del dm

batch = 32
max_epochs = 100

###################
### WEIGHT CALC ###
###################

features_train = features
targets_train = targets

del features, targets

# make the new training array with balanced inputs
weights = np.zeros([len(targets_train)])
t=np.round(targets_train[:, -1], decimals=1)

uni, count = np.unique(t, return_counts=True)

x=1000

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

#shuffle inputs
a = np.arange(len(targets_train))
np.random.shuffle(a)

features = features_train[a,:,:]
targets = targets_train[a,:]

#######################
##### BUILD MODEL #####
#######################

#os.mkdir('wave_period_models')
i=1
while i<9:
    
    os.chdir('wave_period_models')
   
    input2 = Input(shape=(25, 44), dtype='float32', name='era_pc_input')
    x2=Bidirectional(LSTM(20, return_sequences=False))(input2)
    x2=Flatten()(x2)
    x2=Dense(50)(x2)
    output = Dense(25, name='wave_output')(x2)
    
    model=Model(inputs=input2,
                outputs=output)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min', restore_best_weights=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    history = model.fit({'era_pc_input': features},
                        {'wave_output': targets},
                        batch_size=batch,
                        validation_split=0.2,
                        epochs=20, 
                        callbacks=[early_stopping],
                        shuffle=True
                        )
    
    ''' test model '''
    pred = np.squeeze(model.predict({'era_pc_input':features_test}))
    pred=(np.squeeze(pred[:,-1])*target_std)+target_mean
    test = np.squeeze(targets_test)
    test = (test[:, -1]*target_std)+target_mean
    
    thres1 = 2.0
    thres2 = 2.5

    #np.percentile(target,95)
    indexy = np.argwhere(targets_test_ext>thres1)
    indexn = np.argwhere(targets_test_ext>thres2)
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
        
    fig=plt.figure()
    plt.rc('font',family='Times New Roman')
    plt.scatter(test, pred, c='black', alpha=0.15)
    plt.plot([0, 6.5], [-0, 6.5], 'r-.', linewidth=1)
    plt.plot([0, 6.5], [0.5, 7.0], 'r:', linewidth=1.5, label='+/- 0.25')
    plt.plot([0, 6.5], [-0.5, 6.0], 'r:', linewidth=1.5)
    plt.grid(color='r', linestyle=':', linewidth=0.5)
    plt.xlim([0, 6.5])
    plt.ylim([0, 6.5])
    plt.title("Network "+str(i), fontdict={'fontsize' : 18})
    plt.xlabel("Observed Tm [s]", fontdict={'fontsize' : 18})
    plt.ylabel("Predicted Tm [s]", fontdict={'fontsize' : 18})
    plt.legend()
    plt.gca().set_aspect('equal')
    fname="Network"+str(i)+".png"
        
    fig.savefig(fname, 
             dpi=500, format=None, metadata=None,
             bbox_inches=None, pad_inches=0.1,
             facecolor='auto', edgecolor='auto',
             backend=None
             )

    MODELNAME = "FOC_TM_24HOUR_FORECAST_MODEL_"+str(i)
    
    model.save(MODELNAME+'_lstm.h5')  

    np.savetxt("errormetrics_"+str(i)+".csv", errormetrics, delimiter=",")
    os.chdir('..')
    i=i+1

    del model
    