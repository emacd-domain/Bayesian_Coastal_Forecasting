__author__ = "EM"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import packages
from keras.models import load_model


def find_all_underscore(string):
    indices = []
    index = string.find("_")
    while index != -1:
        indices.append(index)
        index = string.find("_", index + 1)
    return indices


def load_models(NNSdirectory):
    print("\nLoading Models")
    files = []
    for option in os.listdir(NNSdirectory):
        if os.path.isfile(option):
            continue
        else:
            files.append(option)

    models = [s for s in files if ".h5" in s]

    model_bank = []
    for model in models:
        m = load_model(NNSdirectory+r'\\'+model)
        model_bank.append(m)
    print("Found "+str(len(model_bank))+" Models")

    return model_bank
