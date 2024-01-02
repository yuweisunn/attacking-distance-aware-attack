import numpy as np
import tensorflow as tf
from numpy import linalg as LA


def NDC(local_models):
    norms = []
    models = []
    for model in local_models:
        norms.append(LA.norm(np.concatenate([w.flatten() for w in model])))

    for i,n in enumerate(norms):
        if n < np.median(norms):
            models.append(local_models[i])

    return models
        

def Krum(local_models, gamma):
    norms = []
    models = []
    
    for model in local_models:
        norm_list = []
        for paired_model in local_models:
            norm_list.append(LA.norm(np.concatenate([(paired_model[i] - w).flatten() for i, w in enumerate(model)])))

        norm_list.sort()
        norms.append(np.sum(norm_list[1:-gamma]))

    return [local_models[np.argsort(norms)[0]]]


def TrimmedMean(local_models, beta):
    norms = []
    models = []
    for model in local_models:
        norms.append(LA.norm(np.concatenate([w.flatten() for w in model])))

    for i in np.argsort(norms)[beta:-beta]:
        models.append(local_models[i])

    return models


def DP(local_models, std):

    for model in local_models:
        for i, w in enumerate(model):
            model[i] = np.random.normal(0,std,np.array((w)).shape) + w

    return local_models

