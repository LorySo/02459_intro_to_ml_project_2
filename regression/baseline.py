import sklearn.linear_model as lm
from dtuimldmtools import bmplot, feature_selector_lr, rlr_validate
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

features = pd.read_csv("breast_cancer_wisconsin_features.csv")
targets = pd.read_csv("breast_cancer_wisconsin_targets.csv")

unique_attributes = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

for attr in unique_attributes:
    features = features.drop(columns=[attr+'2',attr+'3'], axis = 1)

# feature transformation

Y = features['smoothness1']

X = features.drop(columns=['smoothness1', 'perimeter1','area1','symmetry1'], axis = 1)
X_attributes = ['radius1', 'texture1', 'compactness1', 'concavity1', 'concave_points1']


for attr in X_attributes:
    mean = X[attr].mean()
    std = X[attr].std()

    X[attr] = (X[attr] - mean) / std


#X=X[targets['Diagnosis']=='M']
#Y=Y[targets['Diagnosis']=='M']

N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
X_attributes = ["Offset"] + X_attributes
M = M + 1

Y=Y.squeeze()
Y=Y.to_numpy()

def baseline_inner(y_train_inner, y_test_inner, error_inner_baseline, data_validation_length):
    mean_y = np.mean(y_train_inner)
    y_est_test_inner_baseline = mean_y
    validation_error = np.sum((y_est_test_inner_baseline - y_test_inner)**2) / float(len(y_test_inner)) 
    error_inner_baseline.append(validation_error)
    data_validation_length.append(float(len(y_test_inner)))
    return error_inner_baseline, data_validation_length

def baseline_outer(y_train_outer, y_test_outer, test_error_outer_baseline):
    mean_y = np.mean(y_train_outer)
    y_est_test_outer_baseline = mean_y     
    test_error = np.sum((y_est_test_outer_baseline - y_test_outer)**2) / float(len(y_test_outer))
    test_error_outer_baseline.append(test_error)
    return test_error_outer_baseline