import sklearn.linear_model as lm
from dtuimldmtools import bmplot, feature_selector_lr, rlr_validate, train_neural_net
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

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

def ann_inner(X_train, y_train, X_test, y_test, min_n_hidden_units, max_n_hidden_units, M, n_rep_ann, max_iter):
    validation_errors = []
    hidden_units_used = []

    for n_hidden_units in range(min_n_hidden_units, max_n_hidden_units + 1):
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden_units, 1),
        )

        net, final_loss, learning_curve = train_neural_net(
            model, torch.nn.MSELoss(), X_train, y_train,
            n_replicates=n_rep_ann, max_iter=max_iter
        )

        y_est_inner = net(X_test)
        e = (y_est_inner.float() - y_test.float())**2
        error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()[0]
        validation_errors.append(error_rate)
        hidden_units_used.append(n_hidden_units)

    return validation_errors, hidden_units_used

def ann_outer(hidden_units_matrix, val_error_matrix, data_val_len, data_outer_train_length,
              X_train_outer_tensor, y_train_outer_tensor,
              X_test_outer_tensor, y_test_outer_tensor,
              k_outer, test_errors_outer_ANN,
              M, n_rep_ann, max_iter):

    val_error_matrix = np.transpose(val_error_matrix)
    estimated_errors = [
        np.sum(np.multiply(data_val_len, val_error_matrix[s])) / data_outer_train_length
        for s in range(len(val_error_matrix))
    ]
    best_index = np.argmin(estimated_errors)
    optimal_units = hidden_units_matrix[k_outer][best_index]

    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, int(optimal_units)),
        torch.nn.Tanh(),
        torch.nn.Linear(int(optimal_units), 1)
    )

    net, final_loss, learning_curve = train_neural_net(
        model, torch.nn.MSELoss(), X_train_outer_tensor, y_train_outer_tensor,
        n_replicates=n_rep_ann, max_iter=max_iter
    )

    y_est_outer = net(X_test_outer_tensor)
    e = (y_est_outer.float() - y_test_outer_tensor.float())**2
    error_rate = (sum(e).type(torch.float)/len(y_test_outer_tensor)).data.numpy()[0]
    test_errors_outer_ANN.append(error_rate)

    return optimal_units, test_errors_outer_ANN


