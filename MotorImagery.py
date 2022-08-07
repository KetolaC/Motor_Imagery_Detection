## Import libraries

from sklearn import preprocessing as p, model_selection as ms, svm, ensemble, neighbors, metrics as m
from scipy import signal as sig
from statistics import mode
import numpy as np
import pandas as pd
import random
import os

## Get variables ready

cwd = os.getcwd()  # Gets the current working directory
my_dir = cwd + '/data'  # Set the data directory path

random.seed(98)  # Seed rng for replicability

# subjects = [str(i).zfill(3) for i in random.sample(range(1, 109), 20)]  # Select 20 random participants
subjects = [str(i).zfill(3) for i in range(1, 109)]

eeg_data = np.empty((1, 64))
eeg_labels = np.empty(1, dtype='int8')

## Load Data

for s in subjects:
    dirname = 'S' + s
    path = os.path.join(my_dir, dirname)

    for f in os.listdir(path):
        f_path = os.path.join(path, f)
        if 'labels' in f:
            temp_l = np.loadtxt(f_path, dtype="int8", skiprows=1, delimiter=',')
            if 'R04' in f or 'R08' in f or 'R12' in f:
                map = {1.0: 0, 2.0: 1, 3.0: 2}
                temp_l = np.vectorize(map.get)(temp_l)  # Map events to unique labels
            else:
                map = {1.0: 0, 2.0: 3, 3.0: 4}
                temp_l = np.vectorize(map.get)(temp_l)  # Map events to unique labels
            eeg_labels = np.append(eeg_labels, temp_l, axis=0)
        else:
            temp_eeg = pd.read_csv(f_path, header=None, dtype="float32", delimiter=',')
            temp_eeg = temp_eeg.T
            temp_eeg = temp_eeg.to_numpy()
            eeg_data = np.append(eeg_data, temp_eeg, axis=0)

    ## Data Normalization

    scale = p.MinMaxScaler(feature_range=(-1, 1))
    scale.fit(eeg_data)

    eeg_scaled = scale.transform(eeg_data)

## Windowing and feature extraction

fs = 160  # Sampling rate

num_win = int(eeg_scaled.shape[0] / 160)  # Number of 1-second long windows

eeg_feat = np.empty((num_win, 197))  # Array to store eeg features
feat_labels = np.empty((1, num_win))

for i in range(num_win):  # Go through each second of data

    start = i * fs  # Start of sample
    end = start + fs  # End of sample

    epoch = eeg_scaled[start:end]  # Extract the epoch
    feat_labels[1, i] = mode(eeg_labels[start:end])  # Get the dominant label for epoch

    freq, psd = sig.welch(epoch, fs, nperseg=160)  # Get power spectral density and frequency
    psd = np.mean(psd, axis=0)  # Use the average psd
    eeg_feat[i, 0:33] = psd  # Save the average PSD

    mean = np.mean(epoch)  # Get the mean amplitude
    eeg_feat[i, 34] = mean  # Save the mean

    std = np.std(epoch)  # Get the standard deviation
    eeg_feat[i, 35] = std  # Save the standard deviation

    var = np.var(epoch)  # Get the variance
    eeg_feat[i, 36] = var  # Save the variance

    cor = np.corrcoef(epoch)  # Get the Pearson coefficient
    cor = np.mean(cor, axis=0)  # Get average coefficients
    eeg_feat[i, 37:197] = cor  # Save the average coefficients

scale.fit(eeg_feat)  # Second normalization to keep scale consistent
eeg_feat = scale.transform(eeg_feat)

## Parameter selection

seed = 76

lin_svm = {'C': [0.005, 0.01, 0.015, 0.1]}
rbf_svm = {'C': [0.005, 0.01, 0.015, 0.1], 'gamma': ['scale', 'auto']}
knn = {'n_neighbors': [1, 2, 5, 10, 20], 'leaf_size': [10, 15, 30, 45, 60], 'p': [1, 2, 3], 'n_jobs': [-1]}
rf = {'n_estimators': [10, 20, 50, 100, 150], 'n_jobs': [-1]}

lin_svm_param = ms.GridSearchCV(svm.LinearSVC(), lin_svm, cv=10, n_jobs=-1)
lin_svm_param.fit(eeg_feat, feat_labels)
rbf_svm_param = ms.GridSearchCV(svm.SVC(kernel='rbf'), rbf_svm, cv=10, n_jobs=-1)
rbf_svm_param.fit(eeg_feat, feat_labels)
knn_param = ms.GridSearchCV(neighbors.KNeighborsClassifier(), knn, cv=10, n_jobs=-1)
knn_param.fit(eeg_feat, feat_labels)
rf_param = ms.GridSearchCV(ensemble.RandomForestClassifier(), rf, cv=10, n_jobs=-1)
rf_param.fit(eeg_feat, feat_labels)

lsvm_p = lin_svm_param.best_params_
rsvm_p = rbf_svm_param.best_params_
knn_p = knn_param.best_params_
rf_p = rf_param.best_params_

lsvm_s = lin_svm_param.best_score_
print(lsvm_s)
rsvm_s = rbf_svm_param.best_score_
print(rsvm_s)
knn_s = knn_param.best_score_
print(knn_s)
rf_s = rf_param.best_score_
print(rf_s)

## Data Splitting

eeg_train, eeg_test, y_train, y_test = ms.train_test_split(eeg_feat,
                                                           feat_labels,
                                                           test_size=0.15,
                                                           random_state=seed,
                                                           shuffle=True)  # Randomly split the data

## Find the performance of different classifiers

lsvm_mdl = svm.LinearSVC(C=lsvm_p['C'], random_state=seed)

lsvm_mdl.fit(eeg_train, y_train)

print("Linear SVM model fitted")

lsvm_pred = lsvm_mdl.predict(eeg_test)

print("Linear SVM predictions done")

## rbf SVM

svm_mdl = svm.SVC(C=rsvm_p['C'], gamma=rsvm_p['gamma'], random_state=seed)

svm_mdl.fit(eeg_train, y_train)

print("SVM model fitted")

svm_pred = svm_mdl.predict(eeg_test)

print("SVM predictions done")

## kNN

knn_mdl = neighbors.KNeighborsClassifier(n_neighbors=knn_p['n_neighbors'], n_jobs=-1,
                                         leaf_size=knn_p['leaf_size'], p=knn_p['p'])

knn_mdl.fit(eeg_train, y_train)

print("kNN model fitted")

knn_pred = knn_mdl.predict(eeg_test)

print("kNN predictions done")

## RF

rf_mdl = ensemble.RandomForestClassifier(n_estimators=rf_p['n_estimators'], n_jobs=-1,
                                         random_state=seed)

rf_mdl.fit(eeg_train, y_train)

print("RF model fitted")

rf_pred = rf_mdl.predict(eeg_test)

print("RF predictions done")

## Compare results

print("Scores for linear SVM" "")
print("Accuracy: %f " % (m.accuracy_score(y_test, lsvm_pred)))
print("Recall: " + str(m.recall_score(y_test, lsvm_pred, average=None)))
print("F1-score: " + str(m.f1_score(y_test, lsvm_pred, average=None)))

print("Scores for SVM")
print("Accuracy: %f " % (m.accuracy_score(y_test, svm_pred)))
print("Recall: " + str(m.recall_score(y_test, svm_pred, average=None)))
print("F1-score: " + str(m.f1_score(y_test, svm_pred, average=None)))

print("Scores for kNN")
print("Accuracy: %f " % (m.accuracy_score(y_test, knn_pred)))
print("Recall: " + str(m.recall_score(y_test, knn_pred, average=None)))
print("F1-score: " + str(m.f1_score(y_test, knn_pred, average=None)))

print("Scores for RF")
print("Accuracy: %f " % (m.accuracy_score(y_test, rf_pred)))
print("Recall: " + str(m.recall_score(y_test, rf_pred, average=None)))
print("F1-score: " + str(m.f1_score(y_test, rf_pred, average=None)))
