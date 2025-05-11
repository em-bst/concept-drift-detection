""""
This is a test script for the three drift detection algorithms (ADWIN, ADWIN2, DDM)
implemented in this project. The script takes the input dataset elecNormNew.csv.
The test is based on Prequential Evaluation for Naive Bayes estimators, and monitors 2 indicators of
performance : accuracy and running time.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import time

from classifier.detector_classifier import DetectorClassifier
from concept_drift.adwin import AdWin, AdWin2
from concept_drift.DDM import DDM
from evaluation.prequential import prequential


def read_data(filename):
    df = pd.read_csv(filename)
    data = df.values
    return data[:, :-1], data[:, -1]


if __name__ == '__main__':
    n_train = 48
    X, y = read_data(r'C:\Users\emmab\Desktop\Data Mining\projet\A-Concept Drift (2)\A-Concept Drift\Project[1]\Project\concept-drift-master\data\elecNormNew.csv')
    # Set x,y as numeric
    X = X.astype(float)
    label = ["UP", "DOWN"]
    le = preprocessing.LabelEncoder()
    le.fit(label)
    y = le.transform(y)

    w = 1000

    clfs = [
        GaussianNB(),
        DetectorClassifier(GaussianNB(), AdWin(), np.unique(y)), 
        DetectorClassifier(GaussianNB(), AdWin2(), np.unique(y)), 
        DetectorClassifier(GaussianNB(), DDM(), np.unique(y))
    ]
    clfs_label = ["GaussianNB", "AdWin", "AdWin2", "DDM"]

    plt.title("Accuracy (exact match)")
    plt.xlabel("Instances")
    plt.ylabel("Accuracy")

    # Iteration through each classifier
    for i in range(len(clfs)):
        print("\n{}:".format(clfs_label[i]))

        start_time = time.time()
        
        # Run the prequential evaluation while mesuring time
        with np.errstate(divide='ignore', invalid='ignore'):
            y_pre, eval_time = prequential(X, y, clfs[i], n_train)

        end_time = time.time()
        
        #Total running time 
        total_time = end_time - start_time
        print(f"Total running time for {clfs_label[i]}: {total_time:.2f} seconds")

    
        if clfs[i].__class__.__name__ == "DetectorClassifier":
            print("Drift detection: {}".format(clfs[i].change_detected))
        
        # Accuracy
        estimator = (y[n_train:] == y_pre) * 1
        acc_run = np.convolve(estimator, np.ones((w,)) / w, 'same')
        print("Mean acc within the window {}: {}".format(w, np.mean(acc_run)))
        
        # Plotting the accuracy
        plt.plot(acc_run, "-", label=clfs_label[i])

    plt.legend(loc='lower right')
    plt.ylim([0, 1])
    plt.show()
