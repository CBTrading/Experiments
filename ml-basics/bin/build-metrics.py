#!/usr/bin/env python

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

opath = "../data/processed/"
if len(sys.argv) > 1:
    filename = opath + sys.argv[1]
else:
    sys.exit(1)

sns.set(context="notebook", style="darkgrid", palette="muted", color_codes=True)

def sensitivity(tp, tn, fp, fn):
    return tp*1.0 / (tp + fn)

def specificity(tp, tn, fp, fn):
    return tn*1.0 / (tn + fp)

def accuracy(tp, tn, fp, fn):
    return (tp + tn)*1.0 / (tp + tn + fp + fn)

def precision(tp, tn, fp, fn):
    return tp*1.0 / (tp + fp)

def recall(tp, tn, fp, fn):
    return tp*1.0 / (tp + fn)

ml_results = pd.read_csv(filename)

models = {
    "AP": "Averaged Perceptron",
    "BPM": "Bayes Point Machine",
    "BDT": "Boosted Descision Tree",
    "DF": "Decision Forest",
    "DJ": "Decision Jungle",
    "LR": "Logistic Regression",
    "NN": "Neural Network",
    "SVM": "Support Vector Machine"
}
model_evaluation = pd.DataFrame(
    index=models.values(),
    columns=["Sensitivity", "Specificity", "Accuracy", "Precision"],
    data=np.nan
)
for kw in models:
    tn, fp, fn, tp = confusion_matrix(ml_results["Class"], ml_results["Labels_{}".format(kw)]).ravel()

    model_evaluation.loc[models[kw], "Sensitivity"] = sensitivity(tp, tn, fp, fn)
    model_evaluation.loc[models[kw], "Specificity"] = specificity(tp, tn, fp, fn)
    model_evaluation.loc[models[kw], "Accuracy"] = accuracy(tp, tn, fp, fn)
    model_evaluation.loc[models[kw], "Precision"] = precision(tp, tn, fp, fn)

model_evaluation.to_clipboard(excel=True)
