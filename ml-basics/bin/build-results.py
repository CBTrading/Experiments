#!/usr/bin/env python

import os, sys
import pandas as pd

ipath = "/home/mejia/Downloads/"
opath = "../data/processed/"
if len(sys.argv) > 1:
    filename = opath + sys.argv[1]
else:
    sys.exit(1)

dfs = [pd.read_csv(ipath+file) for root, subs, files in os.walk(ipath) for file in files if file.startswith("ML SP500") and file.endswith(".csv")]

kws = "AP BPM BDT DF DJ LR NN SVM".split()
data = pd.DataFrame(data={"Class": dfs[0]["class"]})
for i, df in enumerate(dfs):
    data["Labels_{}".format(kws[i])] = df["Scored Labels"]
    data["Probabilities_{}".format(kws[i])] = df["Scored Probabilities"]

data.to_csv(filename)
