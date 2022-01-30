#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

"""
parse dataset for dnn input
"""

# number of parity measurements
nrip = 10

features = np.load("./data/features.npy")
y = np.load("./data/labels.npy")


# Encode measurements outcomes
stati = ["00", "01", "10","11"]
X = []
for line in features:
    dato = []
    for me in line[0].split(" ")[::-1]:
        dato.append(stati.index(str(me)))
    X.append(dato)


df = pd.DataFrame(X, columns=[str(i) for i in range(nrip)])

# Encoding columns of labels matrices
labels = np.zeros((len(y),nrip))
for d in range(len(y)):
    for i in range(nrip):
        if np.max((y[d][:,i]))> 0:
            labels[d][i] = np.argmax(y[d][:,i]) +1
df["label"] = list(labels)

# save DataFrame
df.to_csv('artificial_dataset.csv')

