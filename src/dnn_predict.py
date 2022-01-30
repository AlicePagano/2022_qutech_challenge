from tensorflow import keras
import numpy as np

"""
predict and parse output of dnn model.

input data is a len = 10 + 1 list or array of states 
string (like "00", "01", ..)
"""

# Encode measurements outcomes
def parse_input(data):
    out_data = []
    states = ["00", "01", "10","11"]
    for me in data[0].split(" ")[::-1]:
        out_data.append(states.index(str(me)))
    
    return out_data

# Predict sample
def dnn_predict(model, sample):

    sample = parse_input(sample)
    pred = model.predict(sample)
    cat_pred = np.zeros_like(pred)
    for n,p in enumerate(pred):
        cat_pred[n,:] = [min(round(i), 4) for i in p]

    outputs = parse_output(cat_pred)
    return outputs

def parse_output(output):
    """
    Apply the transformation from the encoded error
    landscape to the full error landscape

    Parameters
    ----------
    output : array-like
        Output of the neural network model
    """
    err_lands = []
    output = np.array(output, dtype=int)
    for out in output:
        error_landscape = np.zeros( (4, 10) )
        for dt, qub in enumerate(out):
            if qub !=0:
                error_landscape[qub-1, dt] = 1
            
        err_lands.append(error_landscape)

    return err_lands