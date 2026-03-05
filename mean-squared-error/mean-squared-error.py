import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    sum = 0
    i = 0;
    while i<len(y_pred):
        calc = pow(y_pred[i]-y_true[i], 2)
        sum+=calc
        i+=1

    final = sum/len(y_pred)
    return final
    pass
