import numpy as np

def rows_to_numpy(rows):
    if len(rows) <= 1:
        return np.array([])

    data_rows = rows[1:]
    numeric_rows = []
    for row in data_rows:
        numeric_row = [float(value) for value in row]
        numeric_rows.append(numeric_row)

    return np.array(numeric_rows,dtype = np.float32)

def compute_basic_stats(array: np.ndarray) -> dict:
    if array.size == 0:
        return {}
    
    shape = array.shape
    mean = np.mean(array,axis = 0)
    min = np.min(array,axis = 0)
    max = np.max(array,axis = 0)

    return {
        "shape" : shape,
        "mean" : mean,
        "min" : min,
        "max" : max 

    }