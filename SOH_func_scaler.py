import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0.1, 1.1))

def get_data(NAME : str, drop_labels_x : list, drop_labels_y : list, regularization = True):
    """Reads .csv data and returns data(up to 2D) and data_y(1D), data regularized by 0 ~ 1.

    Args:
        NAME (str): File location
        drop_labels_x (list): Labels to be dropped from original data (to form an input data)
        drop_labels_y (list): Labels to be dropped from original data (to form an output data)

    Returns:
        list, list: Input and output data before the sequence generation
    """
    
    data = pd.read_csv(NAME)
    data_y = data.copy()
    data = data.drop(drop_labels_x, axis = 1)
    data_y = data_y.drop(drop_labels_y, axis = 1)
    print(data.columns)
    print(data_y.columns)
    data = data[35:]
    data_y = data_y[35:]
    if regularization:
        scaler.fit(data)
        data = scaler.transform(data)
        scaler.fit(data_y)
        data_y = scaler.transform(data_y)
    else:
        data = data.values
        data_y = data_y.values
    print(f'{NAME} => {data.shape}')
    pl.figure(dpi = 50)
    pl.plot(data_y)
    pl.show()
    
    return data, data_y

def seq_gen_x(data_x, seq_len = 5):
    """Gets input data and returns '+1 dimensional' datas divided by seq_len(sequence length).

    Args:
        data_x (list): Input data
        seq_len (int, optional): sequence length. Defaults to 5.

    Returns:
        np.array: sequence-divided input data
    """
    x_data = []
    for i in range(seq_len, len(data_x)):
        x_data.append(data_x[i - seq_len:i])
    x_data = np.array(x_data).astype(np.float32)
    
    return x_data

def seq_gen_y(data_y, seq_len = 5):
    """Gets output data and returns '+1 dimensional' datas divided by seq_len(sequence length).

    Args:
        data_y (list): Output data
        seq_len (int, optional): sequence length. Defaults to 5.

    Returns:
        np.array: sequence-divided output data
    """
    y_data = []
    for i in range(seq_len, len(data_y)):
        y_data.append([data_y[i]])
    y_data = np.array(y_data).astype(np.float32)
    
    return y_data

def split_data(x_data, y_data):
    
    split_len = int(round(x_data.shape[0] * 0.8))
    print(f'split_len = {split_len}')
    x_train = x_data[:, :, :split_len]
    y_train = y_data[:, :, :split_len]
    x_test = x_data[:, :, split_len:]
    y_test = y_data[:, :, split_len:]
    print(f'x_train = {x_train.shape}')
    print(f'y_train = {y_train.shape}')
    
    return x_train, y_train, x_test, y_test

def flatten_2Dto1D(data):
    data_flatten = data.reshape(-1, 1)
    
    return data_flatten

def prove(model, h5_path, x_data, y_data):
    model.load_weights(h5_path)
    prediction = model.predict(x_data)
    
    prediction_graph = flatten_2Dto1D(prediction)
    y_graph = flatten_2Dto1D(y_data)
    
    Error_rate = []
    Error = []
    for step in range(len(prediction_graph)):
        Error.append((prediction_graph[step] - y_graph[step]))
        Error_rate.append(Error[step] / y_graph[step] * 100)
    
    RMSE_total = np.sqrt(np.mean(np.square(Error)))
    MAE_total = np.mean(np.absolute(Error))
    
    return RMSE_total, MAE_total, Error_rate, prediction_graph, y_graph
        
def show_and_prove(model, h5_path, x_data, y_data, save_path, return_loss = False, show_y = True, plot = True):
    """Shows prediction and y data graphs. Also returns RMSE, MAE, and Error-by-steps.

    Args:
        model (tf.Model): Defined model
        h5_path (str) : .h5 file directory path
        x_data (np.array): Input data for the prediction
        y_data (np.array): Desired output data
        save_path (str) : Directory path to save the graph plots
        return_loss (bool, optional): return RMSE & MAE loss(list) if True. Defaults to False.
        show_y (bool, optional): y_data graph will be also plotted if True. Defaults to True.

    Returns:
        int, int, list: RMSE, MAE, Error rate by cycle steps.
    """
    # param = {'seq_len' : 25, 'sample_len' : 25, 'num_units' : 64, 'num_filters' : 64, 'window' : 3, 'drop_rate' : 0.2, 'num_epochs' : 800}
    RMSE_total, MAE_total, Error_rate, prediction_graph, y_graph = prove(model, h5_path, x_data, y_data)
    
    print(prediction_graph.shape)
    print(save_path)
    if plot:
        pl.figure(dpi=150)
        pl.ylabel('SOH Est.-Ref. Comparison')
        pl.xlabel('Cycles')
        line = pl.plot(prediction_graph, label = 'SOH Estimation')
        pl.setp(line, linewidth=0.5)
        if show_y:
            y_line = pl.plot(y_graph, label = 'SOH Reference')
            pl.setp(y_line, linewidth=0.5)
        pl.legend()
        pl.savefig(f'{save_path}\Estimation-RMSE({RMSE_total:.4f})MAE({MAE_total:.4f}).png')
        pl.show()
    
    if return_loss:
        return RMSE_total, MAE_total, Error_rate