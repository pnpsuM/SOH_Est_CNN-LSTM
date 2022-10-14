import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

def get_data(NAME : str, drop_labels_x : list, drop_labels_y : list):
    """Reads .csv data and returns data(up to 2D) and data_y(1D) 

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
    print(data.columns)
    data_y = data_y.drop(drop_labels_y, axis = 1)
    print(data_y.columns)
    data = data.values
    data_y = data_y.values
    
    return data, data_y

def seq_gen(data_x, data_y, seq_len = 5):
    """Gets input and output data and returns '+1 dimensional' datas divided by seq_len(sequence length).

    Args:
        data_x (list): Input data
        data_y (list): Output data
        seq_len (int, optional): sequence length. Defaults to 5.

    Returns:
        np.array, np.array, int: sequence-divided input and output data, and # of batches
    """
    num_batch = int(np.floor(data_x.shape[0] / seq_len))
    print(f'num_batch = {num_batch}')
    x_data = []
    y_data = []
    for batch in range(num_batch):
        x_data.append(data_x[batch * seq_len:(batch + 1) * seq_len])
        y_data.append(data_y[batch * seq_len + 1:(batch + 1) * seq_len + 1])
    x_data = np.array(x_data).astype(np.float32)
    y_data = np.array(y_data).astype(np.float32)
    print(f'x = {x_data.shape}')
    print(f'y = {y_data.shape}')
    
    return x_data, y_data, num_batch

def split_data(x_data, y_data, num_batch):
    
    split_len = int(round(num_batch * 0.8))
    print(f'split_len = {split_len}')
    x_train = x_data[:, :, :split_len]
    y_train = y_data[:, :, :split_len]
    x_test = x_data[:, :, split_len:]
    y_test = y_data[:, :, split_len:]
    print(f'x_train = {x_train.shape}')
    print(f'y_train = {y_train.shape}')
    
    return x_train, y_train, x_test, y_test

def show_and_prove(model, VERSION, file_path, epoch_index, x_data, y_data, loss_dict, param, return_loss = False, show_y = True):
    """Shows prediction and y data graphs. Also returns loss_dict.

    Args:
        model (tf.Model): Defined model
        VERSION (str): Current Source code version given by custom str.
        file_path (str) : .h5 file directory path
        epoch_index (int): Total epochs # trained
        x_data (np.array): Input data for the prediction
        y_data (np.array): Desired output data
        loss_dict (dict): Dictionary to be containing loss data
        param (dict): Dictionary that contains parameter informations
        return_loss (bool, optional): return loss_dict if True. Defaults to False.
        show_y (bool, optional): y_data graph will be also plotted if True. Defaults to True.
    """
    # param = {'seq_len' : 25, 'sample_len' : 25, 'num_units' : 64, 'num_filters' : 64, 'window' : 3, 'drop_rate' : 0.2, 'num_epochs' : 800}
    model.load_weights(file_path)
    loss = model.evaluate(x_data, y_data)
    loss_dict[f'loss_{epoch_index}'] = loss
    prediction = model.predict(x_data)
    prediction = prediction.reshape(int(prediction.shape[0] * prediction.shape[1] * prediction.shape[2]), 1) 
    y_graph = y_data.reshape(int(y_data.shape[0] * y_data.shape[1] * y_data.shape[2]), 1)
    print(prediction.shape, y_graph.shape)
    print(f'{param["num_filters"]}FL-{param["num_units"]}UN-{epoch_index}EP-{param["seq_len"]}SQ')
    pl.figure(dpi=150)
    pl.plot(prediction)
    if show_y:
        pl.plot(y_graph)
    pl.savefig(f'output\Estimation{VERSION}-{param["num_filters"]}FL-{param["num_units"]}UN-{epoch_index}EP-{param["seq_len"]}SQ.png')
    pl.show()
    
    if return_loss:
        return loss_dict