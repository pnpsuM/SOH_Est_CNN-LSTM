import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

def get_data(NAME : str, drop_labels_x : list, drop_labels_y : list):
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

def show_and_prove(model, VERSION, epoch_index, x_train, y_train, loss_dict, param):
    # param = {'seq_len' : 25, 'sample_len' : 25, 'num_units' : 64, 'num_filters' : 64, 'window' : 3, 'drop_rate' : 0.2, 'num_epochs' : 800}
    model.load_weights(f'Checkpoints\SOH_{VERSION}_Checkpoint.h5')
    loss = model.evaluate(x_train, y_train)
    loss_dict[f'loss_{epoch_index}'] = loss
    prediction = model.predict(x_train)
    prediction = prediction.reshape(int(prediction.shape[0] * prediction.shape[1] * prediction.shape[2]), 1) 
    y_graph = y_train.reshape(int(y_train.shape[0] * y_train.shape[1] * y_train.shape[2]), 1)
    print(prediction.shape, y_graph.shape)
    print(f'Estimation{VERSION}-{param["num_filters"]}FL-{param["num_units"]}UN-{epoch_index}EP-{param["seq_len"]}SQ')
    pl.figure(dpi=150)
    pl.plot(prediction)
    pl.plot(y_graph)
    pl.savefig(f'output\Estimation{VERSION}-{param["num_filters"]}FL-{param["num_units"]}UN-{epoch_index}EP-{param["seq_len"]}SQ.png')
    pl.show()