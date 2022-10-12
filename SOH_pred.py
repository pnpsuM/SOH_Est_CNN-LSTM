from SOH_func import *
import matplotlib.pyplot as pl
from tensorflow import keras
from keras import models, layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
# from sklearn.preprocessing import StandardScaler

# %load_ext tensorboard
FILE_NAME = 'CYCLE_CSV_data.csv'
# drop_labels = ['시험_시간(s)', '사이클_번호', '사이클_시간(s)', '단계_번호', '단계_시간(s)', '인덱스', '보조전압1(V)', '보조전압2(V)', '보조전압3(V)', '온도(\'C)', '누적_용량(Ah)', '전류_범위', 'OCP(V)', '파워(W)', '부하(Ohm)']
drop_labels_x = ['인덱스', '사이클_번호', '충전_용량(Ah)', '누적_용량(Ah)', '충전_에너지(Wh)', '누적_에너지(Wh)', '쿨롱_효율(%)', '에너지_효율(%)', '최대_전압(V)', '충전_최종전압(V)', '단위_충전_용량(Ah/g)', '단위_방전_용량(Ah/g)']
drop_labels_y = ['인덱스', '사이클_번호', '충전_용량(Ah)', '누적_용량(Ah)', '충전_에너지(Wh)', '방전_에너지(Wh)', '누적_에너지(Wh)', '쿨롱_효율(%)', '에너지_효율(%)', '최대_전압(V)', '충전_최종전압(V)', '방전_최종전압(V)', '단위_충전_용량(Ah/g)', '단위_방전_용량(Ah/g)']
y_label = '절대값_용량(Ah)'

data, data_cap = get_data(FILE_NAME, drop_labels_x, drop_labels_y)
print(f'data = {data.shape}')
print(f'data_cap = {data_cap.shape}')
pl.plot(data_cap)
pl.show()

seq_len = 25
sample_len = 8
num_units = 30
num_filters = 6
window = 3
drop_rate = 0.2
num_epochs = 10000
x_data, y_data, num_batch = seq_gen(data, data_cap, seq_len)
x_train, y_train, num_batch = seq_gen(x_data, y_data, sample_len)
print(x_train.shape)

model = models.Sequential()
model.add(layers.ConvLSTM1D(num_filters, window, return_sequences=True, padding="same", input_shape = (None, x_train.shape[-2], x_train.shape[-1])))
model.add(layers.Dense(1))
model.compile(loss = 'mse', optimizer = 'Adam')
model.summary()

epoch_index = 0
loss_dict = {}
epoch_index += num_epochs
callback_list = [ModelCheckpoint(filepath = f'Checkpoints\SOC_Checkpoint_{epoch_index}.h5', monitor = 'val_loss', save_best_only = True)]
fitdata = model.fit(x_train, y_train, epochs=num_epochs, verbose = 1, validation_split=0.2, callbacks=callback_list)