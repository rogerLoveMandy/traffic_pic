# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import time

import argparse
from flyai.dataset import Dataset
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten,MaxPooling2D
from keras.models import Sequential

from model import Model
from path import MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1000, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
args = parser.parse_args()
sqeue = Sequential()
'''
模型需要知道输入数据的shape，
因此，Sequential的第一层需要接受一个关于输入数据shape的参数，
后面的各个层则可以自动推导出中间数据的shape，
因此不需要为每个层都指定这个参数
'''

sqeue = Sequential()

sqeue.add(Conv2D(
    filters=64,
    activation='relu',
    padding='same',
    input_shape=(28,28,1),
    strides=(1,1),
    kernel_size=(3,3)
    ))

sqeue.add(MaxPooling2D(pool_size=(2,2)))
sqeue.add(Dropout(0.5))


sqeue.add(Conv2D(
    filters=256,
    strides=(1,1),
    padding='same',
    kernel_size=(3,3),
    activation='relu'


))


sqeue.add(MaxPooling2D(pool_size=(2,2)))

sqeue.add(Dropout(0.5))

sqeue.add(Flatten())


sqeue.add(Dense(128,activation='relu'))
sqeue.add(Dense(64,activation='relu'))
sqeue.add(Dense(32,activation='relu'))

sqeue.add(Dense(10,activation='softmax'))


# 输出模型的整体信息
# 总共参数数量为784*512+512 + 512*512+512 + 512*10+10 = 669706
sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

dataset = Dataset()
model = Model(dataset)
best_score = 0
for epochs in range(args.EPOCHS):
    first_time = int(time.time())
    x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
    history = sqeue.fit(x_train, y_train,
                        batch_size=args.BATCH,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = sqeue.evaluate(x_test, y_test, verbose=0)
    if score[1] > best_score:
        best_score = score[1]
        model.save_model(sqeue, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (epochs, best_score))
    print(str(epochs + 1) + "/" + str(args.EPOCHS))
