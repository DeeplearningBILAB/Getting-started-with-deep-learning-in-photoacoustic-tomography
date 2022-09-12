import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.layers import  Conv2DTranspose,  concatenate, Input



def Contracting_block(input, filters, kernel_size, padding, activation, dropout = 0.25):
    conv1_1 = Conv2D(filters, kernel_size, activation=activation,
                     padding=padding)(input)  # Convolution layer 1
    conv1_2 = Conv2D(filters, kernel_size, activation=activation,
                     padding=padding)(conv1_1) # Convolutional layer 2
    pool = MaxPooling2D(pool_size=(2, 2))(conv1_2)  # Max pooling layer
    drop_out = Dropout(dropout)(pool) # Dropout
    return [drop_out, conv1_2]


def middle_block(input, filters, kernel_size, padding, activation):
    conv1_1 = Conv2D(filters, kernel_size, activation=activation,
                     padding=padding)(input)  # Convolution layer 1
    conv1_2 = Conv2D(filters, kernel_size, activation=activation,
                     padding=padding)(conv1_1)  # Convolution layer 2
    return conv1_2

def Expansive_block(input, skip, filters, kernel_size,  stride, padding, activation, dropout = 0.5):
    convT_1 = Conv2DTranspose(filters, kernel_size, strides=stride,
                              padding=padding)(input) # Transposed convolution layer
    merge = concatenate([convT_1, skip]) # Concatenation
    drop_out = Dropout(dropout)(merge) # Dropout layer
    conv1_1 = Conv2D(filters, kernel_size, activation=activation,
                     padding=padding)(drop_out)  # Convolution layer 1
    conv1_2 = Conv2D(filters, kernel_size, activation=activation,
                     padding=padding)(conv1_1)  # Convolution layer 2
    return conv1_2



def UNet(input, filters=16, kernel_size=3, padding='same',  activation='relu', strides = (2,2)):

    [out, shortcut1] = Contracting_block(input, filters, kernel_size,
                                         padding, activation)
    [out, shortcut2] = Contracting_block(out, filters * 2, kernel_size,
                                         padding, activation)
    [out, shortcut3] = Contracting_block(out, filters * 2 * 2, kernel_size,
                                         padding, activation)
    [out, shortcut4] = Contracting_block(out, filters * 2 * 2 * 2, kernel_size,
                                         padding, activation)
    out = middle_block(out, filters * 2 * 2 * 2 * 2, kernel_size,
                       padding, activation)
    out = Expansive_block(out, shortcut4, filters * 2 * 2 * 2, kernel_size,
                          strides, padding, activation)
    out = Expansive_block(out, shortcut3, filters * 2 * 2, kernel_size,
                          strides, padding, activation)
    out = Expansive_block(out, shortcut2, filters * 2 , kernel_size,
                          strides, padding, activation)
    out = Expansive_block(out, shortcut1, filters * 2, kernel_size,
                          strides, padding, activation)
    out = Conv2D(1, (1, 1), padding = padding, activation="sigmoid")(out)
    return out

def data_gen(filename):
    with np.load(filename) as data:
        src_images = data['arr_0']
        tar_images = data['arr_1']
    return src_images,tar_images

def plot_history(history, metric, valmetric):
    offset = 0
    data1 = history.history[metric][offset:]
    data2 = history.history[valmetric][offset:]
    epochs = range(offset, len(data1) + offset)
    plt.plot(epochs, data1)
    plt.plot(epochs, data2)
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(metric)
    plt.close()



if __name__=='__main__':
    X_train, Y_train = data_gen('data\\train\\train_set.npz')#load training dataset
    X_valid, Y_valid = data_gen('data\\val\\valid_set.npz')#load valid dataset
    print(np.shape(X_train))
    model_inputs = Input(shape=(128,128,1))#creating input objects
    model_outputs = UNet(model_inputs, filters=16, kernel_size=3,
                         activation='relu', strides = (2,2), padding='same')#creating output objects
    unet_model = Model(model_inputs, model_outputs)#instantiate  the model
    unet_model.compile(optimizer="Adam", loss="mse",
                       metrics=["mae", "accuracy"]) #compiling the U-Net
    unet_model.summary() #visulaizing the layers and its order
    callbacks = [EarlyStopping(patience=10, verbose=1),
                 ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
                 ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True,
                                 save_weights_only=True)]
    history = unet_model.fit(X_train, Y_train, batch_size=2, epochs=100, callbacks=callbacks,
                             validation_data=(X_valid, Y_valid))#fitting the U-Net
    plot_history(history, 'loss', 'val_loss')

