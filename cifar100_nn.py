#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import talos
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

p = {
    'units': [120, 240],
    'hidden_activations': ['relu', 'sigmoid'],
    'activation': ['softmax', 'sigmoid'],
    'loss': ['mse', 'categorical_crossentropy'],
    'optimizer': ['adam', 'adagrad'],
    'batch_size': [1000, 2000]
}

# Load CIFAR-10 data
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

target_train = tf.keras.utils.to_categorical(target_train, 100)
target_test = tf.keras.utils.to_categorical(target_test, 100)

def my_model(input_train, target_train, x_val, y_val, params):
    # Create the model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(params['units'], input_shape=(32,32,1)))
    model.add(Dense(params['units'], activation=params['activation']))
    model.add(Dense(params['units'], activation=params['hidden_activations']))
    model.add(Dense(params['units'], activation=params['hidden_activations']))
    model.add(Dense(100, activation=params['activation']))

    # Compile the model
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    # Fit data to model
    history = model.fit(input_train, target_train,
                batch_size=params['batch_size'],
                epochs=20,
                validation_data=(x_val, y_val),
                verbose=1)
    return history, model

    # Generate generalization metrics
    score = model.evaluate(input_test, target_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    
    plt.plot(history.history['val_loss'])
    plt.title('Validation loss history')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.show()

    # Plot history: Accuracy
    plt.plot(history.history['val_accuracy'])
    plt.title('Validation accuracy history')
    plt.ylabel('Accuracy value (%)')
    plt.xlabel('No. epoch')
    plt.show()

talos.Scan(input_train, target_train, p, my_model, x_val=input_test, y_val=target_test, experiment_name="talos_output")


# In[4]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




