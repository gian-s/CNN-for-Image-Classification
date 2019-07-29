#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

#NAME = "first-CNN-{}".format(int(time.time()))


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = .333)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layers = [0]
layer_sizes = [32]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense".format(conv_layer,layer_size, dense_layer,int(time.time()))
            tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))
            print(NAME)
            


            model = Sequential()

            model.add(Conv2D(layer_size, (3,3),input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))
            
            for l in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            
            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
                
            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss = "binary_crossentropy",
                         optimizer = "adam",
                         metrics = ['accuracy'])

            model.fit(X,y,batch_size = 32, epochs=20, validation_split = .3, callbacks=[tensorboard])

model.save("cats_dogs.model")
           


# In[ ]:





# In[ ]:



