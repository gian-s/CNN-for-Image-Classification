#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import tensorflow as tf

categories = ["Dogs","Cats"]

def prepare(filepath):
    img_size = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


model = tf.keras.models.load_model("cats_dogs.model")
prediction = model.predict([prepare('lola.jpg')])
print(prediction)


# In[3]:


prediction = model.predict([prepare('C:/Users/gianc/lola.jpg')])
print(prediction)
print(categories[int(prediction[0][0])])


# In[ ]:




