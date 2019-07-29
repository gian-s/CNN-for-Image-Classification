#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

data_dir = "C:/Users/gianc/OneDrive/Desktop/PetImages"
CATEGORIES = ["Dogs" , "Cats"]



# In[27]:


img_size = 50
#new_array = cv2.resize(img_array,(img_size, img_size))


# In[30]:


training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(data_dir, category)  #the path to directory for cats and dogs
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(img_size, img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
    
create_training_data()
    
    


# In[31]:


print(len(training_data))


# In[32]:


import random
random.shuffle(training_data)


# In[33]:


for sample in training_data[:10]:
    print(sample[1])


# In[35]:


X = []
y = []


# In[36]:


for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, img_size, img_size, 1)


# In[37]:


import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[40]:


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)


# In[39]:


X[1]


# In[ ]:




