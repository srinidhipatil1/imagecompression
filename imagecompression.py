#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Dense
from keras.models import Model,Sequential
from keras.datasets import mnist


# In[2]:


(X_train,_), (X_test,_)=mnist.load_data()


# In[3]:


X_train=X_train.astype('float32')/float(X_train.max())
X_test=X_test.astype('float32')/float(X_test.max())


# In[4]:


print("Training set : ",X_train.shape)
print("Testing set : ",X_test.shape)


# In[5]:


# Reshaping our images into matrices
X_train=X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
X_test=X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))
print("Training set : ",X_train.shape) #The resolution has changed
print("Testing set : ",X_test.shape)


# In[6]:


input_dim=X_train.shape[1]
encoding_dim=32
compression_factor=float(input_dim/encoding_dim)
 
autoencoder=Sequential()
autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,),activation='relu'))
autoencoder.add(Dense(input_dim,activation='sigmoid'))
 
input_img=Input(shape=(input_dim,))
encoder_layer=autoencoder.layers[0]
encoder=Model(input_img,encoder_layer(input_img))
 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train,X_train,epochs=50, batch_size=256, shuffle=True, validation_data=(X_test,X_test))


# In[7]:


num_images=10
np.random.seed(42)
random_test_images=np.random.randint(X_test.shape[0], size=num_images)
encoded_img=encoder.predict(X_test)
decoded_img=autoencoder.predict(X_test)


# In[8]:


# Display the images and predictions
plt.figure(figsize=(18,4))
 
for i, image_idx in enumerate(random_test_images):
    #plot input image
    ax=plt.subplot(3,num_images,i+1)
    plt.imshow(X_test[image_idx].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
                      
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_img[image_idx].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
 
    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_img[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
                      
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




