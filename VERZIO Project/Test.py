# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 01:12:39 2020

@author: Kartik
"""

#loading the saved model 
from keras.models import load_model 
model = load_model('project_model.h5')

import numpy as np 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# Give the link of the image here to test 
test_image1 =load_img(r'G:\ship.jfif',target_size =(32,32))

test_image =img_to_array(test_image1) 
test_image =np.expand_dims(test_image, axis =0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Aeroplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')
    
#%matplotlib inline 
import matplotlib.pyplot as plt 
plt.imshow(test_image1)