from keras.models import Sequential
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from keras.applications.resnet50 import decode_predictions
import math
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


def image_array(filename , size):
    array = cv2.imread(filename)
    array = cv2.resize(array, (size, size)) 
    return array

def prediction(array):
    #size = 224
    #array = image_array(filename , size)
    #print(array.shape)
    array = np.array(array , dtype = np.float64)
    array = np.reshape(array, (1,224,224,3))
    a = preprocess_input(array)
    model = ResNet50(weights='imagenet', include_top=False)
    features = model.predict(a)
    K.clear_session()
    image = features.reshape(features.shape[0] , -1)
    loaded_model = keras.models.load_model('breast_cancer.h5')
    #print(test)
    #a = test[0]
    #a = np.reshape(a , (1,25088))
    #print(image.shape)
    y_predict = loaded_model.predict(image)
    #print(y_predict.shape)
    K.clear_session()
    print('Benign Probability: ', y_predict[0][0])
    print('Malignant Probability: ', y_predict[0][1])
    if y_predict[0][0] > y_predict[0][1]:
        return "Benign", image
    else:
        return "Malignant" , image
    
    
    
def features(array):
    Z = np.reshape(array, (32, 64))
    G = np.zeros((32,64,3))
    G[Z>1] = [1,1,1]
    G[Z<1] = [0,0,0]
    plt.imshow(G,interpolation='nearest')
    plt.show()