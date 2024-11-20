
# coding: utf-8

# In[1]:

import os
import random
import numpy as np
import matplotlib.pyplot 
from matplotlib.pyplot import imshow, imsave
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance

from sklearn.externals import joblib
from keras.preprocessing import image
import cv2
from PIL import Image


# In[ ]:



# loading saved classifier  
clf_sign = joblib.load('clf_sign.pkl') 
import os


# # Loading model online 

# In[3]:


model = keras.applications.VGG16(weights='imagenet', include_top=True)
print("Loaded model online")


# # loading model offline

# In[899]:


# load json and create model
#json_file = open('prv_VGG16_architecture.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#from keras.models import model_from_json
#model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5")
#print("Loaded model from disk")


# In[714]:


# get_image will return a handle to the image itself, and a numpy array of its pixels to input the network
def get_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #ret, thresh = cv2.threshold(np.array(img), 200, 255, cv2.THRESH_BINARY)
    #return thresh, x
    return img,x


# In[715]:


feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
#feat_extractor.summary()


# # path for probable signatures identified by image processing methods 

# In[1048]:


#images_path = 'C:/Users/1311921/Desktop/Signature_final/Test Documents/temp/GATE12/prob_stamps/'
images_path = "/home/pyimagesearch/Desktop/Signature_final/Test Documents/DETECTED/03/prob_signature/" 



# automate for all
# find all directories 


# In[1049]:


max_num_images = 100
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]
print("keeping %d images to analyze" % len(images))


# # Display and save the probable signature for classification 

# In[1050]:


features = []
store_img = []
for image_path in (images):
    #print(image_path)
    
    img, x = get_image(image_path);    
    # display the resulting images
    matplotlib.pyplot.figure(figsize = (3,4))
    feat = feat_extractor.predict(x)[0]
    features.append(feat)
    imshow(img)  
    store_img.append(img)
features = np.array(features)


# # Classify wheather it is a sign or not 

# In[1051]:


#clf_sign.predict(features)


# In[1052]:


indices = [i for i, x in enumerate(clf_sign.predict(features)) if x == 1]
indices


# In[1053]:


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[1054]:


c=0
for image_path in (images):  
    #print(image_path)
    if c in indices:               
        img, x = get_image(image_path);        
        # display the resulting images
        #matplotlib.pyplot.figure(figsize = (4,4)) 
        #imshow(store_img[c])  
        sign_image_path = images_path + "Sign/"
        ensure_dir(sign_image_path)
        image_name = (sign_image_path +str(c) + ".png")
        ret, thresh = cv2.threshold(np.array(store_img[c]), 200, 255, cv2.THRESH_BINARY) 
        imsave(image_name, thresh)
    c = c+1


# # Deep learning / CNN features extracted and check similarity

# In[1055]:


#reference_sign_database_path = 'C:/Users/1311921/Desktop/Signature_final/sample signs/'
reference_sign_database_path = "/home/pyimagesearch/Desktop/Signature_final/database reference signatures/"


# In[1056]:


# Reference signatures 

max_num_images = 35
#print(reference_sign_database_path)
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(reference_sign_database_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]   

print("keeping %d reference signatures to analyze" % len(images))

# Extracted signatures for testing 


#print(sign_image_path)
max_num_images = 35

S_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(sign_image_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
if max_num_images < len(images):
    S_images = [S_images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]   

print("keeping %d test signatures to analyze" % len(S_images))


# In[1057]:


features = []
for image_path in (images):
    img, x = get_image(image_path);    
    feat = feat_extractor.predict(x)[0]
    features.append(feat)
 
test_features = []
for_display_test = []
for imagex in (S_images):
    img, x = get_image(imagex);
    for_display_test.append(img)
    feat_test = feat_extractor.predict(x)[0]
    test_features.append(feat_test)
    
pca_features = features


# In[1058]:


def get_closest_images(query_image_feat , name, num_results):
    distances = [ distance.euclidean(query_image_feat, feat) for feat in pca_features ]
    #print(distances)    
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:num_results]
    print("min distance sign: " + str(min(distances)))
    aaa = (np.argsort(distances)[:num_results])
    
    d = []
    for i in range(0,num_results):
        #print(distances[aaa[i]])
        d.append(distances[aaa[i]])
        
    # save to a text 

    text_file = open( name +".txt", "w")
    text_file.write("Similar Signatures based on similarity: \n"+ str(d))
    text_file.close()
    return idx_closest

def get_concatenated_images(indexes, name, thumb_height ):
    thumbs = []
    for idx in indexes:
        img = cv2.imread(images[idx],0)         
        img = Image.fromarray(img)
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    
    p = name + ".png"
    cv2.imwrite(p, concat_image)
    return concat_image


# In[1059]:


# automate for all

print(" If min distance < 40 : Match is perfect")
print(" If min distance < 60 & > 40  : Match is possible but chance of error high")
print(" If min distance > 60  : Similar signature not present or may be it is not even a valid signature")


for i in range(0,len(test_features)):
    query_image_idx = i
    print(query_image_idx)
    query_image_feat = test_features[query_image_idx]
    ensure_dir(sign_image_path + "SIMILARITY/")

    name = sign_image_path + "SIMILARITY/"+ "similar_to_" + str(query_image_idx)

    #print("Query image: ")
    #matplotlib.pyplot.figure(figsize = (4,4))
    #imshow(for_display_test[query_image_idx])
    imsave(name + "_original.png", for_display_test[query_image_idx])

    idx_closest = get_closest_images(query_image_feat , name, 3)

    results_image = get_concatenated_images(idx_closest, name ,400)
    #print("Similar Signatures based on similarity: ")
    #imshow(results_image)


# #### END #######
#NB : Please go the respective folders 
