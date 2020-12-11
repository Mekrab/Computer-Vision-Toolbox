#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import os

from collections import defaultdict
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')

#Push physical devices list for GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
# Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
# Invalid device or cannot modify virtual devices once initialized.
    pass


# In[3]:


# Ideas for looking into a linux directory and getting files : GLOB
def read_image_files(image_path, conversion=cv2.COLOR_BGR2RGB, image_extensions = ['.JPG', '.JPEG', '.BMP', '.PNG']):
    image_files = glob.glob(os.path.join(image_path,'*'))
    image_files = [item for item in image_files if any([ext in item.upper() for ext in image_extensions])]        
    image_list = [(os.path.basename(f),cv2.imread(f,conversion)) for f in image_files]
    image_dict = {file:image for (file,image) in image_list}
    return image_dict

def generate_histogram(image_dict, number_bins=8):
    histogram_dict = dict()
    for filename in image_dict:
        image = image_dict[filename]        
        hist0 = cv2.calcHist([image], [0], None, [number_bins], [0, 256])
        hist1 = cv2.calcHist([image], [1], None, [number_bins], [0, 256])
        hist2 = cv2.calcHist([image], [2], None, [number_bins], [0, 256])
        overall_hist = np.concatenate([hist0,hist1,hist2]).ravel()
        
        hist = overall_hist / overall_hist.sum()
        histogram_dict[filename] = hist        
    return histogram_dict


# In[5]:


image_path = 'PATH'
class_list = ['apple','banana','blueberry','lime']

# Process the target images
image_target_dict = read_image_files(os.path.join(image_path,'targets'))
hist_target_dict = generate_histogram(image_target_dict)

image_class_dict = dict()
hist_class_dict = dict()
for c in class_list:
    image_class_dict[c] = read_image_files(os.path.join(image_path,'classes',c))
    hist_class_dict [c] = generate_histogram(image_class_dict[c])
#print(image_class_dict)
# Process the test images


# In[6]:


OPENCV_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL, True),
        ("Chi-Squared", cv2.HISTCMP_CHISQR, False),
        ("Intersection", cv2.HISTCMP_INTERSECT, True,),
        ("Hellinger", cv2.HISTCMP_BHATTACHARYYA, False))


# In[7]:


def compare_histogram(image_name, hist_base, histogram_dict):
    results_dict = dict()
    for (methodName, method, reverse) in OPENCV_METHODS: 

        results = {k : cv2.compareHist(hist_base, hist, method) for (k,hist) in sorted(histogram_dict.items())}
        #print(type(results))
        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)
        #print(type(results))

        results_dict[methodName] = results
        #print(results_dict[methodName][class_name[i]])
        #print (image_name,methodName,results)
        # show the query image
        #target_img = plt.imread(os.path.join(image_path,'targets//' + image_name))
        #plt.imshow(target_img)
        #plt.show()
        #i = i + 1
        
    return results_dict


# In[8]:


# Setup a triple dictionation [class_name of a target][class_name][histogram_name]
target_class_compare_dict = defaultdict(dict)

print(hist_target_dict)

for (image_name, hist_base) in hist_target_dict.items():
    #print(image_name)
    # convert to target name
    target_class = image_name.split('.')[0].split('_')[1]
    #print(target_class)
    
    for class_name in class_list:

        target_class_compare_dict[target_class][class_name] = compare_histogram(class_name, hist_target_dict[image_name], hist_class_dict[class_name])
        #print(target_class_compare_dict[target_class][class_name])
print(target_class_compare_dict.keys())


# In[9]:


print('Apples to Apples \n\n')
print ('Correlation:\n',target_class_compare_dict['apple']['apple']["Correlation"])
print ('Chi-Squared:\n',target_class_compare_dict['apple']['apple']["Chi-Squared"])
print ('Intersection:\n',target_class_compare_dict['apple']['apple']["Intersection"])
print ('Hellinger:\n',target_class_compare_dict['apple']['apple']["Hellinger"])


# In[10]:


for (methodName, method, reverse) in OPENCV_METHODS:
    plt.figure(figsize=(10, 6))

    for index, image_name in enumerate(class_list):
        comparison_list = [x for (x,y) in target_class_compare_dict[image_name][class_name][methodName]]
        plt.plot(comparison_list, marker='',linewidth=1, alpha=0.4) #label = x_axis_values)
    plt.legend(class_list, loc='upper right')
    plt.ylabel('score')
    plt.xlabel('image count')
    plt.title('Class Self Compare ' + methodName)
    plt.show()


# In[15]:


# Compare all of the classes

for class_index in class_list:
    for (methodName, method, reverse) in OPENCV_METHODS:
        plt.figure(figsize=(10, 6))

        for index, class_name in enumerate(class_list):
            comparison_list = [x for (x,y) in target_class_compare_dict[class_index][class_name][methodName]]
            plt.plot(comparison_list, marker='',linewidth=1, alpha=0.4) #label = x_axis_values)
        plt.legend(class_list, loc='upper right')
        plt.ylabel('score')
        plt.xlabel('image count')
        plt.title('Inter-class compare ' + class_index + ' '+ methodName)
        plt.show()


# In[14]:


# Let's first find out the comparison of scores : AVGERAGE, MEDIAN, MAX, MIN


def calculate_score_range(histogram_scores_tuple):
    scores = [score for (score,filename) in histogram_scores_tuple]
    avg_score = sum(scores)/len(scores)
    median_score = scores[int(len(scores)/2)]   # Assume the scores are sorted 
    max_score = max(scores)
    min_score = min(scores)
    return (avg_score, median_score, max_score, min_score)


target_name = 'apple'
print (target_name)
for (methodName, method, reverse) in OPENCV_METHODS: 
    # Check the base class: target_name with itself
    print ( methodName)

    print (target_name, end="")
    print (calculate_score_range(target_class_compare_dict[target_name][target_name][methodName]))
    
    # Check the base class with every other class name (not itself)
    process_list = [class_name for class_name in class_list if class_name != target_name]
    
    for class_name in process_list:
        print (class_name, end="")
        print (calculate_score_range(target_class_compare_dict[target_name][class_name][methodName]))

    print ()
    
target_name = 'banana'
print (target_name)
for (methodName, method, reverse) in OPENCV_METHODS: 
    # Check the base class: target_name with itself
    print ( methodName)

    print (target_name, end="")
    print (calculate_score_range(target_class_compare_dict[target_name][target_name][methodName]))
    
    # Check the base class with every other class name (not itself)
    process_list = [class_name for class_name in class_list if class_name != target_name]
    
    for class_name in process_list:
        print (class_name, end="")
        print (calculate_score_range(target_class_compare_dict[target_name][class_name][methodName]))

    print ()



# In[ ]:




