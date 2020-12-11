#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import os
import zipfile

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


# In[2]:


local_zip = 'PATH'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('PATH')
zip_ref.close()


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


# In[27]:


image_path = ['PATH']
class_list = ['PATH']

image_target_dict = read_image_files(os.path.join(image_path,'targetsrotten'))
hist_target_dict = generate_histogram(image_target_dict)

image_class_dict = dict()
hist_class_dict = dict()
for c in class_list:
    image_class_dict[c] = read_image_files(os.path.join(image_path,'classes',c))
    hist_class_dict [c] = generate_histogram(image_class_dict[c])
    
print(image_target_dict)


# In[5]:


OPENCV_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL, True),
        ("Chi-Squared", cv2.HISTCMP_CHISQR, False),
        ("Intersection", cv2.HISTCMP_INTERSECT, True,),
        ("Hellinger", cv2.HISTCMP_BHATTACHARYYA, False))


# In[6]:


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


# In[7]:


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


# In[10]:


#This compares all of the targets with their own class, for each method.


for (methodName, method, reverse) in OPENCV_METHODS:
    plt.figure(figsize=(15, 12))

    for index, class_name in enumerate(class_list):
        comparison_list = [x for (x,y) in target_class_compare_dict[class_name][class_name][methodName]]
        plt.plot(comparison_list, marker='',linewidth=1, alpha=0.4) #label = x_axis_values)
    plt.legend(class_list, loc='upper right')
    plt.ylabel('score')
    plt.xlabel('image count')
    plt.title('Class Self Compare ' + methodName)
    plt.show()


# In[11]:


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


# In[12]:


def calculate_score_range(histogram_scores_tuple):
    scores = [score for (score,filename) in histogram_scores_tuple]
    avg_score = sum(scores)/len(scores)
    median_score = scores[int(len(scores)/2)]   # Assume the scores are sorted 
    max_score = max(scores)
    min_score = min(scores)
    return (avg_score, median_score, max_score, min_score)

#['applefull','applehalf','appleripe','bananafull','bananahalf','bananaripe']


target_name = 'applefull'
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
    
target_name = 'applehalf'
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

target_name = 'appleripe'
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
 


# In[13]:


target_name = 'bananafull'
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
    
target_name = 'bananahalf'
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

target_name = 'bananaripe'
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


# In[16]:



# Process the target images
image_target_dict = read_image_files(os.path.join(image_path,'targetsrotten'))

bin_size_list = [8,16,32,64]

hist_target_bin_dict = defaultdict(dict)
for number_bins in bin_size_list:
    hist_target_bin_dict[number_bins] = generate_histogram(image_target_dict, number_bins=number_bins)

image_class_dict = dict()
hist_class_bin_dict = defaultdict(dict)

for c in class_list:
    image_class_dict[c] = read_image_files(os.path.join(image_path,'classes',c))
    for number_bins in bin_size_list:
        hist_class_bin_dict[number_bins][c] = generate_histogram(image_class_dict[c], number_bins=number_bins)
#print(image_class_dict)
# Process the test images


# In[24]:


bin_target_class_compare_dict = dict()
for number_bins in bin_size_list:
    # Setup a triple dictionation [class_name of a target][class_name][histogram_name]
    target_class_compare_dict = defaultdict(dict)

    for (image_name, hist_base) in hist_target_dict.items():
        #print(image_name)
        # convert to target name
        target_class = image_name.split('.')[0].split('_')[1]
        #print(target_class)
    
        for class_name in class_list:

            target_class_compare_dict[target_class][class_name] = compare_histogram(class_name, hist_target_bin_dict[number_bins][image_name], hist_class_bin_dict[number_bins][class_name])
        #print(target_class_compare_dict[target_class][class_name])
#print(target_class_compare_dict.keys())
    bin_target_class_compare_dict[number_bins] = dict(target_class_compare_dict)
print(bin_target_class_compare_dict.items())


# In[18]:


for (methodName, method, reverse) in OPENCV_METHODS:

    
    for index, class_name in enumerate(class_list):
        plt.figure(figsize=(10, 6))


        for number_bins in bin_size_list:
            comparison_list = [x for (x,y) in bin_target_class_compare_dict[number_bins][class_name][class_name][methodName]]
            plt.plot(comparison_list, marker='',linewidth=1, alpha=0.4) #label = x_axis_values)
        plt.legend([str(x) for x in bin_size_list], loc='upper right')
        plt.ylabel('score')
        plt.xlabel('image count')
        plt.title('Class Self Compare ' + class_name + ' ' + methodName)
        plt.show()


# In[20]:


from csv import reader
from math import sqrt
import numpy as np    
from collections import Counter 

 # Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def euclidean_distance(row1,row2):
    
 

def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors
  
def majority(arr): 
  
    # convert array into dictionary 
    freqDict = Counter(arr) 
  
    # traverse dictionary and check majority element 
    size = len(arr) 
    for (key,val) in freqDict.items(): 
         if (val > (size/2)): 
             # print(key) 
             return key
    #print('None')
    return -1
    
# Make a prediction with neighbors
def predict_classification(train, num_neighbors, test_row, prediction_type='majority'):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    
    print ("Closet datapoints: ")
    print (neighbors)
    
    print ("Closet datapoints classes: ", end =" ")
    print (output_values)
    
    if prediction_type == 'majority':
        prediction = majority(output_values)
    else:
        prediction = max(set(output_values), key=output_values.count)
    return prediction


# In[ ]:


def prepare_dataset(filename):
    dataset = load_csv(filename)
    #print (dataset)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)

    str_column_to_int(dataset, len(dataset[0])-1)
    return dataset

def run_knn(dataset, num_neighbors, input_data):
    label = predict_classification(dataset, num_neighbors, input_data)
    print('Input Data %s -> Prediction: %s' % (input_data, label))

dataset = prepare_dataset('example_knn.csv')
run_knn(dataset, num_neighbors=3, input_data=[3,4] )

