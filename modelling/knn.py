# KNN Mode1 Proposal

# John Holt aka holtjohn
# 05.03.22
############################################################################
#   - This program runs 10,000 rounds of random seeds through a KNN model  #
#   - Determines range of accuracy for a given k choice                    #
#   - Determines average outcomes for a given k choice                     #
############################################################################

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#######################################
#               Presets               #
#######################################

# File Name With Data
file = 'Final Features Sorted.csv'     
# Best k value result
k = 7

# Second best k value result
#k = 9

# Third best k value result
#k = 5

# Activated Variables
#   - Way to list the variables for the KNN model, deactivate the rest
#   - Not used in final knn model selection
#act_var = []

#######################################

# Import Data
df = pd.read_csv(file,index_col = "File" )
all_data = df.to_numpy()
data = all_data[:,:-1]
truth = all_data[:,-1]
all_features = list(df)

# Normalize
min_vals = np.min(data,axis=0)
boosted_vals = data - min_vals
max_vals = np.max(boosted_vals,axis=0)
normalized = boosted_vals/max_vals

#Split data into proportionally representative random training and testing sets
#   - 5/6 data for training
#   - 1/6 data for testing
blurs = int(np.sum(truth))
blur_train_count = int(5/6*blurs)
clear_train_count = int(5/6*(918-blurs))
var = len(data[0])
test_list = []
train_list = []
test_f_blur = []
test_f_clear = []

# Run 10,000 rounds of KNN model to determine ranges of accuracy
for i in range(10000):
    # Establish indices for random selections of training and testing sets
    blur_train = []
    train = []
    test = []
    while len(blur_train) < blur_train_count:
        index = np.random.randint(blurs)
        if index not in blur_train:
            blur_train.append(index)      
    while len(train) < clear_train_count:
        index = np.random.randint(blurs,918)
        if index not in train:
            train.append(index)
    for entry in blur_train:
        train.append(entry)
    for i in range(918):
        if i not in blur_train and i not in train:
            test.append(i)       
    train_data = normalized[train]
    train_truth = truth[train]
    test_data = normalized[test]
    test_truth = truth[test]
    valist = np.ones(var)
    
    # Deactivation of undesired variables
    #   - Not used in final model selection
    #for i in range(len(valist)):
    #    if all_features[i] not in act_var:
    #        valist[i] = 0
    
    # Code for isolation of specific variables
    #   - Not used in final model selection
    temp_train_data = train_data[:,np.where(valist == 1)[0]]
    temp_test_data = test_data[:,np.where(valist == 1)[0]]

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(temp_train_data,train_truth)

    train_predictions = neigh.predict(temp_train_data)
    train_accuracy = 1- np.sum(np.abs(train_predictions-train_truth))/len(train_predictions)
    test_predictions = neigh.predict(temp_test_data)
    arr_out = np.append(train_predictions,test_predictions)
    test_accuracy = 1- np.sum(np.abs(test_predictions-test_truth))/len(test_predictions)
    train_list.append(train_accuracy)
    test_list.append(test_accuracy)

    # Code for false_blur and false_clear counts
    test_results = test_predictions-test_truth
    fclear = np.count_nonzero(test_results == -1)
    fblur = np.count_nonzero(test_results == 1)
    test_f_clear.append(fclear)
    test_f_blur.append(fblur)

# Output
print("Train Min/Max: (",round(min(train_list),4),", ",round(max(train_list),4),")",sep="")
print("Test Min/Max: (",round(min(test_list),4),", ",round(max(test_list),4),")",sep="")
print("Test Avg:",round(sum(test_list)/len(test_list),4))
print("Test Avg False Blurry:",round(sum(test_f_blur)/len(test_f_blur),3))
print("Test Avg False Clear:",round(sum(test_f_clear)/len(test_f_clear),3))
plt.hist(test_list,bins = 20)






