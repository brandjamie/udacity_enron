#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("./tools/")
import numpy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# start with all the features except for 'email address'
features_list = ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person']

### Task 2: Remove outliers

data_dict.pop("TOTAL", None)
data_dict.pop("BHATNAGAR SANJAY", None)
data_dict.pop("BELFER ROBERT", None)

# Create new feature (as in task 3)
for name in data_dict:
    to_messages = data_dict[name]['to_messages']
    from_messages = data_dict[name]['from_messages']
    to_user_from_poi = data_dict[name]['from_poi_to_this_person']
    from_user_to_poi = data_dict[name]['from_this_person_to_poi']
    if to_messages != 'NaN' and to_user_from_poi != 'NaN':
        fraction_from_poi = int(to_user_from_poi)/int(to_messages)
    else:
        fraction_from_poi = 'NaN'
    if from_messages != 'NaN' and from_user_to_poi != 'NaN':
        fraction_to_poi = int(from_user_to_poi)/int(from_messages)
    else:
        fraction_to_poi = 'NaN'
    
    data_dict[name]['fraction_from_poi'] = fraction_from_poi
    data_dict[name]['fraction_to_poi'] = fraction_to_poi
features_list.append('fraction_to_poi')
features_list.append('fraction_from_poi')

print features_list

### Extract features and labels from dataset for local testing
my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print data.shape
data_new = SelectKBest(f_classif,k=5).fit(features, labels)

results = data_new.get_support()
#print data_new.shape
scores = data_new.scores_
for i, result in enumerate(results):
    if result == True:
        print features_list[i+1] #i+1 as we have removed the poi feature
        print scores[i]
        print result

# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)


