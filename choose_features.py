#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("./tools/")
import numpy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.cross_validation import StratifiedShuffleSplit


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# start with all the features except for 'email address'
features_list = ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person']

### Task 2: Remove outliers

data_dict.pop("TOTAL", None)

### Correct data for Bhatnagar Sanjay

b_sandjay = data_dict['BHATNAGAR SANJAY']
b_sandjay['expenses'] = 137864
b_sandjay['total_payments'] = 137864
b_sandjay['exercised_stock_options'] = 15456734
b_sandjay['restricted_stock'] = 2604490
b_sandjay['restricted_stock_deferred'] = 2604490
b_sandjay['total_stock_value'] = 15456290
b_sandjay['director_fees'] = 'NaN'
b_sandjay['other'] = 'NaN'



### Correct data for B Robert
b_robert = data_dict["BELFER ROBERT"]

b_robert['deferred_income'] = 102500
b_robert['deferral_payments']= 'NaN'
b_robert['expenses'] = 3285
b_robert['directors_fees'] = 102500
b_robert['total_payments']=3285
b_robert['exercised_stock_options']='NaN'
b_robert['restricted_stock_options']=44093

### Modified from tester.py
def test_clf(clf, dataset, feature_list, folds = 200):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    if true_positives+false_positives > 0:
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
    else:
        precision = 0
        recall = 0
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    return {'precision':precision,'recall':recall,'f1':f1}    




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

for name in data_dict:
    to_messages = data_dict[name]['to_messages']
    from_messages = data_dict[name]['from_messages']
    to_user_from_poi = data_dict[name]['from_poi_to_this_person']
    from_user_to_poi = data_dict[name]['from_this_person_to_poi']
    shared_receipt = data_dict[name]['shared_receipt_with_poi']

    if to_messages != 'NaN' and to_user_from_poi != 'NaN':
        fraction_from_poi = float(to_user_from_poi)/float(to_messages)
    else:
        fraction_from_poi = 'NaN'
    if from_messages != 'NaN' and from_user_to_poi != 'NaN':
        fraction_to_poi = float(from_user_to_poi)/float(from_messages)
    else:
        fraction_to_poi = 'NaN'
    if to_messages != 'NaN' and shared_receipt != 'NaN':
        fraction_shared_receipt = float(shared_receipt)/float(to_messages)
    else:
        fraction_shared_receipt = 'NaN'
 
        
    data_dict[name]['fraction_from_poi'] = fraction_from_poi

    data_dict[name]['fraction_to_poi'] = fraction_to_poi
    data_dict[name]['fraction_shared_receipt'] = fraction_shared_receipt

features_list.append('fraction_to_poi')
features_list.append('fraction_from_poi')
features_list.append('fraction_shared_receipt')

### Extract features and labels from dataset for local testing
my_dataset = data_dict

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scores_arr = []
for i in range (1,len(features_list)):
    data_new = SelectKBest(f_classif,k=i).fit(features, labels)
    results = data_new.get_support()
    scores = data_new.scores_
    new_features_list = ['poi']
    for j, result in enumerate(results):
        if result == True:
            new_features_list.append(features_list[j+1])
    scores = test_clf(clf,my_dataset,new_features_list)
    this_score_dict ={'k':i,
                      'precision':scores['precision'],
                      'recall':scores['recall'],
                      'total':scores['precision']+scores['recall'],
                      'f1':scores['f1']}
    scores_arr.append(this_score_dict)

    # prints results in format compatible with org-mode table. 
line_one = "|k|"    
line_two = "|precision|"    
line_three = "|recall|"    
line_four = "|f1|"    

for score in scores_arr:
    line_one = line_one + str(score['k']) + "|"
    line_two = line_two + str(round(score['precision'],3)) + "|"
    line_three = line_three + str(round(score['recall'],3)) + "|"
    line_four = line_four + str(round(score['f1'],3)) + "|"
print line_one
print line_two
print line_three
print line_four

data_new = SelectKBest(f_classif,k='all').fit(features, labels)
results = data_new.get_support()
scores = data_new.scores_
result_dict = {}
for i, result in enumerate(results):
    f = features_list[i+1] #i+1 as we have removed the poi feature
    s = scores[i]
    result_dict[f] =  s
import operator
sorted_results = sorted(result_dict.items(), key = operator.itemgetter(1), reverse = True)

print ""
print ""

line_one = "|"
line_two = "|"
for r in sorted_results:
    line_one = line_one + r[0] + "|"
    line_two = line_two + str(round(r[1],3)) + "|"
    
print line_one
print line_two








