#!/usr/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("./tools/")
import numpy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectFpr
from sklearn.preprocessing import MinMaxScaler

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Initially I tried a number of features based on observation while looking for outliers

#features_list = ['poi','salary','exercised_stock_options','bonus','total_stock_value'] 

# After this I tried the kbestfeatures with different parameters, comparing the results from tester.py

features_list = ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person']

# KBestFeatures and the PCA object are both in the pipeline in the 'optimize_clf' function


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




### Task 3: Create new feature(s)
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
    data_dict[name]['fraction_shared_receipt'] = fraction_to_poi

    
# the features were tested but removed from the final classifier
    
#features_list.append('fraction_to_poi')
#features_list.append('fraction_from_poi')
#features_list.append('fraction_shared_receipt')



### Feature scaling
minmax_features = {}

for key in data_dict:
    for feature in features_list:
        feature_value = data_dict[key][feature]
        if feature_value != 'NaN':
            if not minmax_features.has_key(feature):
                minmax_features[feature] = \
                    {"min":feature_value,"max":feature_value}
            elif feature_value > minmax_features[feature]["max"]:
                minmax_features[feature]["max"]=feature_value
            elif feature_value < minmax_features[feature]["min"]:
                minmax_features[feature]["min"]=feature_value

for key in data_dict:
    for feature in features_list:
        feature_value = data_dict[key][feature]
        if feature != "poi" and feature_value != 'NaN':
            feature_value = data_dict[key][feature]
            minmax = minmax_features[feature]
            mrange = minmax['max']-minmax['min']
            feature_value = feature_value - minmax['min']
            feature_value = float(feature_value) / float(mrange)
            data_dict[key][feature] = feature_value



            

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
# No longer needed as in optimize_clf function
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


def optimize_clf (clf, dataset, feature_list,params):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    scores_arr = []
    pca = PCA()
    selection = SelectKBest(k = 1)
    combined_features = FeatureUnion([("pca",pca),("univ_select",selection)])
    X_features = combined_features.fit(features,labels).transform(features)
    pipeline = Pipeline([("features", combined_features),("clf",clf)])
    pca_range = range(1,len(feature_list))
    params['features__pca__n_components']=pca_range
    k_range = range(1,len(feature_list))
    k_range.append('all')
    params['features__univ_select__k']=k_range
    grid_search = GridSearchCV(pipeline, param_grid= params, scoring = "f1")
    grid_search.fit(features,labels)
    return grid_search.best_estimator_

### KNeighborsClassifier
# Provided as evidence of trying more than one classifier.

# For the complete testing code look at choose_classifier.py

#from sklearn.neighbors import KNeighborsClassifier
#clf= KNeighborsClassifier()
#params = {'clf__n_neighbors':(2,4,6,8),'clf__p':(1,2,4),'clf__weights':('uniform','distance')}
#clfname = "KNeighbors"
#clf = optimize_clf(clf,my_dataset,features_list,params,clfname)

### SVC 
from sklearn.svm import SVC
clf = SVC()
params = {'clf__kernel':('rbf',),'clf__C':(10000,)}
clf = optimize_clf(clf,my_dataset,features_list,params,)




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

#clf.fit(features_train, labels_train)

#print clf.score(features_test, labels_test)
test_classifier(clf,my_dataset, features_list, folds = 200)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
