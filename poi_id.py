#!/usr/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("./tools/")
import numpy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.preprocessing import MinMaxScaler

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Initially I tried a number of features based on observation while looking for outliers

#features_list = ['poi','salary','exercised_stock_options','bonus','total_stock_value'] 

# After this I tried the kbestfeatures with different parameters, compareing the results from tester.py

# kbestfeatures - 8
#features_list = ['poi','salary','total_payments','exercised_stock_options','bonus','restricted_stock','total_stock_value','deferred_income','long_term_incentive']

#kbestfeatures - 4
#features_list = ['poi','salary','exercised_stock_options','bonus','total_stock_value']

#kbestfeatures - 6
#features_list = ['poi','salary','exercised_stock_options','bonus','total_stock_value','deferred_income','long_term_incentive']

#kbestfeatures - 5
features_list = ['poi','salary','exercised_stock_options','bonus','total_stock_value','deferred_income']



### Task 2: Remove outliers

data_dict.pop("TOTAL", None)
data_dict.pop("BHATNAGAR SANJAY", None)
data_dict.pop("BELFER ROBERT", None)


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
    
features_list.append('fraction_to_poi')
features_list.append('fraction_from_poi')
features_list.append('fraction_shared_receipt')



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
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Decision Tree

#from sklearn.tree import tree
#clf = tree.DecisionTreeClassifier(min_samples_split=2)


### Decision tree with Adaboost.

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_split=2),
#                         algorithm = "SAMME.R",
#                         n_estimators = 200,
#                         learning_rate= 0.5)

#clf = AdaBoostClassifier(n_estimators = 200, learning_rate = 0.5)


### kmeans clusters
#from sklearn.cluster import KMeans
#clf= KMeans(n_clusters=2)


### KNeighborsClassifier
#from sklearn.neighbors import KNeighborsClassifier
#clf= KNeighborsClassifier(n_neighbors=2,p=2,weights="distance")

#from sklearn.ensemble import BaggingClassifier

#clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=2,p=2,weights="distance"),max_samples=0.5,max_features=0.5)


#from sklearn.neighbors import KNeighborsClassifier
#clf= KNeighborsClassifier(n_neighbors=2,p=2,weights="distance")


from sklearn.svm import SVC
clf = SVC(kernel="rbf",C=11000.0)
#clf = SVC(kernel="linear",C=1.0)




# AdaBoost with svm
# Adaboost should use weak classifiers so is not normally used with svm

#clf = AdaBoostClassifier(SVC(kernel="linear",C=10000.0),
#                         algorithm = "SAMME",
#                         n_estimators = 400,
#                         learning_rate = 0.25)



#from sklearn import linear_model, decomposition, datasets
#from sklearn.pipeline import Pipeline
#from sklearn import preprocessing


#logistic = linear_model.LogisticRegression()
#svc = SVC(kernel="rbf",C=10000.0)
#pca = decomposition.PCA()
#clf = Pipeline(steps=[('pac',pca),('logistic', svc)])






#from sklearn.ensemble import GradientBoostingClassifier

#clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0, max_depth=1, random_state=0)








#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=60,n_jobs=-1,oob_score=False)


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
