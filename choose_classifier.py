#!/usr/python

import sys
import pickle
sys.path.append("./tools/")
import numpy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
#from tester import test_classifier
#from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
#from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

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


def get_params (clf, dataset, feature_list,params):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, 50, random_state = 42)
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
    gridcv = GridSearchCV(clf,params,scoring = "f1",refit = True).fit(features_train,labels_train)
    return (gridcv)



def optimize_clf (clf, dataset, feature_list,params):
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    scores_arr = []
    for i in range (1,len(features_list)):
        data_new = SelectKBest(f_classif,k=i).fit(features, labels)
#        data_new = SelectKBest(chi2,k=i).fit(features, labels)
        results = data_new.get_support()
        scores = data_new.scores_
        new_features_list = ['poi']
        for j, result in enumerate(results):
            if result == True:
                new_features_list.append(features_list[j+1])
        newclf = get_params(clf,dataset,new_features_list,params)
        newparams = newclf.best_params_
        #print "K" + str(i)
        #print newparams
        scores = test_clf(newclf,dataset,new_features_list)
        this_score_dict ={'k':i,
                          'f1':scores['f1'],
                          'precision':scores['precision'],
                          'recall':scores['recall'],
                          'params':newparams}
        scores_arr.append(this_score_dict)
    topf1 = 0
    topf1k = 0
    for i in range (0,len(features_list)-4):
        if scores_arr[i]['f1'] > topf1:
            topf1 = scores_arr[i]['f1']
            topf1k = i
    return {'top':scores_arr[topf1k],'scores':scores_arr} 




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
    data_dict[name]['fraction_shared_receipt'] = fraction_shared_receipt

#features_list.append('fraction_from_poi')
#features_list.append('fraction_to_poi')
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
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


compare_classifiers = {}
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
params = { }
bestclf = optimize_clf(clf,my_dataset,features_list,params)

compare_classifiers['GaussianNB'] = bestclf


### Decision Tree

# from sklearn.tree import tree


# clf = tree.DecisionTreeClassifier()


# params = {'min_samples_split':(1,2,3,4,5,6)}
# bestclf = optimize_clf(clf,my_dataset,features_list,params)
# compare_classifiers['DecisionTree'] = bestclf

# ### Decision tree with Adaboost.

# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_split=2),
#                          algorithm = "SAMME.R",
#                          n_estimators = 200,
#                          learning_rate= 0.5)
# compare_classifiers['Adaboost w/ optimised DecisionTree'] = test_clf(clf,my_dataset, features_list)

# clf = AdaBoostClassifier(n_estimators = 200, learning_rate = 0.5)
# compare_classifiers['Adaboost w/ default tree'] = test_clf(clf,my_dataset, features_list)


# ### KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# clf= KNeighborsClassifier(n_neighbors=2,p=2,weights="distance")
# compare_classifiers['KNeighbors'] = test_clf(clf,my_dataset, features_list)

# ### Bagging Classifier
# from sklearn.ensemble import BaggingClassifier
# clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=2,p=2,weights="distance"),max_samples=0.5,max_features=0.5)
# compare_classifiers['Bagging Classifier'] = test_clf(clf,my_dataset, features_list)


# ### SVC 
from sklearn.svm import SVC
clf = SVC(kernel="rbf",C=11000.0)
#params = {'kernel':('rbf','linear'),'C':(0.1,1,10,100,1000,10000)}
params = {'kernel':('rbf',),'C':(10000,)}
bestclf = optimize_clf(clf,my_dataset,features_list,params)
compare_classifiers['SVC'] = bestclf



# compare_classifiers['SVC with RBF kernel'] = test_clf(clf,my_dataset, features_list)

# ### SVC linear
# clf = SVC(kernel="linear",C=1.0)
# compare_classifiers['SVC with linear kernel'] = test_clf(clf,my_dataset, features_list)




# # AdaBoost with svm
# # Adaboost should use weak classifiers so is not normally used with svm

# clf = AdaBoostClassifier(SVC(kernel="linear",C=10000.0),
#                          algorithm = "SAMME",
#                          n_estimators = 400,
#                          learning_rate = 0.25)

# compare_classifiers['Adaboost with svm'] = test_clf(clf,my_dataset, features_list)


# from sklearn import linear_model, decomposition, datasets
# from sklearn.pipeline import Pipeline
# from sklearn import preprocessing

# logistic = linear_model.LogisticRegression()
# svc = SVC(kernel="rbf",C=10000.0)
# pca = decomposition.PCA()
# clf = Pipeline(steps=[('pca',pca),('logistic', svc)])


# compare_classifiers['svm with pca'] = test_clf(clf,my_dataset, features_list)




# from sklearn.ensemble import GradientBoostingClassifier

# clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0, max_depth=1, random_state=0)
# compare_classifiers['gradient boosting classifier'] = test_clf(clf,my_dataset, features_list)




line_one = "|classifier|"    
line_two = "|precision|"    
line_three = "|recall|"    
line_four = "|score|"    

bestscore = 0.0
bestscorekey = ""
for key in compare_classifiers:
    score = compare_classifiers[key]['top']
    line_one = line_one + key + "|"
    line_two = line_two + str(round(score['precision'],3)) + "|"
    line_three = line_three + str(round(score['recall'],3)) + "|"
    line_four = line_four + str(round(score['f1'],3)) + "|"
    if score['f1'] > bestscore:
        bestscore = score['f1']
        bestscorekey = key
    
print line_one
print line_two
print line_three
print line_four


#for key in compare_classifiers:





print ""
print ""
print "Optimal Classifier: " + bestscorekey
print ""
print bestscorekey + " parameters: "
print compare_classifiers[bestscorekey]['top']['params']

print "number of K features: " + str(compare_classifiers[bestscorekey]['top']['k'])






