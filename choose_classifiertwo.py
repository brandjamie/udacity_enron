#!/usr/python

import sys
import pickle
sys.path.append("./tools/")
import numpy
from feature_format import featureFormat, targetFeatureSplit
#from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectFpr
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# start with all the features except for 'email address'
features_list = ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person']

# same as the first feature list - will not have features added
features_listtwo = ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person']


# My hand selected list of features for comparison
features_listthree = ['poi','exercised_stock_options', 'total_stock_value', 'bonus','deferred_income']

# Same as above but will not have features added. 
features_listfour = ['poi','exercised_stock_options', 'total_stock_value', 'bonus','deferred_income']

# used to iterate through feature sets
features_list_list = ['features_list','features_listtwo','features_listthree','features_listfour']



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


## Modified From sklearn documentation - feature stacker
## uses pca, kselect best, combined_features, pipeline and gridsearchcv to find the best parameters
def optimize_clf (clf, dataset, feature_list,params):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    scores_arr = []
    pca = PCA()
    selection = SelectKBest(k = 1)
    combined_features = FeatureUnion([("pca",pca),("univ_select",selection)])
    X_features = combined_features.fit(features,labels).transform(features)
    pipeline = Pipeline([("features", combined_features),("clf",clf)])
    ## try with a few components at first to see if it works
    pca_range = range(1,len(feature_list))
    params['features__pca__n_components']=pca_range
    k_range = range(1,len(feature_list))
    k_range.append('all')
    params['features__univ_select__k']=k_range
    grid_search = GridSearchCV(pipeline, param_grid= params, scoring = "f1")
    grid_search.fit(features,labels)
    scores = test_clf(grid_search.best_estimator_,dataset,feature_list)
    scores['params'] = grid_search.best_params_
    return scores 


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

features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')
features_list.append('fraction_shared_receipt')
features_listthree.append('fraction_from_poi')
features_listthree.append('fraction_to_poi')
features_listthree.append('fraction_shared_receipt')


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

          

my_dataset = data_dict


# dictionarys (one for each feature set) to compare algorithms
compare_classifiers = {}
compare_classifierstwo = {}
compare_classifiersthree = {}
compare_classifiersfour = {}

compare_classifiers_dict = {
    'features_list':compare_classifiers,
    'features_listtwo':compare_classifierstwo,
    'features_listthree':compare_classifiersthree,
    'features_listfour':compare_classifiersfour}


### algorithms to try:
### each algorithm has a parameters array to be sent to grid search.


### Gaussian NB
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
params = { }
bestclf = optimize_clf(clf,my_dataset,features_list,params)
compare_classifiers['GaussianNB'] = bestclf
bestclf = optimize_clf(clf,my_dataset,features_listtwo,params)
compare_classifierstwo['GaussianNB'] = bestclf

bestclf = optimize_clf(clf,my_dataset,features_listthree,params)
compare_classifiersthree['GaussianNB'] = bestclf

bestclf = optimize_clf(clf,my_dataset,features_listfour,params)
compare_classifiersfour['GaussianNB'] = bestclf

# ### Decision Tree

# from sklearn.tree import tree
# clf = tree.DecisionTreeClassifier()
# params = {'clf__min_samples_split':(1,2,3,4,5,6)}

# bestclf = optimize_clf(clf,my_dataset,features_list,params)
# compare_classifiers['DecisionTree'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listtwo,params)
# compare_classifierstwo['DecisionTree'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listthree,params)
# compare_classifiersthree['DecisionTree'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listfour,params)
# compare_classifiersfour['DecisionTree'] = bestclf


# ### Decision tree with Adaboost.   

# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(algorithm = "SAMME", n_estimators = 200, learning_rate = 1)

# params = {'clf__n_estimators':(50,100,200),'clf__learning_rate':(0.5,1,2)}
# bestclf = optimize_clf(clf,my_dataset,features_list,params)
# compare_classifiers['Adaboost'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listtwo,params)
# compare_classifierstwo['Adaboost'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listthree,params)
# compare_classifiersthree['Adaboost'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listfour,params)
# compare_classifiersfour['Adaboost'] = bestclf



# ### KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# clf= KNeighborsClassifier()

# params = {'clf__n_neighbors':(2,4,6,8),'clf__p':(1,2,4),'clf__weights':('uniform','distance')}
# bestclf = optimize_clf(clf,my_dataset,features_list,params)
# compare_classifiers['KNeighbours'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listtwo,params)
# compare_classifierstwo['KNeighbours'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listthree,params)
# compare_classifiersthree['KNeighbours'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listfour,params)
# compare_classifiersfour['KNeighbours'] = bestclf



# ### SVC 
from sklearn.svm import SVC
clf = SVC()
params = {'clf__kernel':('rbf','linear'),'clf__C':(0.1,1,10,100,1000,10000)}
# This line can be uncommented for testing (to increase speed). 
params = {'clf__kernel':('rbf',),'clf__C':(10000,)}
bestclf = optimize_clf(clf,my_dataset,features_list,params)
compare_classifiers['SVC'] = bestclf
bestclf = optimize_clf(clf,my_dataset,features_listtwo,params)
compare_classifierstwo['SVC'] = bestclf

bestclf = optimize_clf(clf,my_dataset,features_listthree,params)
compare_classifiersthree['SVC'] = bestclf

bestclf = optimize_clf(clf,my_dataset,features_listfour,params)
compare_classifiersfour['SVC'] = bestclf




# ###  GradientBoostingClassifier
# from sklearn.ensemble import GradientBoostingClassifier

# clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0, max_depth=1, random_state=0)
# params = {'clf__n_estimators':(50,100,200),'clf__learning_rate':(0.5,1,1.5)}
# bestclf = optimize_clf(clf,my_dataset,features_list,params)
# compare_classifiers['Gradient Boosting'] = bestclf

# bestclf = optimize_clf(clf,my_dataset,features_listtwo,params)
# compare_classifierstwo['Gradient Boosting'] = bestclf







### Print out results ##############


for flkey in features_list_list:
    line_one = "|classifier|"
    line_two = "| n_pca |"
    line_three = "| n_k |"
    line_four = "|precision|"    
    line_five = "|recall|"    
    line_six = "|f1 score|"    
    
    bestscore = 0.0
    bestscorekey = ""
    for key in compare_classifiers_dict[flkey]:
        score = compare_classifiers_dict[flkey][key]
        line_one = line_one + key + "|"
        line_two = line_two + str(score['params']['features__pca__n_components']) + "|"
        line_three = line_three + str(score['params']['features__univ_select__k']) + "|"
        line_four = line_four + str(round(score['precision'],3)) + "|"
        line_five = line_five + str(round(score['recall'],3)) + "|"
        line_six = line_six + str(round(score['f1'],3)) + "|"
        if score['f1'] > bestscore:
            bestscore = score['f1']
            bestscorekey = key

    print ("For featureset:" + str(flkey))
    print line_one
    print line_two
    print line_three
    print line_four
    print line_five
    print line_six
    print ""
    print ""
    print "Optimal Classifier: " + bestscorekey
    print ""
    print bestscorekey + " parameters: "
    print compare_classifiers_dict[flkey][bestscorekey]['params']
    print ""
    print ""






# ### features two


# line_one = "|classifier|"
# line_two = "| n_pca |"
# line_three = "| n_k |"
# line_four = "|precision|"    
# line_five = "|recall|"    
# line_six = "|f1 score|"    

# bestscore = 0.0
# bestscorekey = ""
# for key in compare_classifierstwo:
#     score = compare_classifierstwo[key]
#     line_one = line_one + key + "|"
#     line_two = line_two + str(score['params']['features__pca__n_components']) + "|"
#     line_three = line_three + str(score['params']['features__univ_select__k']) + "|"
#     line_four = line_four + str(round(score['precision'],3)) + "|"
#     line_five = line_five + str(round(score['recall'],3)) + "|"
#     line_six = line_six + str(round(score['f1'],3)) + "|"
#     if score['f1'] > bestscore:
#         bestscore = score['f1']
#         bestscorekey = key
# print ""
# print ""
# print ""
# print ("For featureset two (without fractions)")
# print line_one
# print line_two
# print line_three
# print line_four
# print line_five
# print line_six

 
# print ""
# print ""
# print "Optimal Classifier: " + bestscorekey
# print ""


# print bestscorekey + " parameters: "
# print compare_classifierstwo[bestscorekey]['params']
# print ""
# print ""


# ### features set three


# line_one = "|classifier|"
# line_two = "| n_pca |"
# line_three = "| n_k |"
# line_four = "|precision|"    
# line_five = "|recall|"    
# line_six = "|f1 score|"    

# bestscore = 0.0
# bestscorekey = ""
# for key in compare_classifiersthree:
#     score = compare_classifiersthree[key]
#     line_one = line_one + key + "|"
#     line_two = line_two + str(score['params']['features__pca__n_components']) + "|"
#     line_three = line_three + str(score['params']['features__univ_select__k']) + "|"
#     line_four = line_four + str(round(score['precision'],3)) + "|"
#     line_five = line_five + str(round(score['recall'],3)) + "|"
#     line_six = line_six + str(round(score['f1'],3)) + "|"
#     if score['f1'] > bestscore:
#         bestscore = score['f1']
#         bestscorekey = key

# print ("For feature set three (with fractions)")
# print line_one
# print line_two
# print line_three
# print line_four
# print line_five
# print line_six
# print ""
# print ""
# print "Optimal Classifier: " + bestscorekey
# print ""


# print bestscorekey + " parameters: "
# print compare_classifiersthree[bestscorekey]['params']




# ### for features set four

# line_one = "|classifier|"
# line_two = "| n_pca |"
# line_three = "| n_k |"
# line_four = "|precision|"    
# line_five = "|recall|"    
# line_six = "|f1 score|"    

# bestscore = 0.0
# bestscorekey = ""
# for key in compare_classifiersfour:
#     score = compare_classifiersfour[key]
#     line_one = line_one + key + "|"
#     line_two = line_two + str(score['params']['features__pca__n_components']) + "|"
#     line_three = line_three + str(score['params']['features__univ_select__k']) + "|"
#     line_four = line_four + str(round(score['precision'],3)) + "|"
#     line_five = line_five + str(round(score['recall'],3)) + "|"
#     line_six = line_six + str(round(score['f1'],3)) + "|"
#     if score['f1'] > bestscore:
#         bestscore = score['f1']
#         bestscorekey = key

# print ("For feature set four (with out fractions")
# print line_one
# print line_two
# print line_three
# print line_four
# print line_five
# print line_six
# print ""
# print ""
# print "Optimal Classifier: " + bestscorekey
# print ""


# print bestscorekey + " parameters: "
# print compare_classifiersfour[bestscorekey]['params']









#### Get info about PCA and K select features

print "Stats for PCA and K select features:"
print ""
print ""

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
datatwo = featureFormat(my_dataset, features_listtwo, sort_keys = True)
labelstwo, featurestwo = targetFeatureSplit(datatwo)

datathree = featureFormat(my_dataset, features_listthree, sort_keys = True)
labelsthree, featuresthree = targetFeatureSplit(datathree)

datafour = featureFormat(my_dataset, features_listfour, sort_keys = True)
labelsfour, featuresfour = targetFeatureSplit(datafour)



n_comp = len(features_list)-1
n_comptwo = len(features_listtwo)-1
n_compthree = len(features_listthree)-1
n_compfour = len(features_listfour)-1


pca = PCA(n_components = n_comp)
pcatwo = PCA(n_components = n_comptwo)
pcathree = PCA(n_components = n_compthree)
pcafour = PCA(n_components = n_compfour)

pca.fit(features,labels)
ratios = pca.explained_variance_ratio_

print "PCA Explained Variance Ratio for feature set one"
print ""
ratio_string = "|"
for r in ratios:
    ratio_string = ratio_string + str(round(r,5)) + "|"
print ratio_string
print ""

pcatwo.fit(featurestwo,labelstwo)
ratiostwo = pcatwo.explained_variance_ratio_

print "PCA Explained Variance Ratio for feature set two"
print ""
ratio_string = "|"
for r in ratiostwo:
    ratio_string = ratio_string + str(round(r,5)) + "|"
print ratio_string
print ""

pcathree.fit(featuresthree,labelsthree)
ratiosthree = pcathree.explained_variance_ratio_

print "PCA Explained Variance Ratio for feature set three"
print ""
ratio_string = "|"
for r in ratiosthree:
    ratio_string = ratio_string + str(round(r,5)) + "|"
print ratio_string
print ""


pcafour.fit(featuresfour,labelsfour)
ratiosfour = pcafour.explained_variance_ratio_

print "PCA Explained Variance Ratio for feature set four"
print ""
ratio_string = "|"
for r in ratiosfour:
    ratio_string = ratio_string + str(round(r,5)) + "|"
print ratio_string
print ""

#### kbest for features one

print ""
print ""
print "K Best scores for feature set one"
print ""


kbest = SelectKBest(f_classif,k='all').fit(features, labels)
results = kbest.get_support()
scores = kbest.scores_
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



#### k best for features two

print ""
print ""
print "K Best scores for feature set two"
print ""

kbest = SelectKBest(f_classif,k='all').fit(featurestwo, labelstwo)
results = kbest.get_support()
scores = kbest.scores_
result_dict = {}
for i, result in enumerate(results):
    f = features_listtwo[i+1] #i+1 as we have removed the poi feature
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






#### k best for features three

print ""
print ""
print "K Best scores for feature set three"
print ""

kbest = SelectKBest(f_classif,k='all').fit(featuresthree, labelsthree)
results = kbest.get_support()
scores = kbest.scores_
result_dict = {}
for i, result in enumerate(results):
    f = features_listthree[i+1] #i+1 as we have removed the poi feature
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





#### k best for features four

print ""
print ""
print "K Best scores for feature set four"
print ""

kbest = SelectKBest(f_classif,k='all').fit(featuresfour, labelsfour)
results = kbest.get_support()
scores = kbest.scores_
result_dict = {}
for i, result in enumerate(results):
    f = features_listfour[i+1] #i+1 as we have removed the poi feature
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





