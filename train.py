import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from itertools import combinations 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import tree
import pickle

print('In Train Module')

print('Loading Train and Test data')
Xtr = np.load('TrainX.npy')
Ytr = np.load('TrainY.npy')
Xtest = np.load('TestX.npy')
Ytest = np.load('TestY.npy')


#Splitting Train data into Train and Dev data
print('Splitting Train data into Train and Dev data in ration 80:20')
Xtrain = Xtr[:int(0.8*len(Xtr))]
Ytrain = Ytr[:int(0.8*len(Ytr))]
Xdev = Xtr[int(0.8*len(Xtr)):]
Ydev = Ytr[int(0.8*len(Ytr)):]

#Most frequent classifier
def most_freq_classifier(Xtr,Ytr,Xtest,Ytest):
    most_freq = stats.mode(Ytr)[0][0]
    if (most_freq ==0.0):
        prediction = np.zeros(len(Ytest))
    elif(most_freq==1.0):
        prediction = np.ones(len(Ytest))
    return prediction


#Baseline classifier
print('Using Most Frequent Classifier as Baseline Classifier')
print('Predicting using Most Frequent Classifier')
prediction = most_freq_classifier(Xtr,Ytr,Xtest,Ytest)
accuracy = np.mean(prediction==Ytest)
print('Accuracy of Most Frequent Classifier  on Test Data : ', accuracy)


#Feature selector method
#Finds features to be deleted to give best accuracy
#Parameter n : Number of features to be deleted. Features are deleted one at a time
def feature_selector(X,Y,classifier,n=40):
    final_list = []
    #max_accuracy = 0
    Xtrain = X[:int(0.8*len(X))]
    Ytrain = Y[:int(0.8*len(Y))]
    Xdev = X[int(0.8*len(X)):]
    Ydev = Y[int(0.8*len(Y)):]
    classifier.fit(Xtrain,Ytrain)
    max_accuracy = classifier.score(Xdev,Ydev)
    print('Default Score = ', max_accuracy)
    features = []
    a=0
    for number in range(n):
        #print('number =',number)
        #accuracies = np.zeros(len(X[0]))
        accuracy = 0
        for i in range(len(Xtrain[0])):
            if i in features:
                #print(i, ' already in features')
                continue
            dummy = features.copy()
            dummy.append(i)
            ##print('removing', dummy)
            new_xtrain = np.delete(Xtrain,dummy, axis=1)
            new_xdev = np.delete(Xdev,dummy, axis=1)
            #print('features =',len(new_xtrain[0]))
            #new_ytrain = np.delete(Ytrain,i)
            classifier.fit(new_xtrain,Ytrain)
            accuracy = (classifier.score(new_xdev,Ydev))
            ##print(accuracy)
            if(accuracy>max_accuracy):
                #print('better accueacy found', accuracy,' > ',max_accuracy)
                max_accuracy = accuracy
                final_list = dummy
                #print('new list of features to delete', final_list)
        features = final_list
        if(len(features) != (number+1)):
            print('No additional feature removal performs better')
            print('Features to be deleted: ',features, 'Best accuracy on dev after deleting features: ', max_accuracy)
            #print('breaking')
            break
        
    return features, max_accuracy


#Dtree
print('Predicting using Decision Tree')
dtree = tree.DecisionTreeClassifier()
dtree.fit(Xtrain,Ytrain)
score = dtree.score(Xdev,Ydev)
print('DTree score : ',score)
#Knn
print('Predicting using KNN')
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(Xtrain,Ytrain)
score = knn.score(Xdev,Ydev)
print('KNN score : ',score)
#SVC
print('Predicting using Kernel SVM')
svc = svc_classifier = SVC()
svc.fit(Xtrain,Ytrain)
score = svc.score(Xdev,Ydev)
print('SVC Score : ', score)


##
print('kernel SVM performs better before tuning')
##


#Decision Tree Tuning
print('Tuning Decision Tree')
best_depth = 0
best_score = 0 
for d in range(1,100):
    dtree = tree.DecisionTreeClassifier(max_depth = d)
    dtree.fit(Xtrain,Ytrain)
    score = dtree.score(Xdev,Ydev)
    #print(score)
    if (score>best_score):
        best_score = score
        best_depth = d


print('Best max_depth : ',best_depth)
print('Accuracy on Dev data using best max_depth : ',best_score)


#model
print('Creating DTree model using best depth')
dtree = tree.DecisionTreeClassifier(max_depth=best_depth)

#Feature Selection
print('Using feature selector')
del_list, best_acc = feature_selector(Xtr,Ytr,dtree,n=40)



####
#print('Transforming the Train and Test data by deleting the features')
#newXtrain = np.delete(Xtr,del_list, axis=1)
#newXtest = np.delete(Xtest,del_list,axis=1)
#dtree.fit(newXtrain, Ytr)
#test_accuracy = dtree.score(newXtest,Ytest)
#print('Test Accuracy = ',test_accuracy)
####


#KNN Tuning
print('Tuning KNN')
best_score = 0
best_k = 0
for k in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain,Ytrain)
    score = (knn.score(Xdev,Ydev))
    if(score>best_score):
        best_score = score
        best_k = k

#print(accuracies)
#optimum_k = np.argmax(accuracies)+1
print('Optimum k = ',best_k)
print('Accuracy on Dev data using Optimum k= ',best_score)

#Model
print('Creating KNN model using optimum k')
knn  = KNeighborsClassifier(n_neighbors=best_k)

#Feature Selection
print('Using feature selector')
del_list, best_acc = feature_selector(Xtr,Ytr,knn,40)

###
#print('Transforming the Train and Test data by deleting the features')
#newXtrain = np.delete(Xtr,del_list, axis=1)
#newXtest = np.delete(Xtest,del_list,axis=1)
#knn.fit(newXtrain,Ytr)
#test_accuracy = knn.score(newXtest,Ytest)
#print('Test Accuracy = ',test_accuracy)
###

#SVC Tuning
print('Tuning SVC')
print('Finding best gamma')
best_gamma = 0
best_score = 0
arr = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
for g in arr:
    svc_classifier = SVC(C=3.0,gamma=g)
    svc_classifier.fit(Xtr, Ytr)
    score =(svc_classifier.score(Xtest,Ytest))
    if(score>best_score):
        best_score = score
        best_gamma = g

#print('best score = ', best_score)
max_score = 0
best_c = 0
for c in range(1,101):
    svc_classifier = SVC(C=c, gamma=best_gamma)
    svc_classifier.fit(Xtrain, Ytrain)
    score = svc_classifier.score(Xdev,Ydev)
    if(score > max_score):
        #print('better score found', score, 'C=',c)
        max_score = score
        best_c = c
print('best gamma=', best_gamma)
print('best C = ',best_c)
print('Accuracy on Dev data using best gamma and best c = ', max_score)


#Model
svc_classifier = SVC(C=best_c, gamma=best_gamma)
print('Using Feature Selection')
del_list, best_acc = feature_selector(Xtr,Ytr,svc_classifier,n=40)
###
##Transforming data
#print('Transforming Train and Test data')
#newXtrain = np.delete(Xtr,del_list, axis=1)
#newXtest = np.delete(Xtest,del_list,axis=1)
#svc_classifier.fit(newXtrain, Ytr)
#test_accuracy = svc_classifier.score(newXtest,Ytest)
#print('Test Accuracy = ',test_accuracy)


print('SVC performs best on dev data even after tuning and feature selection')


print('Dumping featuers to be deleted to a file')
new_list = np.array(del_list)
np.save('del-features.npy',del_list)

print('Creating final model')
#Transforming data
print('Transforming Train and Test data')
newXtrain = np.delete(Xtr,del_list, axis=1)
newXtest = np.delete(Xtest,del_list,axis=1)
svc_classifier.fit(newXtrain, Ytr)

print('Saving Classifier')
pickle.dump(svc_classifier,open('final_classifier','wb'))
