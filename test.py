import pickle
import numpy as np
from scipy import stats

Xtr = np.load('TrainX.npy')
Ytr = np.load('TrainY.npy')
Xtest = np.load('TestX.npy')
Ytest = np.load('TestY.npy')

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


print('Loading features to be deleted')
del_list = np.load('del-features.npy')
del_list = list(del_list)

print('Loading Classifier')
clf = pickle.load(open('final_classifier','rb'))

print('Transforming Test Data')
newXtest = np.delete(Xtest,del_list,axis=1)

print('Predicting using Best Classifier')
test_accuracy = clf.score(newXtest,Ytest)
print('Test Accuracy = ',test_accuracy)