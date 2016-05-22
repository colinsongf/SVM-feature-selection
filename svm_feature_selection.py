#   author: Giangiacomo Mercatali
#   email:  giangiacomo.mercatali@postgrad.manchester.ac.uk

from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# Import data
import numpy as np
from stav_datasets import spectr
X,y=np.array(spectr['data']), np.array(spectr['target'])
means=[]
stds=[]
num_features=[]

# SVM linear classifier
lsvm = SVC(kernel='linear')
logreg = linear_model.LogisticRegression(penalty='l1')

# Meta-transformer for selecting features based on importance of SVM weights
sfm = SelectFromModel(estimator = lsvm)

# Pipeline including feature selection and classification
p = Pipeline([ ('fs', sfm), ('svc', logreg) ])

# K-Folds cross validation iterator on k = 30 (CV Leave One Out)
kfcv = cross_validation.KFold(30, 30)

# select subset of features on varying threshold and test the accuracy on CV LOO
for i in np.arange(0,1,0.1):
    p.set_params(fs__threshold = i)
    scores = cross_validation.cross_val_score(p,X,y,cv=kfcv, scoring='accuracy')
    means.append(scores.mean())
    stds.append(scores.std())
    num_features.append(sfm.fit_transform(X,y).shape[1])

# Plot
import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()
ind=0

for i in np.arange(0,1,0.1):
    str1=str(num_features[ind])
    plt.annotate(str1, xy=(i, means[ind]), xytext=(i+0.001, means[ind]+0.02 ) )
    ind = ind + 1

plt.plot(np.arange(0,1,0.1),means)
plt.yticks(np.arange(0.2, 0.8, 0.05))
plt.title('SVM - threshold feature selection')
plt.ylabel('LOO CV Accuracy')
plt.xlabel('Threshold')
plt.show()
plt.savefig('svm feature selection.png', format='png')