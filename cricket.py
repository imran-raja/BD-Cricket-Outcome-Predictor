'''
@author: Imran Raja Singh Bhuiyan
@maintainer: Imran Raja Singh Bhuiyan
@date: 19.07.2017
'''

# loading libraries
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Input Data
raw_data = np.loadtxt('bddetails_15.csv', delimiter=',')

# Create design matrix X and target vector y
data = raw_data[:, 0:-1]
data = np.array(data)
target = raw_data[:, 15].astype(np.int)
target = np.array(target)

# # Standardization
# for i in range(4, 15):
#     data_j = list(data[:,i])
#     for j in range(len(data_j)):
#         min_j = min(data_j)
#         max_j = max(data_j)
#         jj = (data_j[j] - min_j)/(max_j - min_j)
#         data[j, i] = jj


# Taking a random seed at 38
randomstate = [i for i in range(37, 40)]

test_scores_logit, test_scores_svc, test_scores_rfc, test_scores_xgb = [], [], [], []

# Perform 5-fold cross validation
for rs in randomstate:
	kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=rs)
	
	logit = LogisticRegression(C = 0.001, class_weight = 'balanced',solver = 'liblinear', max_iter = 100000)
	svc = svm.SVC(kernel = 'linear', C = 0.01)
	rfc = RandomForestClassifier(n_estimators=13, max_features=6, random_state=38)
	xgb = XGBClassifier(n_estimators=250, max_depth = 2, learning_rate = 0.1)

	scores_logit = cross_val_score(logit, data, target, cv=kfold, scoring='accuracy')
	scores_svc = cross_val_score(svc, data, target, cv=kfold, scoring='accuracy')
	scores_rfc = cross_val_score(rfc, data, target, cv=kfold, scoring='accuracy')
	scores_xgb = cross_val_score(xgb, data, target, cv=kfold, scoring='accuracy')

	test_scores_logit.append((scores_logit.mean()))
	test_scores_svc.append((scores_svc.mean()))
	test_scores_rfc.append((scores_rfc.mean()))
	test_scores_xgb.append((scores_xgb.mean()))


plt.plot(randomstate, test_scores_logit, 'b', label='logit')
plt.plot(randomstate, test_scores_svc, 'r', label='svc')
plt.plot(randomstate, test_scores_rfc, 'm', label='Random Forest')
plt.plot(randomstate, test_scores_xgb, 'c', label='XG Boost')

plt.legend(loc='upper right')
plt.show()
