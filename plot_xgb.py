# loading libraries
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import model_selection
from xgboost import XGBClassifier

# Input data
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

test_scores_xgb = []

def rfc_func(max_depth, color):
	test_scores_xgb = []
	n_estimators = [i for i in range(50,401)]

	for rs in n_estimators:
		kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=38)
		xgb = XGBClassifier(n_estimators=rs, max_depth= max_depth, learning_rate = 0.1)
		scores_xgb = cross_val_score(xgb, data, target, cv=kfold, scoring='accuracy')
		test_scores_xgb.append(scores_xgb.mean())

	plt.plot(n_estimators, test_scores_xgb, '--bo', color = color, label='Max Depth: {}'.format(max_depth))
	plt.legend(loc='lower right')

rfc_func(2, 'g')
rfc_func(3, 'b')
rfc_func(4, 'r')
rfc_func(5, 'm')
rfc_func(6, 'k')
rfc_func(7, 'c')


# plt.title('Accuracy vs Number of Trees, for XGB', fontsize=16)
plt.xlabel('Number of Trees', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(loc='lower right')
plt.show()

