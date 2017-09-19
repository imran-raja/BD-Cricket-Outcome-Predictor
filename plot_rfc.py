# loading libraries
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# Input data
raw_data = np.loadtxt('bddetails_15.csv', delimiter=',')

# create design matrix X and target vector y
data = raw_data[:, 0:-1]
data = np.array(data)
target = raw_data[:, 15].astype(np.int)
target = np.array(target)

# # standardization
# for i in range(4, 15):
#     data_j = list(data[:,i])
#     for j in range(len(data_j)):
#         min_j = min(data_j)
#         max_j = max(data_j)
#         jj = (data_j[j] - min_j)/(max_j - min_j)
#         data[j, i] = jj

def rfc_func(max_features, color):
	test_scores_rfc = []
	n_est = [i for i in range(1,100)]


	for rs in n_est:
		kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=38)
		rfc = RandomForestClassifier(n_estimators=rs, max_features= max_features, random_state=38)
		scores_rfc = cross_val_score(rfc, data, target, cv=kfold, scoring='accuracy')
		scores_rfc = np.array(scores_rfc)
		test_scores_rfc.append(scores_rfc.mean())

	plt.plot(n_est, test_scores_rfc, '--bo', color = color, label='Max Features: {}'.format(max_features))
	plt.legend(loc='upper right')


rfc_func(3, 'b')	
rfc_func(4, 'r')
rfc_func(5, 'm')
rfc_func(6, 'k')
rfc_func(7, 'c')
rfc_func(8, 'g')

plt.title('Accuracy vs Number of Trees, for Random Forest Classifier', fontsize=16)
plt.xlabel('Number of Trees', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(loc='lower right')
plt.show()

