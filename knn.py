# Loading libraries
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

# Read values from the input csv file
raw_data = np.loadtxt('bddetails_15.csv', delimiter=',')

# Create design matrix X and target vector y
data = raw_data[:, 0:-1]
data = np.array(data)
target = raw_data[:, 15].astype(np.int)
target = np.array(target)

# Standardization
for i in range(0, 15):
    data_j = list(data[:,i])
    for j in range(len(data_j)):
        min_j = min(data_j)
        max_j = max(data_j)
        jj = (data_j[j] - min_j)/(max_j - min_j)
        data[j, i] = jj

neighbors = [i for i in range(1, 30, 2)]
randomstate = [38]

# Empty list that will hold cv scores
test_scores = []
# Perform 5-fold cross validation
for k in neighbors:
	knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance')
	for rs in randomstate:
		kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=rs)
		scores = cross_val_score(knn, data, target, cv=kfold, scoring='accuracy')
		test_scores.append((scores.mean()))

# Changing to misclassification error
MSE = [1 - x for x in test_scores]
k_len = []
for k in neighbors:
	for i in randomstate:
		k_len.append(k)
k_len = np.array(k_len)
rs_len = randomstate * len(neighbors)
rs_len = np.array(rs_len)

# Determining best k
optimal_k = k_len[MSE.index(min(MSE))]
optimal_rs = rs_len[MSE.index(min(MSE))]

# Print results
print("The optimal number of neighbors is %d" % optimal_k)
print("The optimal number of random state is %d" % optimal_rs)
print("The minimum misclassification error is %f" % min(MSE))
print("The maximum accuracy is %f" % (1- min(MSE)))

# Plot accuray vs k
plt.plot(neighbors, test_scores, '--bo')
plt.title('kNN Algorithm', fontsize=16)
plt.xlabel('Number of Neighbors K', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.show()
