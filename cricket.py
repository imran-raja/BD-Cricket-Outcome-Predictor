# loading libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from xgboost import XGBClassifier

raw_data = np.loadtxt('bddetails_15.csv', delimiter=',')

# create design matrix X and target vector y
data = raw_data[:, 0:-1]
data = np.array(data)
target = raw_data[:, 14].astype(np.int)
target = np.array(target)

# print(raw_data[:0].dtype)

# standardization
for i in range(4, 10):
    data_j = list(data[:,i])
    for j in range(len(data_j)):
        min_j = min(data_j)
        max_j = max(data_j)
        jj = (data_j[j] - min_j)/(max_j - min_j)
        data[j, i] = jj

randomstate = [i for i in range(0, 50)]
# randomstate = [31]

cv_scores = []
test_scores_logit = []
test_scores_svc = []
test_scores_linear_svc = []
test_scores_rfc = []
test_scores_gnb = []
test_scores_xgb = []

# perform 5-fold cross validation
for rs in randomstate:
	kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=rs)

	logit = LogisticRegression(C = 0.01, class_weight = 'balanced', solver = 'newton-cg', max_iter = 100000)
	svc = svm.SVC(kernel = 'linear', C = 0.01)
	# linear_svc = svm.LinearSVC(C = 0.1)
	rfc = RandomForestClassifier(n_estimators=20, max_features=3, random_state=33)
	gnb = GaussianNB()
	xgb = XGBClassifier()

	scores_logit = cross_val_score(logit, data, target, cv=kfold, scoring='accuracy')
	scores_svc = cross_val_score(svc, data, target, cv=kfold, scoring='accuracy')
	# scores_linear_svc = cross_val_score(linear_svc, data, target, cv=kfold, scoring='accuracy')
	scores_rfc = cross_val_score(rfc, data, target, cv=kfold, scoring='accuracy')
	scores_gnb = cross_val_score(gnb, data, target, cv=kfold, scoring='accuracy')
	scores_xgb = cross_val_score(gnb, data, target, cv=kfold, scoring='accuracy')


	test_scores_logit.append((scores_logit.mean()))
	test_scores_svc.append((scores_svc.mean()))
	# test_scores_linear_svc.append((scores_linear_svc.mean()))
	test_scores_rfc.append((scores_rfc.mean()))
	test_scores_gnb.append((scores_gnb.mean()))
	test_scores_gnb.append((scores_xgb.mean()))

average = reduce(lambda x, y: x + y, test_scores_gnb) / float(len(test_scores_gnb))
cv_scores.append(average)

x_values = []
for i in range(len(randomstate)):
	x_values.append(randomstate[i])

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

print("The minimum misclassification error is %f" % min(MSE))
print("The maximum accuracy is %f" % (1- min(MSE)))

plt.plot(x_values, test_scores_logit, 'b', label='logit-liblinear-0.01')
plt.plot(x_values, test_scores_svc, 'r', label='svc')
# plt.plot(x_values, test_scores_linear_svc, 'c', label='svc-linear')
plt.plot(x_values, test_scores_rfc, 'm', label='Random Forest')
plt.plot(x_values, test_scores_gnb, 'k', label='Gaussian NB')
plt.plot(x_values, test_scores_xgb, 'c', label='XG Boost')
plt.legend(loc='upper right')

plt.show()
