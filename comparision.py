import numpy as np
import matplotlib.pyplot as plt

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%.2f' % (height),
                ha='center', va='bottom')

N = 5
acc_wo_stan = (60.28, 64.57, 80.09, 73.17, 66.10) #without standardization
acc_w_stan = (60.38, 55.11, 80.09, 71.76, 43.36) #with standardization

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, acc_wo_stan, width, color='r')


rects2 = ax.bar(ind + width, acc_w_stan, width, color='b')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy', fontsize =16)
ax.set_title('A Comparision of 5 Supervised Algorithms', fontsize =16)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('KNN', 'Logistic Regression', 'XGB', 'Random Forest', 'SVM'), fontsize =16)
plt.ylim((0,100))
ax.legend((rects1[0], rects2[0]), ('Without Standardization', 'With Standardization'))

autolabel(rects1)
autolabel(rects2)

plt.show()