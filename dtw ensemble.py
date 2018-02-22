import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_curve,auc

df = pd.read_csv("fil.csv")
df.drop(df.columns[[0]],axis=1,inplace=True)


y = df[df.columns[[-1]]]
x = df.drop(df.columns[[4097]],axis=1,inplace=False)

print(y.head())
print(x.head())


import pywt
n_channels = 5
ll=[]
for a in range(0,500):
	ll = ll + [pywt.wavedec(x.iloc[a], pywt.Wavelet("db1"), level=n_channels-1)]



			   
print(ll[0][0])



fig, axs = plt.subplots(5, 1, sharex=True)

from pandas import Series
series = Series(ll[0][0])

axs[0].plot(series)

series = Series(ll[0][1])

axs[1].plot(series)

series = Series(ll[0][2])

axs[2].plot(series)

series = Series(ll[0][3])

axs[3].plot(series)

series = Series(ll[0][4])

axs[4].plot(series)


plt.title("Decomposition of time series")


plt.show()




def mean_signal(m):
    return np.mean(m)

def std_signal(m):
    return np.std(m)

def mean_square_signal(m):
    return np.mean(m ** 2)

def abs_diffs_signal(m):
    return np.sum(np.abs(np.diff(m)))

def skew_signal(m):
    return scipy.stats.skew(m)
	




l = []
for a in range(0,500):
	mean = []
	std = []
	max = []
	mean_square = []
	abs_diff = []
	skew = []
	min = []
	for b in range(0,5):
		mean = mean + [mean_signal(np.array(ll[a][b]))]
		std = std + [std_signal(np.array(ll[a][b]))]
		max = max + [np.array(ll[a][b]).max()]
		mean_square= mean_square + [mean_square_signal(np.array(ll[a][b]))]
		abs_diff = abs_diff + [abs_diffs_signal(np.array(ll[a][b]))]
		skew = skew + [skew_signal(np.array(ll[a][b]))]
		min = min + [np.array(ll[a][b]).min()]
		
	
	
	l= l + [mean+std+max+min+mean_square+abs_diff+skew]
    

features = pd.DataFrame(l)
print(features.head())




names = []
ac = []


from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, features, y, cv=kfold)
names = names + ["DT Bagging"]
ac = ac + [results.mean()]



from sklearn.ensemble import RandomForestClassifier

seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, features, y, cv=kfold)
names = names + ["RF Bagging"]
ac = ac + [results.mean()]




from sklearn.ensemble import ExtraTreesClassifier

seed = 7
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, features, y, cv=kfold)
names = names + ["ET Bagging"]
ac = ac + [results.mean()]



from sklearn.ensemble import AdaBoostClassifier

seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, features, y, cv=kfold)
names = names + ["AdaBoost"]
ac = ac + [results.mean()]



from sklearn.ensemble import GradientBoostingClassifier

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, features, y, cv=kfold)
names = names + ["GradientBoost"]
ac = ac + [results.mean()]



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, features, y, cv=kfold)
names = names + ["Voting"]
ac = ac + [results.mean()]



print(names)
print(ac)


y_pos = np.arange(len(names))
plt.bar(y_pos,ac,align="center")
plt.xticks(y_pos,names)
plt.ylabel("Accuary")
plt.title("Ensemble DWT")
plt.show()



table={'Name' : names,'Accuracy':ac}
table = pd.DataFrame(table,columns=['Name','Accuracy'])

print(table)

table.to_csv("ensemble dwt.csv")