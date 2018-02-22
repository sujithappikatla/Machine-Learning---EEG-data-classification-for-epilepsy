import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import roc_curve,auc

df = pd.read_csv("fil.csv")
df.drop(df.columns[[0]],axis=1,inplace=True)


y = df[df.columns[[-1]]]
x = df.drop(df.columns[[4097]],axis=1,inplace=False)



"""
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
	

l=[]
for a in range(0,500):
	mean = mean_signal(x.iloc[a])
	std = std_signal(x.iloc[a])
	max = x.iloc[a].max()
	min = x.iloc[a].min()
	
	l= l + [[mean,std,max,min]]
    
features = pd.DataFrame(l)

"""

import pywt
n_channels = 5
ll=[]
for a in range(0,500):
	ll = ll + [pywt.wavedec(x.iloc[a], pywt.Wavelet("db2"), level=n_channels-1)]




fig, axs = plt.subplots(5, 1, sharex=True)

from pandas import Series
series = Series(x.iloc[0])

axs[0].plot(series)

series = Series(x.iloc[100])

axs[1].plot(series)

series = Series(x.iloc[200])

axs[2].plot(series)

series = Series(x.iloc[300])

axs[3].plot(series)

series = Series(x.iloc[400])

axs[4].plot(series)

plt.show()







    
"""


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
		min = min + [np.array(ll[a][b]).min()]
		
	
	
	l= l + [mean+std+max+min]
    

dwtfeatures = pd.DataFrame(l)



from sklearn.metrics import confusion_matrix

def sensitivity(classifier,X,y):
    y_pre = classifier.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    res = (tp/(tp+fn))*100
    return res


def specificity(classifier,X,y):
    y_pre = classifier.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    res = (tn/(tn+fp))*100
    return res


def acuracy(classifier,X,y):
    y_pre = classifier.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    res = ((tp+tn)/(tp+fn+tn+fp))*100
    return res

def ppv(classifier,X,y):
    y_pre = classifier.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    res = (tp/(tp+fp))*100
    return res


def npv(classifier,X,y):
    y_pre = classifier.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    res = (tn/(tn+fn))*100
    return res

def au(classifier,X,y):
    y_pre = classifier.predict(X)
    fpr,tpr,t = roc_curve(y,y_pre)
    roc_auc = auc(fpr,tpr)
    return roc_auc





from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors=13)
from sklearn import svm
clf3 = svm.SVC(kernel="linear")
from sklearn.naive_bayes import GaussianNB
clf4 = GaussianNB()
from sklearn import tree
clf5 = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
clf6 = BaggingClassifier(base_estimator=cart, n_estimators=100, random_state=7)
from sklearn.ensemble import RandomForestClassifier
clf7 = RandomForestClassifier(n_estimators=100, max_features=4)
from sklearn.ensemble import ExtraTreesClassifier
clf8 = ExtraTreesClassifier(n_estimators=100, max_features=4)
from sklearn.ensemble import AdaBoostClassifier
clf9 = AdaBoostClassifier(n_estimators=30, random_state=7)
from sklearn.ensemble import GradientBoostingClassifier
clf10 = GradientBoostingClassifier(n_estimators=100, random_state=7)
from sklearn.ensemble import VotingClassifier
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = GaussianNB()
estimators.append(('gaussianb', model3))
clf11 = VotingClassifier(estimators)



clf = [clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8,clf9,clf10,clf11]
metrics = [sensitivity,specificity,acuracy,ppv,npv,au]
xset = [features,dwtfeatures]

#evaluation metrics

ss = []
eva = []
for var in clf:
    for fea in xset:
        ss=[]
        for metric in metrics:
            sen = cross_val_score(estimator=var,X=fea,y=y.values.ravel(),scoring=metric,cv=10)
            ss = ss + [sen.mean()]
        eva = eva + [ss]
        
        
eva = pd.DataFrame(eva)
eva.to_csv("evaluation_metrics.csv")        
   
#confusion matrci

x_train,x_test,y_train,y_test = train_test_split(features,y,test_size=0.2,random_state=4)     
dwtx_train,dwtx_test,dwty_train,dwty_test = train_test_split(dwtfeatures,y,test_size=0.2,random_state=4)     


cmatrix = []
for var in clf:
    i=0
    for fea in range(0,2):
        if i==0:
            var.fit(x_train,y_train)
            y_pre = var.predict(x_test)
            cm = confusion_matrix(y_test, y_pre)
            cmatrix = cmatrix + [cm]
            i =i + 1
        else:
            var.fit(dwtx_train,dwty_train)
            dwty_pre = var.predict(dwtx_test)
            cm = confusion_matrix(dwty_test, dwty_pre)
            cmatrix = cmatrix + [cm]
            

"""
"""
#plot


cd = [[{"clf":clf1,"name":"Logistic","color":"green"},{"clf":clf2,"name":"KNN","color":"blue"},{"clf":clf3,"name":"SVC","color":"black"},{"clf":clf4,"name":"GaussianNB","color":"cyan"},{"clf":clf5,"name":"Decision Trees","color":"orange"}],[{"clf":clf6,"name":"DT Bagging","color":"green"},{"clf":clf7,"name":"RF Bagging","color":"blue"},{"clf":clf8,"name":"ET Bagging","color":"black"},{"clf":clf9,"name":"AdaBoost","color":"cyan"},{"clf":clf10,"name":"GradientBoost","color":"orange"},{"clf":clf11,"name":"Voting","color":"pink"}]]

for clf_dict in cd:
    plt.title('ROC - Before DWT')
    for dic in clf_dict:
        dic['clf'].fit(x_train,y_train)
        y_pre = dic['clf'].predict(x_test)
        fpr,tpr,t = roc_curve(y_test,y_pre)
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr, tpr, dic["color"],label=dic["name"]+' (AUC = %0.2f)'% roc_auc)
    
    
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.show()
    
    
    plt.title('ROC - After DWT')
    for dic in clf_dict:
        dic['clf'].fit(dwtx_train,dwty_train)
        dwty_pre = dic['clf'].predict(dwtx_test)
        fpr,tpr,t = roc_curve(dwty_test,dwty_pre)
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr, tpr, dic["color"],label=dic["name"]+' (AUC = %0.2f)'% roc_auc)
    
    
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.show()

"""
