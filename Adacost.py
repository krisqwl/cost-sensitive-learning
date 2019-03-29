#继承实现
#AdaCost 
#class-dependent cost-sensitive classifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
class AdaCostClassifier(AdaBoostClassifier):
    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])


        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                            * ((n_classes - 1.) / n_classes)
                            * (y_coding * np.log(y_predict_proba)).sum(axis=1))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0))*
                                   self._beta(y, y_predict))

        return sample_weight, 1., estimator_error
    
    def _beta(self, y, y_predict):
        res = []
        for i in zip(y, y_predict):
            if i[0] == 0 and i[0] == 1:
                res.append(4)   # FP
            elif i[0] == 1 and i[1] == 0:
                res.append(1.5)   # FN
            elif i[0] == 1 and i[1] == 1:    
                res.append(0.5)   # TP
            else:
                res.append(0.5)
        return np.array(res)

from costcla.datasets import load_creditscoring1
data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=44)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
adacost=AdaCostClassifier().fit(X_train,y_train)
adaboost=AdaBoostClassifier().fit(X_train,y_train)

y_pre_adacost=adacost.predict(X_test)
y_pre_adaboost=adaboost.predict(X_test)

fpr,tpr,threshold=metrics.roc_curve(y_test,y_pre_adacost)
print('AdaCost{:.2f}'.format(metrics.auc(fpr,tpr)))
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_adaboost)
print('AdaBoost{:.2f}'.format(metrics.auc(fpr,tpr)))

#原始实现
#AdaCost 
#example-dependent cost-sensitive classifier
import numpy as np
import math
def decision_stump(x,y,w=np.ones(len(y))):
    temp_accu=0
    temp_threshold=0
    idx=0
    temp_accuracy=0
    temp_thresholding=1
    t_compare='la'
    temp_compare='la'
    for i in range(x.shape[1]):
        thresholds=[np.percentile(x[:,i],per) for per in np.random.rand(5)]
        for threshold in thresholds:
            compare='la'
            temp_clf=[1 if temp_x>=threshold else -1 for temp_x in x[:,i]]
            accu=(((temp_clf*y)==1).dot(np.ones(y.shape[0])))/x.shape[0]
            if accu<0.5:
                temp_clf[temp_clf==1]=0
                temp_clf[temp_clf==-1]=1
                temp_clf[temp_clf==0]=-1
                compare='le'
                accu=1-accu
            err=(((temp_clf*y)!=1).dot(w))/x.shape[0]
            if accu>=temp_accu:
                temp_accu=accu
                temp_threshold=threshold
                t_compare=compare
                t_err=err
        if temp_accu>=temp_accuracy:
            temp_accuracy=temp_accu
            temp_thresholding=temp_threshold
            idx=i
            temp_compare=t_compare
            temp_err=t_err
    return temp_err,temp_accuracy,temp_thresholding,idx,temp_compare

def h(x,idx,threshold,temp_compare):
    if temp_compare=='la':
        y_pre=[1 if temp_x>=threshold else -1 for temp_x in x[:,idx]]
    else:
        y_pre=[1 if temp_x<threshold else -1 for temp_x in x[:,idx]]
    return y_pre

def beta(y_pre,y,cost_mat):
    temp_cost=np.array([])
    for y_pre_i,y_i,cost_mat_i in zip(y_pre,y,cost_mat):
        if y_pre_i==1 and y_i==1:
            temp_cost=np.append(temp_cost,cost_mat_i[2])
        elif y_pre_i==1 and y_i==-1:
            temp_cost=np.append(temp_cost,cost_mat_i[0])
        elif y_pre_i==-1 and y_i==-1:
            temp_cost=np.append(temp_cost,cost_mat_i[3])
        else:
            temp_cost=np.append(temp_cost,cost_mat_i[1])
    return temp_cost

def ada_cost_fit(X,y,cost_matrix,T=6):
    D=np.ones(y.shape[0])/sum(np.ones(y.shape[0]))
    alphas=[]
    thresholds=[]
    compares=[]
    idxs=[]
    iteration_num=1
    while iteration_num<=T:
        err,accuracy,threshold,idx,compare=decision_stump(X,y,D)
        y_pre=h(X,idx,threshold,compare)
        y_pre=np.array(y_pre)
        err=1-accuracy
        alpha=1/2*math.log((1-err)/err)
        exp_temp=np.exp(-alpha*y*y_pre*beta(y_pre,y,cost_matrix))
        D=D*exp_temp
        D=D/(D.dot(np.ones_like(D)))
        alphas.append(alpha)
        thresholds.append(threshold)
        compares.append(compare)
        idxs.append(idx)
        iteration_num+=1
    return alphas,thresholds,compares,idxs

def ada_cost_pre(X,y,alphas,thresholds,compares,idxs):
    y_pre=np.zeros_like(y)
    for alpha,threshold,compare,idx in zip(alphas,thresholds,compares,idxs):
        y_pre_temp=h(X,idx,threshold,compare)
        y_pre=alpha*np.array(y_pre_temp)+y_pre
        y=[-1 if y_t<=0 else 1 for y_t in y_pre]
    return y
