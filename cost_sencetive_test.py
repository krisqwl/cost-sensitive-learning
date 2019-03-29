'''
代价敏感的学习方法是机器学习领域中的一种新方法，它主要考虑在分类中，当不同的分类错误会导致不同的惩罚力度时如何训练分类器。
例如在医疗中，“将病人误诊为健康人的代价”与“将健康人误诊为病人的代价”不同；在金融信用卡盗用检测中，“将盗用误认为正常使用的代价”与将“正常使用误认为盗用的代价”也不同。
通常，不同的代价被表示成为一个N×N的矩阵Cost中，其中N是类别的个数。Cost[i,j]表示将一个i类的对象错分到j类中的代价。
按照对问题的解决方法的不同，对代价敏感学习的算法研究可以分成三类。


第一类代价敏感的学习方法关注于如何直接构造一个代价敏感的学习模型，对不同的分类器模型研究者们提出了不同的解决办法，它们包括
(1)决策树：Knoll等和Bradford等为决策树提出了代价敏感的剪枝方
法，Bradford等研究了在代价敏感的条件下如何对决策树进行剪枝使得
损失达到最小，研究表明基于拉普拉斯方法的剪枝方法能够取得最好的
效果，Drummond和Holte研究了代价敏感学习的决策树的节点分裂
方法。
(2)Boosting：Fan等研究着提出了代价敏感的Boosting算法Ada-Cost
(3)神经网络：Geibel和Wysotzki提出了基于Perceptron分类算法的代价敏感的学习方法，在文章中作者对不可分的类提出了代价敏感的参数更新规则。
例如Kukar和Kononenko为神经网络提出了新的后向传播算法，使之能够满足代价敏感学习的要求。
(4)Fumera和Roli[37]以及Bradford等从结构风险最小的角度来看代价敏感问题，提出了代价敏感的支持向量机分类算法。


第二类代价敏感的学习方法基于对分类结果的后处理，即按照传统的学习方法学习一个分类模型，然后对其分类结果按照贝叶斯风险理论对结果进行调整，
以达到最小的损失。和第一类代价敏感学习方法相比，这种方法的优点在于其不依赖于所使用的具体的分类器。
Domingos提出了一种叫做MetaCost的过程，它把底层的分类器看成一个黑箱子，不对分类器做任何的假设和改变，MetaCost可以应用到任何个数的基分类器和任何形式的代价矩阵上。
给定一个样例x，基分类器得出它属于第j个类的概率为Pr(j|x)，这样，认为x属于第i个类的贝叶斯最优预测的风险为：
R(i|x)=ΣP(j|x)C(i,j)(C(i,j)是把属于类别j的分为类别i的代价)


第三种代价敏感的学习方法基于传统的学习模型，通过改变原始训练数据的分布来训练得到代价敏感的模型。
Chan和Stolfo提出了层次化模型(Stratification)，把分布不均匀的训练数据调整为正负例均匀分布的数据。
Zadrozny等研究者基于cost-proportionate的思想，对训练数据调节权值，在实际应用中，其类似于Boosting算法，可以通过为分类模型调节权值来进行实现，又可以通过采样(subsampleing)来实现。
Abe等提出了对多类分类问题中如何实现代价敏感的学习进行了探讨，提出了一种新的迭代学习方法。
'''


#BayesMinimumRiskClassifier(meta-cost)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.models import BayesMinimumRiskClassifier
from costcla.metrics import savings_score
from sklearn import metrics

data=load_creditscoring1()
sets=train_test_split(data.data,data.target,data.cost_mat,test_size=0.33,random_state=0)
X_train,X_test,y_train,y_test,cost_mat_train,cost_mat_test=sets
f=RandomForestClassifier(n_estimators=100,random_state=0).fit(X_train,y_train)
y_prob_test=f.predict_proba(X_test)
y_pred_test_rf=f.predict(X_test)

#实用用ROCconvexhull算法做概率调整
f_bmr=BayesMinimumRiskClassifier(calibration=True)
f_bmr.fit(y_test,y_prob_test)
y_pred_test_bmr=f_bmr.predict(y_prob_test,cost_mat_test)


fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_rf)
print('The auc_score of using only RandomForestis{:.2f}'.format(metrics.auc(fpr,tpr)))
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_bmr)
print('The auc_score of using RandomForest and BayesMinimumRiskClassifieris{:.2f}'.format(metrics.auc(fpr,tpr)))


#Thresholding
#cost-sensetive技术主要用于临界值threshold的设置，理论最优值为T=cost(0,1)/(cost(0,1)+cost(1,0)),cost(0,1)为将正类预测为负类的代价
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.models import ThresholdingOptimization
from costcla.metrics import savings_score
data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.4, random_state=0)
X_train, X_validation_test, y_train, y_validation_test, cost_mat_train, cost_mat_validation_test = sets
X_validation, X_test, y_validation, y_test, cost_mat_validation, cost_mat_test= \
train_test_split(X_validation_test,y_validation_test,cost_mat_validation_test,test_size=0.25,random_state=0)
f = RandomForestClassifier(n_estimators=10,random_state=0).fit(X_train, y_train)
y_prob_validation = f.predict_proba(X_validation)
y_prob_test = f.predict_proba(X_test)
f_t = ThresholdingOptimization().fit(y_prob_validation, cost_mat_validation, y_validation)
y_pred_test_rf = f.predict(X_test)
y_pred_test_rf_t = f_t.predict(y_prob_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_rf)
print('The threshold is '+str(f_t.threshold_))
print('The auc_score of using only RandomForest is {:.2f}'.format(metrics.auc(fpr,tpr)))
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_rf_t)
print('The auc_score of using RandomForest and ThresholdingOptimization is {:.2f}'.format(metrics.auc(fpr,tpr)))


#CostSensitiveLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring2
from costcla.models import CostSensitiveLogisticRegression
from costcla.metrics import savings_score
data = load_creditscoring2()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=44)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
y_pred_test_lr = LogisticRegression(solver='lbfgs').fit(X_train, y_train).predict(X_test)
f = CostSensitiveLogisticRegression(solver='ga')
f.fit(X_train, y_train, cost_mat_train)
y_pred_test_cslr = f.predict(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_lr)
print('The auc_score of using only LogisticRegression is {:.2f}'.format(metrics.auc(fpr,tpr)))
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_cslr)
print('The auc_score of using CostSensitiveLogisticRegression is {:.2f}'.format(metrics.auc(fpr,tpr)))


#CostSensitiveDecisionTreeClassifier
#example-dependent
'''
1.构建树的过程中，cost-sensetive主要作用于impurity,Ic(S) = min(Cost(f0(S)), Cost(f1(S))),将其代入信息增益Gain(xj,lj)中，以cost-sensetive的方法来选择最优特征
2.修剪枝的过程，计算删除一个节点后代价
3.预测过程也同样使用cost-sensetive，对于每个leaf，模型训练完之后比较其中全预测为0和全为1的代价，谁的代价低就预测为谁
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score
data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
f = CostSensitiveDecisionTreeClassifier()
y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_rf)
print('The auc_score of using only LogisticRegression is {:.2f}'.format(metrics.auc(fpr,tpr)))
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_csdt)
print('The auc_score of using CostSensitiveLogisticRegression is {:.2f}'.format(metrics.auc(fpr,tpr)))


#ensemble-CostSensitiv-DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.models import CostSensitiveRandomForestClassifier
from costcla.models import CostSensitiveBaggingClassifier
from costcla.models import CostSensitivePastingClassifier
from costcla.models import CostSensitiveRandomPatchesClassifier
from costcla.metrics import savings_score
data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
y_pred_test_rf = RandomForestClassifier(n_estimators=10,random_state=0).fit(X_train, y_train).predict(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_rf)
print('The auc_score of using RandomForestClassifier is {:.2f}'.format(metrics.auc(fpr,tpr)))
for combination in ['majority_voting','stacking','stacking_proba','stacking_bmr', \
                       'stacking_proba_bmr','majority_bmr','weighted_bmr']:
    for f in [CostSensitiveRandomForestClassifier(n_estimators=10,combination=combination), \
              CostSensitiveBaggingClassifier(n_estimators=10,combination=combination), \
              CostSensitivePastingClassifier(n_estimators=10,combination=combination), \
              CostSensitiveRandomPatchesClassifier(n_estimators=10,combination=combination)]:
        y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
        print(f.__class__.__name__+' '+combination)
		fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_csdt)
        print('The auc_score is {:.2f}'.format(metrics.auc(fpr,tpr)))
        print('*'*60)


#Cost-proportionate rejection sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.sampling import cost_sampling
data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
X_cps_r, y_cps_r, cost_mat_cps_r =  cost_sampling(X_train, y_train, cost_mat_train, method='RejectionSampling')
y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_rf)
print('The rf\'s auc_score is {:.2f}'.format(metrics.auc(fpr,tpr)))
y_pred_test_rf_cps_r = RandomForestClassifier(random_state=0).fit(X_cps_r, y_cps_r).predict(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_rf_cps_r)
print('The rf_r\'s auc_score is {:.2f}'.format(metrics.auc(fpr,tpr)))




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
adacost=AdaCostClassifier()fit(X_train,y_train)
adaboost=AdaBoostClassifier().fit(X_train,y_train)

y_pre_adacost=adacost.predict(X_test)
y_pre_adaboost=adaboost.predict(X_test)

fpr,tpr,threshold=metrics.roc_curve(y_test,y_pre_adacost)
print('AdaCost{:.2f}'.format(metrics.auc(fpr,tpr)))
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_adaboost)
print('AdaBoost{:.2f}'.format(metrics.auc(fpr,tpr)))


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


#THresholding-moving
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def cost_tm(y_true,y_pre):
    if y_true==0:
        if y_pre==0:
            c=1
        elif y_pre==1:
            c=2
        else:
            c=4
    elif y_true==1:
        if y_pre==0:
            c=1.5
        elif y_pre==1:
            c=1
        else:
            c=3
    else:
        if y_pre==0:
            c=1.5
        elif y_pre==1:
            c=1.5
        else:
            c=1
    return c

X, y = make_classification(n_classes=3, class_sep=0.8,
weights=[0.1,0.2,0.7], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=5000, random_state=10)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=40)
clf=SVC(gamma='auto',probability=True).fit(X_train,y_train)
y_proba=clf.predict_proba(X_test)

o=np.empty_like(y_proba)
for j in range(len(y_proba)):
    for i in [0,1,2]:
        o[j][i]=sum([y_proba[j][i]*cost_tm(i,c) for c in [0,1,2]])
    o[j]=o[j]/sum(o[j])

y_pre=clf.predict(X_test)
y_tm_pre=o.argmax(axis=1)

print('SVC的分类报告为：')
print(classification_report(y_test,y_pre))

print('SVC+thresholding-moving的分类报告为：')
print(classification_report(y_test,y_tm_pre))
















