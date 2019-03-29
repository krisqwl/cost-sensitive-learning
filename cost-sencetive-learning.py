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



