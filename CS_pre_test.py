import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from costcla.models import BayesMinimumRiskClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring2
from costcla.models import CostSensitiveLogisticRegression
from costcla.models import CostSensitiveDecisionTreeClassifier

from costcla.models import CostSensitiveBaggingClassifier
from costcla.models import CostSensitivePastingClassifier
from costcla.models import CostSensitiveRandomPatchesClassifier
from costcla.sampling import cost_sampling
from costcla.metrics import savings_score


data = load_creditscoring2()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=10)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_rf)
print('The auc_score of RandomForest is {:.2f}'.format(metrics.auc(fpr,tpr)))
print('*'*90)

y_prob_test= RandomForestClassifier(random_state=0).fit(X_train, y_train).predict_proba(X_test)

f_bmr=BayesMinimumRiskClassifier(calibration=True)
f_bmr.fit(y_test,y_prob_test)
y_pred_test_bmr=f_bmr.predict(y_prob_test,cost_mat_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_bmr)
print('The auc_score of using RandomForest and BayesMinimumRiskClassifieris{:.2f}'.format(metrics.auc(fpr,tpr)))
print('*'*90)


f = CostSensitiveLogisticRegression(solver='ga')
f.fit(X_train, y_train, cost_mat_train)
y_pred_test_cslr = f.predict(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_lr)
print('The auc_score of CostSensitiveLogisticRegression is {:.2f}'.format(metrics.auc(fpr,tpr)))
print('*'*90)


f = CostSensitiveDecisionTreeClassifier()
f.fit(X_train, y_train, cost_mat_train)
y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_csdt)
print('The auc_score of using CostSensitiveDecisionTreeClassifier is {:.2f}'.format(metrics.auc(fpr,tpr)))
print('*'*90)



for f in [CostSensitiveRandomForestClassifier(n_estimators=10), \
          CostSensitiveBaggingClassifier(n_estimators=10), \
          CostSensitivePastingClassifier(n_estimators=10), \
          CostSensitiveRandomPatchesClassifier(n_estimators=10)]:
    y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    print(f.__class__.__name__)
    fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_csdt)
    print('The auc_score is {:.2f}'.format(metrics.auc(fpr,tpr)))
    print('*'*90)
    
    
X_cps_r, y_cps_r, cost_mat_cps_r =  cost_sampling(X_train, y_train, cost_mat_train, method='RejectionSampling')
y_pred_test_rf_cps_r = RandomForestClassifier(random_state=0).fit(X_cps_r, y_cps_r).predict(X_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_test_rf_cps_r)
print('The rf_r\'s auc_score is {:.2f}'.format(metrics.auc(fpr,tpr)))