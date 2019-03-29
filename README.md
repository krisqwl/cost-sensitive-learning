# cost-sensitive-learning
## Cost-Sensitive Learning
* 代价敏感的学习方法是机器学习领域中的一种新方法，它主要考虑在分类中，当不同的分类错误会导致不同的惩罚力度时如何训练分类器。
   
	* 例如在医疗中，“将病人误诊为健康人的代价”与“将健康人误诊为病人的代价”不同；在金融信用卡盗用检测中，“将盗用误认为正常使用的代价”与将“正常使用误认为盗用的代价”也不同。
	* 通常，不同的代价被表示成为一个$N \times N$的矩阵$Cost$中，其中N是类别的个数。$Cost[i|j]$表示将一个实际属于j类的对象错分到i类中的代价。
* 按照对问题的解决方法的不同，对代价敏感学习的算法研究可以分成三类。
	* 第一类代价敏感的学习方法关注于如何直接构造一个代价敏感的学习模型。
	* 第二类代价敏感的学习方法基于对分类结果的后处理，即按照传统的学习方法学习一个分类模型，然后对其分类结果按照贝叶斯风险理论对结果进行调整，以达到最小的损失。与第一类代价敏感学习方法相比，这种方法的优点在于其不依赖于所使用的具体的分类器。
	* 第三种代价敏感的学习方法基于传统的学习模型，通过改变原始训练数据的分布来训练得到代价敏感的模型。

### 重要概念
**Classification cost matrix**


||Actual Positive($y_i=1$)|Actual Negative($y_i=0$)|
|:---:|:---:|:---:|
|**Predicted Positive($c_i=1$)**|$C_{TP_i}$|$C_{FP_i}$|
|**Predicted Negative($c_i=0$)**|$C_{FN_i}$|$C_{TN_i}$|

**注意**: _要与sklearn中的混淆矩阵区分开_

**confusion_matrix**

||Predicted Negative($c_i=0$)|Predicted Positive($c_i=1$)|
|:---:|:---:|:---:|
|**Actual Negative($y_i=0$)**|$TN_i$|$FP_i$|
|**Actual Positive($y_i=1$)**|$FN_i$|$TP_i$|

## 代价敏感的学习模型
### CostSensetive-LogisticRegression
* 最小化损失函数: $J^c(\theta)=\frac{1}{N}\sum_{i=1}^{N}{y_i(h_{\theta}(X_i)C_{TP_i}+(1-h_{\theta}(X_i))C_{FN_i})+(1-y_i)(h_{\theta}(X_i)C_{FP_i}+(1-h_{\theta}(X_i))C_{TN_i})}$
 	* $\hat{p}=P(y=1|X_i)=h_{\theta}(X_i)=g(\sum_{j=1}^k{\theta^jx_i^j})$
 		* $g(z)=\frac{1}{1+e^{-z}}$  

 * 采用极大似然估计或者梯度下降法来获取最优解
 	* 注：普通LogisticRegression的损失函数为
 		* $J(\theta)=\frac{1}{N}\sum_i^N{-y_ilog(h_{\theta}(X_i))-(1-y_i)log(1-h_{\theta}(X_i))}$
 		* Cost-Sensitive LogisticRegression则是将损失函数换成cost损失函数

### CostSensitiveDecisionTreeClassifier
* CS树的构造：
	* $I_c(S)=min\{Cost(f_0(S)),Cost(f_1(S))\}$ 
		* $Cost(f(S))=\sum_{i=1}^N{y_i(c_iC_{TP_i}+(1-c_i)C_{FN_i})+(1-y_i)(c_iC_{FP_i}+(1-c_i)C_{TN_i})}$
	* $Gain(C^j,l^j)=I_c(S)-\frac{|S^l|}{|S|}I_c(S^l)-\frac{|S^r|}{|S|}I_c(S^r)$  
	* 通过Cost-Sensitive增益的方式来确定分裂特征和分裂标准
	* 注：和普通的决策树相比，CS的方法舍弃了信息熵的判别准则，将其改为以CS准则来选取最优分裂点
* CS树的剪枝：
	* $PC_c=Cost(f(S))-Cost(f^*(S))$
		* $f^*(S)$为去掉某节点后的分类器
		* $PC_c$越大，说明该节点被去掉后损失减少越多，该节点便有更大的可能被剪去

### 基于Decision Tree的ensemble方法
* 样本的子集的提取方法：
	* Bagging
		* 对原始数据进行有放回的随机抽样(bootstrap) 
	* Pasting
		* 对原始数据进行无放回的随机抽样
	* Random Forests
		* 选取分裂节点时，随机选取一组特征子集，然后对原始数据进行有放回的随机抽样
	* Random Patches
		* 每个基础分类器的训练使用样本点和特征的双重随机Bootstrap子集 
* 分类器的组合方法：
	* Majority voting
	* Cost-sensitive weighted voting
		* $f_{w,v}(S,M,\alpha)=argmax_{c \in {0,1}} \sum_{j=1}^T{\alpha_jI_c(M_j(S))}$
		* $\alpha_j=\frac{Savings(M_j(S_J^{oob}))}{\sum_{j=1}^T{Savings(M_j(S_j^{oob}))}}$
			* $S_j^{oob}=S-S_j$ 
			* $Savings(f(S))=\frac{Cost_l(S)-Cost(f(S))}{Cost_l(S)}$
			* $Cost_l(S)=min\{Cost(f_0(S)),Cost(f_1(S))\}$
	* Cost-sensitive stacking(基于CSLR的参数估计)
		* 将不同的基本分类器结合起来，在它们的基础输出上学习第二级算法。 
		* $f_s(S,M,\beta)=g(\sum_{j=1}^T{\beta_jM_j(S)})$ 
			* $g(z)=\frac{1}{1+e^{-z}}$   
		* 最小化损失函数：$J(S,M,\beta)=\sum_{i=1}^{N}{y_i(f_S(x_i,M,\beta)C_{TP_i}+(1-f_S(x_i,M,\beta))C_{FN_i})+(1-y_i)(f_S(x_i,M,\beta)C_{FP_i}+(1-f_S(x_i,M,\beta))C_{TN_i})}$ 

### 基于Decision Tree的其他组合方法
* stacking
	* a Cost Sensitive Logistic Regression is used to learn the combination
* stacking\_bmr
	* a Cost Sensitive Logistic Regression is used to learn the probabilities and a BayesMinimumRisk for the prediction
* stacking\_proba\_bmr
	* a Cost Sensitive Logistic Regression trained with the estimated probabilities is used to learn the probabilities, and a BayesMinimumRisk for the prediction
* majority\_bmr
	* the BayesMinimumRisk algorithm is used to make the prediction using the predicted probabilities of majority_voting
* weighted\_bmr
	* the BayesMinimumRisk algorithm is used to make the prediction using the predicted probabilities of weighted_voting

### AdaCost算法（有待优化）
* 基于AdaBoost算法cost-sensitive改进算法

### 测试
* 基于一份模拟数据集，该数据集正负样本比例为1:50，数据量为50000
* 针对AdaBoost，AdaCost\_class\_dependent，AdaCost\_example\_dependent，分别做了5次测试，以AUC得分作为评价标准。

### 测试结果
> 
|**AUC_SCORE**|测试一|测试二|测试三|测试四|测试五|平均值|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**AdaBoost**|0.75|0.78|0.80|0.79|0.78|0.78|
|**AdaCost\_class\_dependent**|0.82|0.83|0.81|0.84|0.83|0.826|
|**AdaCost\_example\_dependent**|0.83|0.84|0.82|0.87|0.84|***0.84***|

> 在数据类别严重不平衡时，两种AdaCost方法要明显优于传统的AdaBoost方法，说明加入Cost\_Sensitive的方法可以改善样本不平衡所带来的问题。

> 虽然就平均值而言，AdaCost\_example\_dependent效果最好，但是两种AdaCost方法的优劣不能明确评价，因为针对不同数据类型，cost\_matrix的设置需要具体情况具体分析。

## 基于对分类结果的事后处理（meta-cost）
* BayesMinimumRisk
	* 先使用一个传统分类器得到预测结果和预测概率
	* 再使用贝叶斯最小风险准则做再次处理
		* $R(x_i│1)=p_iC_{TP_i}+(1−p_i)C_{FN_i}$
		* $R(x_i│0)=p_iC_{FP_i}+(1−p_i)C_{TN_i}$

* Threshold-Moving
	* $O_i^*=\eta\sum_{c=1}^C=O_iCost[i,c]$
		* $\eta$使得$\sum_{i=1}^CO_i^*=1$

## 改变原始训练数据的分布
* Cost-proportionate rejection sampling
	* Under-Sampling
	* 以随机均匀分布和错判代价比为基准
	* 错判代价高的更倾向于被选择，错判代价低的倾向于被拒绝
