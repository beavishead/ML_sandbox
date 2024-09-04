"""
PROBLEM : BINARY CLASSIFICATION

TASK: GIVEN THE SET OF FEATURES AND TARGETS TRAIN THE CLASSIFIER THAT GIVES AT LEAST 94%
ACCURACY ON THE HIDDEN TEST SET

THE FOLLOWING CODE GIVES THE FULL THOUGHT AND TRIAL-AND-ERROR PROCCESS TO FIND THE SOLUTION

OUTLINE:
1** - import and preprocess data
2** - split into train and validation set
3** - train logistic regression and gradient boosting classifier
4** - check accuracy on the validation set(with drawing conclusion and follow-up steps)
5** - apply PCA dimensionality reduction to mitigate features' correlation
6** - train logreg, gradboost and random forest with PCA reduced features
7** - drop highly correlated features
8** - Variance inflation factor
9** - Conclusion
"""


### 1 import train data

# import pandas
import pandas as pd

# upload into pandas dataframe
path="ML_sandbox/static/data/binary_classification/train.csv"
df = pd.read_csv(path)

# split into features target
features = df.iloc[:,:-1]
target = df.iloc[:,-1]

# standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled= scaler.fit_transform(features)

### 2 split into train and val set
from sklearn.model_selection import train_test_split
X_train, X_val,y_train,y_val = train_test_split(features_scaled,target,test_size=0.2,random_state=42,shuffle=True)

### 3 train logistic regression and gradient boost model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

log_reg = LogisticRegression(penalty='l2', solver='liblinear')
log_reg.fit(X_train,y_train)

grad_boost = GradientBoostingClassifier()
grad_boost.fit(X_train,y_train)

### 4 check models accuracy
from sklearn.metrics import accuracy_score
print('logistic regression predictor: ', accuracy_score(y_val, log_reg.predict(X_val)))
#print('gradient boosting predictor: ', accuracy_score(y_val, grad_boost.predict(X_val)))
### 5 make further conlcusions and implications

"""
INTE$MIDIATE CONCLUSION
the suggested models give only 86% and 87% accuracy on the validation set,
but we need the accuracy at least 94%

FU$THE$ STEPS:
-check for correlated features, perhaps delete some of them
-apply dimensionality reduction
"""

### 6 apply dimensionality reduction using PCA

# import pca library
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=None)
X_pca = pca.fit_transform(X_train)

#instantiate the class with num of components that explain at least 95% variance
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)

#apply same transformator to validation set
X_val_pca = pca.transform(X_val)

# train and validate new models
log_reg_pca = LogisticRegression()
log_reg_pca.fit(X_train_pca, y_train)

print("logreg using pca: ",accuracy_score(y_val, log_reg_pca.predict(X_val_pca)))

grad_boost_pca = GradientBoostingClassifier()
grad_boost_pca.fit(X_train_pca, y_train)

print("gradboost using pca: ", accuracy_score(y_val, grad_boost_pca.predict(X_val_pca)))

from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier()
forest_classifier.fit(X_train_pca, y_train)

print('random forest using pca: ', accuracy_score(y_val, forest_classifier.predict(X_val_pca)))

"""
INTE$MEDIATE $ESULTS:

pca dim reduction didn't bring anything
let's build the correlation matrix and try to choose relevant features
"""
# import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#build correlation matrix
corr_matrix = pd.DataFrame(features_scaled).corr()

#make visualization
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True,cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

# let's find the pairs of correlated features and exclude every first in the pair
high_corr_feat= np.where(corr_matrix>0.8)
high_corr_feat = [(corr_matrix.index[x], corr_matrix.columns[y]) for x,y in zip(*high_corr_feat) if x!=y and x<y]
print(high_corr_feat)

drop_features = [df.columns[x[0]] for x in high_corr_feat]
df_reduced = df.drop(columns=drop_features)
print(df_reduced.shape)

#now let's build the correlation matrix again and train any model on this reduced feature set
df_reduced.corr()

#visualize
plt.figure(figsize=(10,8))
sns.heatmap(df_reduced.corr(), annot=True,cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

#build model
features_tr, features_val, target_tr, target_val = train_test_split(df_reduced.iloc[:,:-1], df_reduced.iloc[:,-1] ,train_size=0.8)

new_scaler = StandardScaler()
features_tr= new_scaler.fit_transform(features_tr)
features_val = new_scaler.fit_transform(features_val)

grad_boost_red=GradientBoostingClassifier()
grad_boost_red.fit(features_tr,target_tr)

print("acc score with reduced features: ", accuracy_score(target_val, grad_boost_red.predict(features_val)))

log_reg_red=LogisticRegression()
log_reg_red.fit(features_tr,target_tr)

print("acc score with reduced features and Logistic regression: ", accuracy_score(target_val, log_reg_red.predict(features_val)))

"""
INTE$MEDIATE #ESULTS
this approach hasn't worked either
the journey continues
"""


### 8 Varience Inflation Factor
# install and import the library
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df.iloc[:,:-1]

# Set the display option to avoid scientific notation
pd.set_option('display.float_format', lambda x: '%.6f' % x)

vif_data = pd.DataFrame()
vif_data['feature']=X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


# CONCLUSION
# Gradient boosting classifier without any feature reduction gives the highest accuracy score 87%

