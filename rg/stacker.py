import sys
  
import joblib
import numpy as np

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

#------------------------------------------------------------------------
# -- initialize and get random forest
rf = RandomForestClassifier(max_depth=4, min_samples_leaf=60, \
        min_samples_split=4, n_estimators=100, random_state=42, n_jobs=-1)

rf = joblib.load('random_forest.joblib')

# -- xgboost
xgb = XGBClassifier(colsample_bytree=0.8, gamma=1.0, learning_rate=0.01, \
        max_depth=4, min_child_weight=5, n_estimators=200, reg_alpha=1.0, \
        reg_lambda=1.0, subsample=0.8, random_state=42, n_jobs=-1)
xgb = joblib.load('xgb.joblib')


# -- MLP classifier
mlp = MLPClassifier(early_stopping=True, solver='adam',  \
        validation_fraction=0.1, n_iter_no_change=100, random_state=42,   \
        activation='tanh', alpha=0.1, hidden_layer_sizes=(200, 200),  \
        max_iter=100000000,  batch_size='auto', learning_rate='adaptive',  \
        learning_rate_init=10e-5, power_t=0.5, shuffle=True,  tol=10e-10,  \
        verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,   \
        beta_1=0.9, beta_2=0.999, epsilon=1e-08)
mlp = joblib.load('mlp.joblib')


#-----------------------------------------------------------------------

# Get training and validation data:
# [ much more goes here ]
X_train = np.array(x1[indtrain,:])
X_test  = np.array(x1[indval,:])
y_train = np.array(v1[indtrain])
y_test  = np.array(v1[indval])

#-----------------------------------------------------------------------
# Using pre-trained estimators
stacker = StackingClassifier(
  estimators = [
      ('rf', rf),
      ('xgb', xgb),
      ('mlp', mlp)
      ],
  final_estimator=RandomForestClassifier(random_state=43),
  cv = 5
)

stacker.fit(X_train, y_train)
ystack = stacker.predict(X_test)
for i in range(0,len(y_test) ):
    print(i, ystack[i], y_test[i])

# scores for individual classifier, then stacker
for name, clf in stacker.named_estimators_.items():
    print('name = ',name,' score ', clf.score(X_test, y_test) )

# feature importance -------------------------------------------------
# Extract names from your estimators list
base_model_names = [name for name, clf in stacker.estimators]

# Map them to the final estimator's importances
importances = stacker.final_estimator_.feature_importances_

for name, score in zip(base_model_names, importances):
    print(f"Base Model: {name:5} | Importance to Stacker: {score:.4f}")

# For the base model's random forest
base_rf = stacker.named_estimators_['rf']
for i, score in enumerate(base_rf.feature_importances_):
    print(f"RF {i}: {score:.4f}")

