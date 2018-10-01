import xgboost as xgb
from sklearn.grid_search import GridSearchCV

from sklearn import preprocessing 
import warnings
if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    le = preprocessing.LabelEncoder()
    le.fit([1, 2, 2, 6])
    le.transform([1, 1, 2, 6])
    le.inverse_transform([0, 0, 1, 2])
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

import pandas as pd
df = pd.read_csv("book2.csv")
X = df.iloc[:,0:14]
y = df.iloc[:,14]

xgb_model = xgb.XGBClassifier()
optimization_dict = {'max_depth': [1,2,3,4,5,6],
                     'n_estimators': [50,100,200],
                     }

optimization_dict1 = {'subsample': [0.8,0.9,1], 'max_delta_step': [0,1,2,4],'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
                     }


model = GridSearchCV(xgb_model, optimization_dict, 
                     scoring='accuracy', verbose=1)

model = GridSearchCV(xgb_model, optimization_dict1, 
                     scoring='accuracy', verbose=1)

model.fit(X,y)
print(model.best_score_)
print(model.best_params_)