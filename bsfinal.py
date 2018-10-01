
#load csv and decide x and y
import pandas as pd
df = pd.read_csv("book2.csv")
X = df.iloc[:,0:14]
y = df.iloc[:,14]

#split test and train data
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=3)
X_train.fillna(X_train.mean())
#create pipeline
from sklearn import decomposition
from sklearn.pipeline import Pipeline
import numpy as np
import sklearn.metrics as sm
pca = decomposition.PCA(random_state=3)
# print(pca)
pipe = Pipeline(steps=[('pca', pca)])
# print(pipe)
# -------------------------------------------------------------------------------------------------

#append NB to pipeline
from sklearn.naive_bayes import GaussianNB
pipeline_nb = Pipeline([('pca',decomposition.PCA(n_components=10,random_state=3))])
nb = GaussianNB()
pipeline_nb
pipeline_nb.steps.append(('nb',nb))

pipeline_nb.fit(X_train,y_train)
prednb_labels = pipeline_nb.predict(X_test)

#print y predictions and mean squared error
# print('Naive Bayes Predicted Values:')
# print(prednb_labels)

#print accuracy
accuracynb = sm.accuracy_score(prednb_labels,y_test)
print("NB:")
e1=np.sqrt(sm.mean_squared_error(y_test, prednb_labels))
print 'Mean Squared Error:' , e1
print("accuracy is\t{:f}".format(accuracynb))
print("--------------------------------------------------------------------------------")

# -------------------------------------------------------------------------------------------------

#append Decision Tree classification to pipeline
from sklearn.tree import DecisionTreeClassifier
pipeline_cart = Pipeline([('pca',decomposition.PCA(n_components=10,random_state=3))])
cart = DecisionTreeClassifier()
pipeline_cart
pipeline_cart.steps.append(('cart',cart))

pipeline_cart.fit(X_train,y_train)
predcart_labels = pipeline_cart.predict(X_test)

#print y predictions and mean squared error
# print('CART Values:')
# print(predcart_labels)

#print accuracy
accuracycart = sm.accuracy_score(predcart_labels,y_test)
print("CART:")
e2=np.sqrt(sm.mean_squared_error(y_test, predcart_labels))
print 'Mean Squared Error:' , e2
print("accuracy is\t{:f}".format(accuracycart))
print("--------------------------------------------------------------------------------")

# -------------------------------------------------------------------------------------------------

#append K neighbours classification to pipeline
from sklearn.neighbors import KNeighborsClassifier
pipeline_knn = Pipeline([('pca',decomposition.PCA(n_components=10,random_state=3))])
knn = KNeighborsClassifier()
pipeline_knn
pipeline_knn.steps.append(('knn',knn))

pipeline_knn.fit(X_train,y_train)
predknn_labels = pipeline_knn.predict(X_test)

#print y predictions and mean squared error
# print('KNN Values:')
# print(predknn_labels)

#print accuracy
accuracyknn = sm.accuracy_score(predknn_labels,y_test)
print("KNN:")
e3=np.sqrt(sm.mean_squared_error(y_test, predknn_labels))
print 'Mean Squared Error:' , e3
print("accuracy is\t{:f}".format(accuracyknn))
print("--------------------------------------------------------------------------------")

# -------------------------------------------------------------------------------------------------

#append Feature importance classification to pipeline
from sklearn.ensemble import ExtraTreesClassifier
pipeline_fi = Pipeline([('pca',decomposition.PCA(n_components=10,random_state=3))])
fi = ExtraTreesClassifier()
pipeline_fi.steps.append(('fi',fi))

pipeline_fi.fit(X_train,y_train)
predfi_labels = pipeline_fi.predict(X_test)

#print y predictions and mean squared error
# print('Feature Importance Values:')
# print(predfi_labels)

#print accuracy
accuracyfi = sm.accuracy_score(predfi_labels,y_test)
print("Feature Importance:")
e4=np.sqrt(sm.mean_squared_error(y_test, predfi_labels))
print 'Mean Squared Error:' , e4
print("accuracy is\t{:f}".format(accuracyfi))
print("--------------------------------------------------------------------------------")

# -------------------------------------------------------------------------------------------------

#append  Random forest to pipeline
from sklearn.ensemble import RandomForestClassifier
pipeline_rf = Pipeline([('pca',decomposition.PCA(n_components=10,random_state=3))])
rf = RandomForestClassifier()
pipeline_rf.steps.append(('rf',rf))

pipeline_rf.fit(X_train,y_train)
predrf_labels = pipeline_rf.predict(X_test)

#print y predictions and mean squared error
# print('Random Forest Values:')
# print(predrf_labels)

#print accuracy
accuracyrf = sm.accuracy_score(predrf_labels,y_test)
print("Random Forest:")
e5=np.sqrt(sm.mean_squared_error(y_test, predrf_labels))
print 'Mean Squared Error:' , e5
print("accuracy is\t{:f}".format(accuracyrf))
print("--------------------------------------------------------------------------------")

# -------------------------------------------------------------------------------------------------

#append  Multi Layer perceptron to pipeline
from sklearn.neural_network import MLPClassifier
pipeline_mlp = Pipeline([('pca',decomposition.PCA(n_components=10,random_state=3))])
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
pipeline_mlp.steps.append(('mlp',mlp))

pipeline_mlp.fit(X_train,y_train)
predmlp_labels = pipeline_mlp.predict(X_test)

#print y predictions and mean squared error
# print('MLP Values:')
# print(predmlp_labels)

#print accuracy
accuracymlp = sm.accuracy_score(predmlp_labels,y_test)
print("MLP classifier:")
e6=np.sqrt(sm.mean_squared_error(y_test, predmlp_labels))
print 'Mean Squared Error:' , e6
print("accuracy is\t{:f}".format(accuracymlp))
print("--------------------------------------------------------------------------------")

# -------------------------------------------------------------------------------------------------

#append  XGBoost to pipeline
from xgboost import XGBClassifier
pipeline_xg = Pipeline([('pca',decomposition.PCA(n_components=10,random_state=3))])
xg = XGBClassifier()
pipeline_xg.steps.append(('xg',xg))

pipeline_xg.fit(X_train,y_train)
predxg_labels = pipeline_xg.predict(X_test)

#print y predictions and mean squared error
# print('XG Values:')
# print(predmlp_labels)

#print accuracy
accuracyxg = sm.accuracy_score(predxg_labels,y_test)
print("XGBoost classifier:")
e7=np.sqrt(sm.mean_squared_error(y_test, predxg_labels))
print 'Mean Squared Error:' , e7
print("accuracy is\t{:f}".format(accuracyxg))
print("--------------------------------------------------------------------------------")

#plotting agraph to compare algorithms
import matplotlib.pyplot as plt
names = ['NB','CART','KNN','FI','RF','MLPC','XGB']
accuracy_result = [accuracynb,accuracycart,accuracyknn,accuracyfi,accuracyrf,accuracymlp,accuracyxg]
mean_error = [e1,e2,e3,e4,e5,e6,e7]
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.clf()
plt.plot(names,accuracy_result,color='blue')
# plt.plot(names,mean_error,color='red')
plt.show()