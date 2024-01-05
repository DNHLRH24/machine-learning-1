from lazypredict.Supervised import LazyClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

cc = pd.read_csv('cc.csv')

X = cc.drop('output',axis=1)
y = cc['output']

scaler=StandardScaler()
X=pd.DataFrame(scaler.fit_transform(X))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)


print(models)
print(predictions)