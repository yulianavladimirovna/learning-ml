# MULTICLASS CLASSIFICATION
# ONE VS REST
from sklearn import datasets
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# load iris dataset
iris = datasets.load_iris()
# create a dataframe
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target
iris_df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.insert(len(iris_df.columns), "Free", [1 for i in range(len(iris_df.index))],
               True)  # column of free coefficients

n_classes = len(iris_df['class'].unique())
clfs = dict()

for cl in range(n_classes):
    # building a training sample
    df_tmp = iris_df.copy()
    keys = [*range(n_classes)]
    values = [-1] * n_classes
    values[cl] = 1
    d = dict(zip(keys, values))
    df_tmp['class'].replace(d, inplace=True)

    X_tmp = df_tmp.drop('class', axis=1)
    y_tmp = df_tmp['class']
    # building a binary classifier
    clf = SVC(kernel='linear', probability=True).fit(X_tmp, y_tmp)
    clf.fit(X_tmp, y_tmp)

    clfs[cl] = clf

# divide the sample into a training and a test sample
X = iris_df.drop('class', axis=1)
y = iris_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

y_pred = []
for index, row in X_test.iterrows():
    # comparing classifiers
    probs = {i: clfs[i].predict_proba(X_test.loc[index].to_frame().T)[0][1] for i in range(n_classes)}
    # choose according to the maximum probability
    res = max(probs, key=probs.get)
    y_pred.append(res)

print(accuracy_score(y_test, y_pred))
