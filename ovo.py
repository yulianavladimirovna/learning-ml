# MULTICLASS CLASSIFICATION
# ONE VS ONE
import itertools
from sklearn import datasets
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# load iris dataset
wine = datasets.load_wine()
# create a dataframe
wine_df = pd.DataFrame(wine.data)
wine_df['class'] = wine.target
a = wine.feature_names
a.append('class')
wine_df.columns = a
wine_df.insert(len(wine_df.columns), "Free", [1 for i in range(len(wine_df.index))],
               True)  # column of free coefficients

n_classes = len(wine_df['class'].unique())
clfs = dict()

for comb in itertools.combinations([*range(n_classes)], 2):
    # building a training sample
    df_tmp = wine_df.loc[wine_df['class'].isin(comb)]

    X_tmp = df_tmp.drop('class', axis=1)
    y_tmp = df_tmp['class']
    # building a binary classifier
    clf = SVC(kernel='linear', probability=True).fit(X_tmp, y_tmp)
    clf.fit(X_tmp, y_tmp)

    clfs[comb] = clf

# divide the sample into a training and a test sample
X = wine_df.drop('class', axis=1)
y = wine_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

y_pred = []
for index, row in X_test.iterrows():
    # comparing classifiers
    probs = {i: 1 for i in range(n_classes)}
    for comb in itertools.combinations([*range(n_classes)], 2):
        p_0, p_1 = clfs[comb].predict_proba(X_test.loc[index].to_frame().T)[0]
        probs[comb[0]] *= p_0
        probs[comb[1]] *= p_1
    # choose according to the maximum probability
    res = max(probs, key=probs.get)
    y_pred.append(res)

print(accuracy_score(y_test, y_pred))
