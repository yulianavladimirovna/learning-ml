import nltk
import pandas as pd
import numpy as np
from nltk import word_tokenize, sent_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class GaussianNaiveBayes:
    def __init__(self):
        self.sigmas = dict()  # std о каждому столбцу отдельно по таргетам
        self.mus = dict()  # mu о каждому столбцу отдельно по таргетам
        self.probs = list()  # вероятность каждого таргета
        self.targets = list()

    def fit(self, X, y):
        self.__init__()
        self.targets = pd.unique(y)
        for target in self.targets:
            data_concat = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            data_target = data_concat[data_concat['target'] == target]
            p_target = data_target.shape[0] / data_concat.shape[0]
            self.probs.append(p_target)
            self.sigmas[target] = list()
            self.mus[target] = list()
            for column in data_target.drop(['target'], axis=1).columns:
                mu = data_target[column].mean()
                sigma = data_target[column].std()
                self.sigmas[target].append(sigma)
                self.mus[target].append(mu)

    def predict(self, X):
        preds = list()
        for index, row in X.iterrows():
            target_probs = dict()
            for target in self.targets:
                prob = self.probs[target]
                target_probs[target] = prob
                for column_n in range(X.shape[1]):
                    mu = self.mus[target][column_n]
                    sigma = self.sigmas[target][column_n]
                    x = X.loc[index, column_n]
                    sigma += 1e-9  # сглаживание, чтобы избежать деления на ноль
                    p = ((1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((x - mu) ** 2 / (2 * sigma ** 2))))
                    p = min(p, 1)  # ограничение на максимальное значение вероятности
                    target_probs[target] = target_probs.get(target, 1) * p
            preds.append(max(target_probs, key=target_probs.get))  # выбор таргета по наибольшей вероятности
        return preds


def outlier_std(data, col, threshold=3):
    mean = data[col].mean()
    std = data[col].std()  # сигма
    up = mean + threshold * std  # верхняя грань
    low = mean - threshold * std  # нижняя грань

    # все, что за пределами трех сигм - аномалия
    anomalies = pd.concat([data[col] > up, data[col] < low], axis=1).any(axis=1)
    return anomalies, up, low


def get_column_outliers(data, func=outlier_std, threshold=3):
    # создание столбца - является ли значение выбросом
    outliers = pd.Series(data=[False] * len(data), index=data.index, name='is_outlier')

    comparison_table = {}

    # по каждому признаку
    for column in data.columns:
        anomalies, up, low = func(data, column, threshold=threshold)
        comparison_table[column] = [up, low, sum(anomalies), 100 * sum(anomalies) / len(anomalies)]
        outliers.loc[anomalies[anomalies].index] = True

    comparison_table = pd.DataFrame(comparison_table).T
    comparison_table.columns = ['up', 'low', 'anomalies_count', 'anomalies_percentage']
    return comparison_table, outliers


def transform_text(text):
    text = text.lower()  # привели к маленьким буквам
    text = nltk.word_tokenize(text)  # разбили на токены - слова
    ps = PorterStemmer()  # стеммер - оставляет только основы

    # удаление символов
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    # удаление стоп-слов
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def main():
    # предобработка
    data = pd.read_csv('https://drive.google.com/uc?id=1gz6c3tg0cXHcsXdpyUXRkD-oqKOTYyyH', encoding='windows-1251')
    data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    encoder = LabelEncoder()
    data['target'] = encoder.fit_transform(data['target'])
    data = data.drop_duplicates()
    data['num_char'] = data['text'].apply(len)
    data['num_words'] = data['text'].apply(lambda x: len(word_tokenize(x)))
    data['num_sentences'] = data['text'].apply(lambda x: len(sent_tokenize(x)))
    data_num = data.drop(['text', 'target'], axis=1)
    data_cat = data[['text', 'target']]
    comparison_table, std_outliers = get_column_outliers(data_num)
    data_num['is_outlier'] = std_outliers
    data = pd.concat([data_num, data_cat], axis=1)
    data = data.loc[data['is_outlier'] != True]
    data = data.drop(['is_outlier'], axis=1)
    data['transformed_text'] = data['text'].apply(transform_text)
    tfidf = TfidfVectorizer(max_features=500)
    X = pd.DataFrame(tfidf.fit_transform(
        data['transformed_text']).toarray())  # строки - номера сообщений, столбцы - слова, на пересечении tfidf
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # классификация
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print(f"The accuracy score for GNB classifier {accuracy_score(y_test, y_pred)}")


if __name__ == '__main__':
    main()
