import nltk
import pandas as pd
from nltk import word_tokenize, sent_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class MultinomialNaiveBayes:
    def __init__(self):
        self.word_freq = dict()
        self.n = 0
        self.probs = dict()  # вероятность каждого таргета
        self.targets = list()

    def fit(self, X, y):
        self.__init__()
        self.n = X.shape[1]
        self.targets = pd.unique(y)
        for target in self.targets:
            data_concat = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            data_target = data_concat[data_concat['target'] == target]
            p_target = data_target.shape[0] / data_concat.shape[0]
            self.probs[target] = p_target
            self.word_freq[target] = list()
            for column in data_target.drop(['target'], axis=1).columns:
                word_freq = (data_target[column].sum() + 1) / (data_target.values.sum() + data_target.shape[1])
                self.word_freq[target].append(word_freq)

    def predict(self, X):
        preds = list()
        for index, row in X.iterrows():
            target_probs = dict()
            for target in self.targets:
                prob = self.probs[target]
                target_probs[target] = prob
                for column_n in range(X.shape[1]):
                    word = X.loc[index, column_n]
                    p = self.word_freq[target][column_n] ** word
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
    vectorizer = CountVectorizer(max_features=1000)
    X = pd.DataFrame(vectorizer.fit_transform(
        data['transformed_text']).toarray())  # строки - номера сообщений, столбцы - слова, на пересечении количество вхождений
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # классификация
    mnb = MultinomialNaiveBayes()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    print(f"The accuracy score for MNB classifier {accuracy_score(y_test, y_pred)}")


if __name__ == '__main__':
    main()
