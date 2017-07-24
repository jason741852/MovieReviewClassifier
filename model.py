import nltk, re, pprint
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support as score
import random
import glob

GOOD=1
BAD=0



class ReviewModel():

    def __init__(self, dataframe=None):
        self.model = MultinomialNB()
        self.dataframe = dataframe;
        pass

    def get_data(self):
        # Read all positive and negative reviews into a single DataFrame
        list=[]
        column_names = ["review","label"]
        for filename in glob.glob("txt_sentoken/neg/*.txt"):
            with open(filename, 'r') as content_file:
                tuple = (content_file.read().replace("\n",""), BAD)
                list.append(tuple)

        for filename in glob.glob("txt_sentoken/pos/*.txt"):
            with open(filename, 'r') as content_file:
                tuple = (content_file.read().replace("\n",""), GOOD)
                list.append(tuple)

        df = pd.DataFrame(list, columns=column_names)

        return df


    def parse_and_split_data(self, df):
        X = df['review']
        self._vect = CountVectorizer(analyzer='word', stop_words='english', max_features = 3, ngram_range=(1, 1), binary=False)
        mcl_transformed = self._vect.fit_transform(X)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(mcl_transformed, y, test_size=0.5, random_state=random.randint(1, 100))

        self.X = X
        return (X_train, X_test, y_train, y_test)

    def train(self, X, y):
        self.model = self.model.fit(X,y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def report_scores(self, y_pred, y_actual):
        precision, recall, fscore, _ = score(y_actual, y_pred, labels=[0, 1])
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))



def main():
    #Instanstiate the wrapper model
    model = ReviewModel()

    df = model.get_data()

    X_train, X_test, y_train, y_test = model.parse_and_split_data(df)

    model = model.train(X_train, y_train)

    pred = model.predict(X_test)

    model.report_scores(pred, y_test)


if __name__ == '__main__':
    main()
