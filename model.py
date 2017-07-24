import nltk, re, pprint
import pandas as pd
from nltk import word_tokenize
import glob



class ReviewModel():

    def __init__(self, dataframe=None):
        self.dataframe = dataframe;
        pass

    def get_data(self):
        # Read all positive and negative reviews into a single DataFrame
        list=[]
        column_names = ["review","score"]
        for filename in glob.glob("txt_sentoken/neg/*.txt"):
            with open(filename, 'r') as content_file:
                tuple = (content_file.read().replace("\n",""), False)
                list.append(tuple)

        for filename in glob.glob("txt_sentoken/pos/*.txt"):
            with open(filename, 'r') as content_file:
                tuple = (content_file.read().replace("\n",""), True)
                list.append(tuple)

        df = pd.DataFrame(list, columns=column_names)

        return df

    def parse_and_split_data(self, df):
        pass

    def engineer_features(self, df):
        pass

    def tokenizer(self):
        file_content = open

    def train(self, x, y):
        pass

    def predict(self, x):
        pass






def main():
    #Instanstiate the wrapper model
    model = ReviewModel()

    df = model.get_data()


if __name__ == '__main__':
    main()
