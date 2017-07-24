import nltk, re, pprint
import pandas as pd
from nltk import word_tokenize
import glob



class ReviewModel():

    def __init__(self, dataframe=None):
        self.dataframe = dataframe;
        pass

    def get_data(self):
        column_names = ["msg"]
        #with open('txt_sentoken/neg/cv000_29416.txt', 'r') as content_file:
        #     df = content_file.read().replace("\n","")
        #    df = pd.read_csv(content_file, names=column_names)
        #    print (df)
        i=0
        list=[]
        for filename in glob.glob("txt_sentoken/neg/*.txt"):
            frame = pd.read_csv(filename, names=column_names)
            list.append(frame)
        df = pd.concat(list)
        print (df)


        return True

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
