import pandas as pd

def get_data():
    train_neg = pd.read_csv('../data/train.negative.csv', quotechar=None, quoting=3, sep='\t', header=None)
    train_non = pd.read_csv('../data/train.non-negative.csv', quotechar=None, quoting=3, sep='\t', header=None)

    test_neg = pd.read_csv('../data/test.negative.csv', quotechar=None, quoting=3, sep='\t', header=None)
    test_non = pd.read_csv('../data/test.non-negative.csv', quotechar=None, quoting=3, sep='\t', header=None)

    print(type(train_neg))