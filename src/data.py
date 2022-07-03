import pandas as pd
import text_process as tp
import spacy
from nltk.corpus import stopwords

def get_data():

    # read data
    train_neg = pd.read_csv('../data/train.negative.csv', quotechar=None, quoting=3, sep='\t', header=None)
    train_non = pd.read_csv('../data/train.non-negative.csv', quotechar=None, quoting=3, sep='\t', header=None)

    test_neg = pd.read_csv('../data/test.negative.csv', quotechar=None, quoting=3, sep='\t', header=None)
    test_non = pd.read_csv('../data/test.non-negative.csv', quotechar=None, quoting=3, sep='\t', header=None)

    # remove punctuation
    train_neg[0] = train_neg[0].apply(lambda x: tp.remove_punctuation(x))
    train_non[0] = train_non[0].apply(lambda x: tp.remove_punctuation(x))

    test_neg[0] = test_neg[0].apply(lambda x: tp.remove_punctuation(x))
    test_non[0] = test_non[0].apply(lambda x: tp.remove_punctuation(x))

    # lower case
    train_neg[0] = train_neg[0].apply(lambda x: x.lower())
    train_non[0] = train_non[0].apply(lambda x: x.lower())

    test_neg[0] = test_neg[0].apply(lambda x: x.lower())
    test_non[0] = test_non[0].apply(lambda x: x.lower())

    # lemmatization
    nlp = spacy.load('en_core_web_sm')

    tok_lem_sentence_train_neg = [[token.lemma_ for token in nlp(row[0].strip())] for index, row in train_neg.iterrows()]
    tok_lem_sentence_train_non = [[token.lemma_ for token in nlp(row[0].strip())] for index, row in train_non.iterrows()]

    tok_lem_sentence_test_neg = [[token.lemma_ for token in nlp(row[0].strip())] for index, row in test_neg.iterrows()]
    tok_lem_sentence_test_non = [[token.lemma_ for token in nlp(row[0].strip())] for index, row in test_non.iterrows()]

    # # remove stop words
    # stop_words = set(stopwords.words('english'))

    # rmv_sw_sentence_train_neg = [ [word for word in sentence if not word in stop_words and word != ' '] for sentence in tok_lem_sentence_train_neg]
    # rmv_sw_sentence_train_non = [ [word for word in sentence if not word in stop_words and word != ' '] for sentence in tok_lem_sentence_train_non]

    # rmv_sw_sentence_test_neg = [ [word for word in sentence if not word in stop_words and word != ' '] for sentence in tok_lem_sentence_test_neg]
    # rmv_sw_sentence_test_non = [ [word for word in sentence if not word in stop_words and word != ' '] for sentence in tok_lem_sentence_test_non]

    # sort class data
    train_list = []
    train_sentence_list = []
    train_class_list = []

    for sentence in tok_lem_sentence_train_neg:
        train_sentence_list.append(sentence)
        train_class_list.append([1])

        str_sentence = " ".join(sentence)
        train_list.append({'sentence': str_sentence, 'label': 1})

    for sentence in tok_lem_sentence_train_non:
        train_sentence_list.append(sentence)
        train_class_list.append([0])

        str_sentence = " ".join(sentence)
        train_list.append({'sentence': str_sentence, 'label': 0})

    test_list = []
    test_sentence_list = []
    test_class_list = []

    for sentence in tok_lem_sentence_test_neg:
        test_sentence_list.append(sentence)
        test_class_list.append([1])

        str_sentence = " ".join(sentence)
        test_list.append({'sentence': str_sentence, 'label': 1})

    for sentence in tok_lem_sentence_test_non:
        test_sentence_list.append(sentence)
        test_class_list.append([0])

        str_sentence = " ".join(sentence)
        test_list.append({'sentence': str_sentence, 'label': 0})

    train_df = pd.DataFrame(train_list)
    test_df = pd.DataFrame(test_list)

    return train_df, test_df, train_sentence_list, train_class_list, test_sentence_list, test_class_list