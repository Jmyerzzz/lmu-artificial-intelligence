'''
Sam Chami
Jackson Myers
John Scott
Ben Smith

naive_bayes.py
'''

import unittest
import warnings
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.impute import SimpleImputer


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier that preprocesses existing data to train an agent such that the
    agent can learn how to accurately predict a feature: specifically, we are determining
    whether or not an individual earns more or less than 50,000 USD per year when we know
    some related features about those persons.
    """

    def preprocess(file_name):
        """
        Preprocesses the data for the NBC so that it is properly formatted for our training model.
        """
        names = ("age", "wrk_cls", "edu", "edu_num", "marital_sts",
                 "occu_code", "relation", "race", "sex", "gain",
                 "loss", "hours", "country", "income")
        original_data = pd.read_csv(file_name, dtype=object, names=names, skipinitialspace=True)
        original_data = pd.DataFrame(original_data)

        imp = SimpleImputer(missing_values='?', strategy='most_frequent', verbose=2)
        data = imp.fit_transform(original_data)

        enc = preprocessing.OrdinalEncoder()
        data = enc.fit_transform(data)

        est = preprocessing.KBinsDiscretizer(n_bins=7, encode="ordinal")
        data = est.fit_transform(data)
        return data

    def train_model(data):
        """ Trains agent on preprocessed data """
        classifier = MultinomialNB()
        X, Y = data[:, :-1], data[:, -1]
        classifier.fit(X, Y)
        return classifier


class NaiveBayesClassifierTests(unittest.TestCase):
    """ Tests to determine the efficacy of our agent """
    def setUp(self):
        """ Suppresses annoying warning """
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)

    def test(self):
        """
        Train agent using income-training data and test accuracy of agent on income-test data.
        """
        processed_training_data = NaiveBayesClassifier.preprocess("income-training.csv")
        processed_test_data = NaiveBayesClassifier.preprocess("income-test.csv")
        trained_model = NaiveBayesClassifier.train_model(processed_training_data)
        accuracy = trained_model.score(processed_test_data[:, :-1], processed_test_data[:, -1])
        print("Agent Prediction Accuracy: ", round(accuracy * 100, 2), "%")

if __name__ == "__main__":
    unittest.main()
