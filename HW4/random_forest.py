'''
Sam Chami
Jackson Myers
John Scott
Ben Smith

random_forest.py
'''

import unittest
import warnings
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

class RandomForest:

    def preprocess(file_name):
        """
        Preprocesses the data for the Random Forest so that it is properly formatted for
        our training model.
        """

        names = ["age", "wrk_cls", "edu", "edu_num", "marital_sts",
                 "occu_code", "relation", "race", "sex", "gain",
                 "loss", "hours", "country", "income"]

        original_data = pd.read_csv(file_name, dtype=object, names=names, skipinitialspace=True)
        original_data = pd.DataFrame(original_data)

        imp = SimpleImputer(missing_values='?', strategy='most_frequent', verbose=2)
        data = imp.fit_transform(original_data)

        enc = preprocessing.OrdinalEncoder()
        data = enc.fit_transform(data)

        est = preprocessing.KBinsDiscretizer(n_bins=8, encode="ordinal")
        data = est.fit_transform(data)
        return data

    def train_model(data):
        """
        Trains agent on preprocessed data.
        """
        X, Y = data[:, :-1], data[:, -1]

        clf = RandomForestClassifier(n_estimators=45, max_depth=10, random_state=0)
        clf.fit(X, Y)
        return clf

class RandomForestTests(unittest.TestCase):
    """Unit Tests"""
    def setUp(self):
        """
        Suppresses annoying warning
        """
        warnings.simplefilter('ignore', category=ImportWarning)

    def test(self):
        """
        Train agent using income-training data and test accuracy of agent on income-test data.
        """
        processed_training_data = RandomForest.preprocess("income-training.csv")
        processed_test_data = RandomForest.preprocess("income-test.csv")

        trained_model = RandomForest.train_model(processed_training_data)
        accuracy = trained_model.score(processed_test_data[:, :-1], processed_test_data[:, -1])
        print("Agent Prediction Accuracy: ", round(accuracy * 100, 2), "%")

if __name__ == "__main__":
    unittest.main()
