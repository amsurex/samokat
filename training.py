import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

file1_path = 'val_10k.csv'
val_data = pd.read_csv(file1_path)
file3_path = 'train_40k.csv'
train_data = pd.read_csv(file3_path)


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.cat1_clf = None
        self.cat2_clfs = {}
        self.cat3_clfs = {}

    def fit(self, x, y):
        assert 'Cat1' in y.columns
        assert 'Cat2' in y.columns
        assert 'Cat3' in y.columns

        cat1_data = x
        cat1_labels = y['Cat1']
        self.cat1_clf = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', self.base_classifier())
        ])
        self.cat1_clf.fit(cat1_data, cat1_labels)

        for cat1_class in self.cat1_clf.classes_:
            cat2_data = x[y['Cat1'] == cat1_class]
            cat2_labels = y[y['Cat1'] == cat1_class]['Cat2']
            if len(cat2_labels) > 0:
                clf = Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', self.base_classifier())
                ])
                clf.fit(cat2_data, cat2_labels)
                self.cat2_clfs[cat1_class] = clf

        for cat1_class, cat2_clf in self.cat2_clfs.items():
            for cat2_class in cat2_clf.classes_:
                cat3_data = x[(y['Cat1'] == cat1_class) & (y['Cat2'] == cat2_class)]
                cat3_labels = y[(y['Cat1'] == cat1_class) & (y['Cat2'] == cat2_class)]['Cat3']
                if len(cat3_labels) > 0:
                    clf = Pipeline([
                        ('tfidf', TfidfVectorizer()),
                        ('clf', self.base_classifier())
                    ])
                    clf.fit(cat3_data, cat3_labels)
                    self.cat3_clfs[(cat1_class, cat2_class)] = clf

        return self

    def predict(self, x):
        if not isinstance(x, pd.Series):
            x = pd.Series(x)

        cat1_preds = self.cat1_clf.predict(x)
        cat2_preds = []
        cat3_preds = []

        for i, cat1_pred in enumerate(cat1_preds):
            if cat1_pred in self.cat2_clfs:
                cat2_pred = self.cat2_clfs[cat1_pred].predict([x[i]])[0]
            else:
                cat2_pred = None
            cat2_preds.append(cat2_pred)

            if cat2_pred and (cat1_pred, cat2_pred) in self.cat3_clfs:
                cat3_pred = self.cat3_clfs[(cat1_pred, cat2_pred)].predict([x[i]])[0]
            else:
                cat3_pred = None
            cat3_preds.append(cat3_pred)

        return cat1_preds, cat2_preds, cat3_preds


x_train_lcpn = train_data['Text']
y_train_lcpn = train_data[['Cat1', 'Cat2', 'Cat3']]
x_val_lcpn = val_data['Text']
y_val_lcpn = val_data[['Cat1', 'Cat2', 'Cat3']]

hierarchical_clf = HierarchicalClassifier(base_classifier=RandomForestClassifier)
hierarchical_clf.fit(x_train_lcpn, y_train_lcpn)

y_pred_cat1, y_pred_cat2, y_pred_cat3 = hierarchical_clf.predict(x_val_lcpn)

# Сохранение модели с использованием joblib
with open('hierarchical_clf.pkl', 'wb') as f:
    joblib.dump(hierarchical_clf, f)
