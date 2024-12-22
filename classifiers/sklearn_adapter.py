from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

class SklearnClassifierAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=None, params = None):
        super().__init__()
        if base_classifier is None:
            self.base_classifier = RandomForestClassifier()
        else:
            self.base_classifier = base_classifier(**(params or {}))
    
    def fit(self, X, y):
        self.base_classifier.fit(X, y)
        return self
    
    def predict(self, X):
        return self.base_classifier.predict(X)
    
    def score(self, X, y):
        return self.base_classifier.score(X, y)
