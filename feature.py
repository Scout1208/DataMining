from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, chi2
from datasets.mental_healthy import MentalHealthyDataset_test
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
class RFECVWithProgress(RFECV):
    def fit(self, X, y):
        self._n_features = X.shape[1]
        
        # 包裝進度條
        with tqdm(total=self._n_features, desc="Feature Selection Steps", leave=True) as pbar:
            original_support_ = None
            while self._n_features > self.min_features_to_select:
                # 每次減少一步，並更新進度條
                super().fit(X, y)
                support_ = self.support_
                self._n_features = support_.sum()

                # 進度條更新
                if original_support_ is None:
                    pbar.update(self._n_features)
                else:
                    pbar.update((original_support_ != support_).sum())
                original_support_ = support_

        return self
columns = [
    "Gender",
    "Age",
    "City",
    "Profession",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "Pressure",
    "Satisfaction"
]
data = MentalHealthyDataset_test(columns)
X = data.feature
y = data.label
X.replace(-1, 3, inplace=True)

# 使用 Chi-Square 進行特徵選擇
selector = SelectKBest(score_func=chi2, k="all")  # 選擇前 10 個特徵
selector.fit(X,y)
scores = selector.scores_
columns = X.columns
feature_scores = pd.DataFrame({"Feature": columns, "Score": scores})
print(feature_scores.sort_values(by="Score", ascending=False))


# 缺失或無法分的就設為 -1
# 不再做 dropna(subset=["Age"])，確保 row 數不變。
X = data.feature
y = data.label

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
estimator = RandomForestClassifier(random_state=42)

# 使用 RFECVWithProgress
selector = RFECVWithProgress(estimator=estimator, step=1, cv=cv, scoring='accuracy')
selector.fit(X, y)

print(f"Optimal number of features: {selector.n_features_}")
