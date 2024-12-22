from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from tqdm import tqdm
import pandas as pd
from datasets.mental_healthy import MentalHealthyDataset_test

# Define columns and load data
columns = [
    "Gender", "Age", "City", "Profession", "Sleep Duration", "Dietary Habits",
    "Degree", "Have you ever had suicidal thoughts ?", "Work/Study Hours",
    "Financial Stress", "Family History of Mental Illness", "Pressure",
    "Satisfaction"
]
data = MentalHealthyDataset_test(columns)
X = data.feature
y = data.label

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define estimator with parallel processing
estimator = RandomForestClassifier(random_state=42, n_jobs=-1)

# Initialize RFECV with optimized parameters
selector = RFECV(estimator=estimator, step=1, cv=cv, scoring='f1', n_jobs=-1)

# Fit RFECV
selector.fit(X, y)

# Output optimal number of features
print(f"Optimal number of features: {selector.n_features_}")

# Optional: Display selected features
selected_features = X.columns[selector.support_]
print("Selected features:", selected_features)
# X = data.feature
# y = data.label
# X.replace(-1, 3, inplace=True)

# # 使用 Chi-Square 進行特徵選擇
# selector = SelectKBest(score_func=chi2, k="all")  # 選擇前 10 個特徵
# selector.fit(X,y)
# scores = selector.scores_
# columns = X.columns
# feature_scores = pd.DataFrame({"Feature": columns, "Score": scores})
# print(feature_scores.sort_values(by="Score", ascending=False))

# Optimal number of features: 13 3, accuracy
# Selected features: Index(['Gender', 'Age', 'City', 'Profession', 'Sleep Duration',
#        'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
#        'Work/Study Hours', 'Financial Stress',
#        'Family History of Mental Illness', 'Pressure', 'Satisfaction'],
#       dtype='object')