import os
import pandas as pd
from datasets.mental_healthy import MentalHealthyDataset, MentalHealthyDataset_test
from classifiers.sklearn_adapter import SklearnClassifierAdapter
from sklearn.metrics import precision_score, recall_score, f1_score
# 引入需要的分類器
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# 定義分類器清單，使用類別而非實例
# classifiers = {
#     "LogisticRegression": LogisticRegression,
#     "SVM": SVC,
#     "NaiveBayes": GaussianNB,
#     "DecisionTree": DecisionTreeClassifier,
#     "KNeighbors": KNeighborsClassifier,
#     "RandomForest": RandomForestClassifier,
#     "GradientBoostingClassifier":GradientBoostingClassifier,
#     "AdaBoostClassifier": AdaBoostClassifier,
#     "MLPClassifier": MLPClassifier
# }

# 定義不同的目標變數清單
target_lists = {
    # "targets_export": [
    #     "Profession",
    #     "Pressure",
    #     "Satisfaction",
    #     "Sleep Duration",
    #     "Have you ever had suicidal thoughts ?",
    #     "Work/Study Hours",
    #     "Financial Stress",
    #     "Family History of Mental Illness"
    # ],
    # "targets_chi2": [
    #     "Age",
    #     "Profession",
    #     "Dietary Habits",
    #     "Degree",
    #     "Have you ever had suicidal thoughts ?",
    #     "Work/Study Hours",
    #     "Financial Stress",
    #     "Pressure"
    # ],
    "targets_myself": [
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
}

targets_and_classifiers = {
    # "targets_export": {
    #     "RandomForest": (RandomForestClassifier, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}),
    #     "GradientBoosting": (GradientBoostingClassifier, {'learning_rate': 0.25, 'max_depth': 2, 'min_samples_split': 2, 'n_estimators': 350, 'subsample': 0.8}),
    #     "AdaBoost": (AdaBoostClassifier, {'estimator': DecisionTreeClassifier(max_depth=5), 'learning_rate': 0.1, 'n_estimators': 250}),
    #     # "MLPClassifier": (MLPClassifier, {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 100), 'learning_rate_init': 0.005, 'max_iter': 600, 'solver': 'adam'})
    # },
    # "targets_chi2": {
    #     "RandomForest": (RandomForestClassifier, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}),
    #     "GradientBoosting": (GradientBoostingClassifier, {'learning_rate': 0.25, 'max_depth': 2, 'min_samples_split': 2, 'n_estimators': 350, 'subsample': 0.75}),
    #     "AdaBoost": (AdaBoostClassifier, {'estimator': DecisionTreeClassifier(max_depth=5), 'learning_rate': 0.1, 'n_estimators': 200}),
    #     # "MLPClassifier": (MLPClassifier, {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 100), 'learning_rate_init': 0.005, 'max_iter': 600, 'solver': 'adam'})
    # },
    "targets_myself": {
        "RandomForest": (RandomForestClassifier, {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}),
        "GradientBoosting": (GradientBoostingClassifier, {'learning_rate': 0.25, 'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 250, 'subsample': 0.75}),
        "AdaBoost": (AdaBoostClassifier, {'estimator': DecisionTreeClassifier(max_depth=4), 'learning_rate': 0.15, 'n_estimators': 250}),
        # "MLPClassifier": (MLPClassifier,{'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (100, 100), 'learning_rate_init': 0.005, 'max_iter': 600, 'solver': 'adam'})
    }
}
# 建立輸出目錄
output_dir = "predictions_best_ture"
os.makedirs(output_dir, exist_ok=True)

# 設定起始 ID
start_id = 140700

# 遍歷每個目標變數清單
for target_name, classifiers in targets_and_classifiers.items():
    print(f"處理目標變數清單: {target_name}")
    targets = target_lists[target_name]
    # 建立訓練和測試資料集
    # train = MentalHealthyDataset_test(targets=targets, data_type="train")
    train = MentalHealthyDataset_test(targets=targets, data_type="train")
    test = MentalHealthyDataset_test(targets=targets, data_type="test")
    print(f"  訓練集大小: {len(train)}, 測試集大小: {len(test)}")
    X_train, y_train = train.feature, train.label
    X_test = test.feature
    print(type(train.feature), train.feature.shape)
    print(type(train.label), train.label.shape)
    print(type(test.feature), test.feature.shape)
    # 遍歷每個分類器
    for clf_name, (clf_class, clf_params) in classifiers.items():
        adapter = SklearnClassifierAdapter(base_classifier=clf_class, params=clf_params)
        print(f"  Training {clf_name} for {target_name} with params {clf_params}")
        
        try:
            # 訓練模型
            adapter.fit(X_train, y_train)
            
            # 預測
            predictions = adapter.predict(X_test)
            
            # 評估模型性能（例如準確率）
            y_pred = adapter.predict(X_train)
            accuracy = adapter.score(X_train, y_train)
            precision = precision_score(y_train, y_pred)
            recall = recall_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred)
            print(f"    {clf_name} 在 {target_name} 上的訓練準確率: {accuracy:.4f}\nprecision:{precision:.4f}\nrecall:{recall:.4f}\nf1:{f1:.4f}")
            
            # 將預測結果轉換為 DataFrame
            df = pd.DataFrame(predictions, columns=["Depression"])
            
            # 生成 ID 欄位
            df["id"] = range(start_id, start_id + len(df))
            
            # 調整欄位順序，將 'id' 放在最前面
            df = df[["id","Depression"]]
            
            # 定義檔案名稱，將空格和特殊字符替換為底線以避免檔案名稱問題
            safe_target_list = target_name.replace(" ", "_")
            filename = f"test_predictions_{safe_target_list}_{clf_name}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # 儲存為 CSV 檔案
            df.to_csv(filepath, index=False)
            print(f"    預測結果已儲存至: {filepath}")
        
        except Exception as e:
            print(f"    處理分類器 {clf_name} 在目標變數清單 {target_name} 時發生錯誤: {e}")

print("所有目標變數清單、分類器的預測已完成並儲存。")
