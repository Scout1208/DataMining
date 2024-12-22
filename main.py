import os
import pandas as pd
from datasets.mental_healthy import MentalHealthyDataset, MentalHealthyDataset_test
from classifiers.sklearn_adapter import SklearnClassifierAdapter

# 引入需要的分類器
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# 定義分類器清單，使用類別而非實例
classifiers = {
    # "LogisticRegression": LogisticRegression,
    # "SVM": SVC,
    # "NaiveBayes": GaussianNB,
    # "DecisionTree": DecisionTreeClassifier,
    # "KNeighbors": KNeighborsClassifier,
    "RandomForest": RandomForestClassifier,
    "GradientBoostingClassifier":GradientBoostingClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    # "MLPClassifier": MLPClassifier
}

# 定義不同的目標變數清單
target_lists = {
    "targets_export": [
        "Profession",
        "Pressure",
        "Satisfaction",
        "Sleep Duration",
        "Have you ever had suicidal thoughts ?",
        "Work/Study Hours",
        "Financial Stress",
        "Family History of Mental Illness"
    ],
    "targets_chi2": [
        "Age",
        "Profession",
        "Dietary Habits",
        "Degree",
        "Have you ever had suicidal thoughts ?",
        "Work/Study Hours",
        "Financial Stress",
        "Pressure"
    ],
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

# 建立輸出目錄
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

# 設定起始 ID
start_id = 140700

# 遍歷每個目標變數清單
for target_list_name, targets in target_lists.items():
    print(f"處理目標變數清單: {target_list_name}")
    
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
    for clf_name, clf_class in classifiers.items():
        adapter = SklearnClassifierAdapter(base_classifier=clf_class)
        print(f"  開始處理分類器: {clf_name}")
        
        try:
            # 訓練模型
            adapter.fit(X_train, y_train)
            
            # 預測
            predictions = adapter.predict(X_test)
            
            # 評估模型性能（例如準確率）
            accuracy = adapter.score(X_train, y_train)
            print(f"    {clf_name} 在 {target_list_name} 上的訓練準確率: {accuracy:.4f}")
            
            # 將預測結果轉換為 DataFrame
            df = pd.DataFrame(predictions, columns=["Depression"])
            
            # 生成 ID 欄位
            df["id"] = range(start_id, start_id + len(df))
            
            # 調整欄位順序，將 'id' 放在最前面
            df = df[["id","Depression"]]
            
            # 定義檔案名稱，將空格和特殊字符替換為底線以避免檔案名稱問題
            safe_target_list = target_list_name.replace(" ", "_")
            filename = f"test_predictions_{safe_target_list}_{clf_name}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # 儲存為 CSV 檔案
            df.to_csv(filepath, index=False)
            print(f"    預測結果已儲存至: {filepath}")
        
        except Exception as e:
            print(f"    處理分類器 {clf_name} 在目標變數清單 {target_list_name} 時發生錯誤: {e}")

print("所有目標變數清單、分類器的預測已完成並儲存。")
