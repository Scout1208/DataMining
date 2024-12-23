import pandas as pd
from datasets.mental_healthy import MentalHealthyDataset, MentalHealthyDataset_test

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
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
models_and_params = {
    # "RandomForestClassifier": {
    #     "model": RandomForestClassifier(random_state=42),
    #     "random_params": {
    #         "n_estimators": [50, 100, 200, 300, 500],
    #         "max_depth": [3, 5, 10, 20, None],
    #         "min_samples_split": [2, 5, 10],
    #         "min_samples_leaf": [1, 2, 5],
    #         "max_features": ["sqrt", "log2", None]
    #     },
    #     "grid_params": {
    #         "n_estimators": [100, 200],
    #         "max_depth": [10, 20, None],
    #         "min_samples_split": [2, 5],
    #         "min_samples_leaf": [1, 2],
    #         "max_features": ["sqrt", "log2"]
    #     }
    # },
    # "GradientBoostingClassifier": {
    #     "model": GradientBoostingClassifier(random_state=42),
    #     "random_params": {
    #         "n_estimators": [50, 100, 200, 300],
    #         "learning_rate": [0.01, 0.05, 0.1, 0.2],
    #         "max_depth": [3, 5, 10],
    #         "min_samples_split": [2, 5, 10],
    #         "subsample": [0.6, 0.8, 1.0]
    #     },
    #     "grid_params": {
    #         'n_estimators': [250, 300, 350],
    #         'learning_rate': [0.15, 0.2, 0.25],
    #         'max_depth': [2, 3, 4],
    #         'min_samples_split': [2, 3],
    #         'subsample': [0.75, 0.8, 0.85]
    #     }
    # },
    # "AdaBoostClassifier": {
    #     "model": AdaBoostClassifier(random_state=42),
    #     "random_params": {
    #         "n_estimators": [50, 100, 200, 300],
    #         "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
    #         "estimator": [
    #             DecisionTreeClassifier(max_depth=1),
    #             DecisionTreeClassifier(max_depth=3),
    #             DecisionTreeClassifier(max_depth=5)
    #         ]
    #     },
    #     "grid_params": {
    #         'n_estimators': [150, 200, 250],
    #         'learning_rate': [0.05, 0.1, 0.15],
    #         'estimator': [DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=5)]
    #     }
    # },
    "MLPClassifier":{
        "model": MLPClassifier(random_state=42),
        "random_params":{
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["adam", "sgd"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "max_iter": [200, 400, 800]
        },
        "grid_params": {
            'hidden_layer_sizes': [(100, 50), (100, 100)],
            'learning_rate_init': [0.005],
            'activation': ['logistic', 'relu'],
            'alpha': [0.0001, 0.001],
            'solver': ['adam'],
            'max_iter': [600, 800, 1000]
        }

    }
}
# 遍歷每個目標變數清單
for target_list_name, targets in target_lists.items():
    print(f"處理目標變數清單: {target_list_name}")
    
    # 建立訓練和測試資料集
    # train = MentalHealthyDataset_test(targets=targets, data_type="train")
    train = MentalHealthyDataset_test(targets=targets, data_type="train")
    print(f"  訓練集大小: {len(train)}")
    X_train, y_train = train.feature, train.label
    print(type(train.feature), train.feature.shape)
    print(type(train.label), train.label.shape)
    # 遍歷每個分類器
    results = []
    for model_name, config in models_and_params.items():
        print(f"Running hyperparameter search for {model_name}...")
        
        model = config["model"]
        # random_params = config["random_params"]
        grid_params = config["grid_params"]

        # RandomizedSearchCV
        # random_search = RandomizedSearchCV(
        #     estimator=model,
        #     param_distributions=random_params,
        #     n_iter=20,  # 随机搜索迭代次数
        #     scoring="accuracy",
        #     cv=5,  # 交叉验证折数
        #     verbose=1,
        #     random_state=42,
        #     n_jobs=-1
        # )
        # random_search.fit(X_train, y_train)
        # print(f"{model_name} RandomizedSearchCV Best Params: {random_search.best_params_}")
        # print(f"{model_name} RandomizedSearchCV Best Score: {random_search.best_score_}")
        # results.append({
        #     "model": model_name,
        #     "method": "RandomizedSearchCV",
        #     "best_params": random_search.best_params_,
        #     "best_score": random_search.best_score_
        # })

        GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            scoring="accuracy",
            cv=5,  # 交叉验证折数
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        print(f"{model_name} GridSearchCV Best Params: {grid_search.best_params_}")
        print(f"{model_name} GridSearchCV Best Score: {grid_search.best_score_}")
        results.append({
            "model": model_name,
            "method": "GridSearchCV",
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_
        })

    for result in results:
        print(result)

