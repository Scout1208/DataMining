pip install -r requirements.txt

### mental_healthy.py: for data preprocessing

- **MentalHealthyDataset**: strategy `dropna`
- **MentalHealthyDataset_test**: strategy `fillna`

### sklearn_adapter.py: self define class for sklearn classifier

### feature.py: for feature selection

- **Chi-Square**: select top 8 according to scores from 13 features
- **RFECV**: select all features from 13 features

### hyperparameter.py: for tuning the hyperparameter of classifier

- **RandomizedSearchCV**: for large range of parameters or time limit, search fast, roughly
- **GridSearchCV**: for small range of parameters, precisely

### main.py: for running the classifiers and generate the predict results

- The recent one is using `MentalHealthyDataset_test`, and it will generate the directory `prediction_best` for the predict results using the best parameter from `hyperparameter.py`.