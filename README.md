pip install -r requirements.txt

mental_healthy.py: for data preprocessing<br>
<tab>MentalHealthyDataset: strategy dropna<br>
<tab>MentalHealthyDataset_test: strategy fillna<br>

sklearn_adapter.py: self define class for sklearn classifier

feature.py: for feature selection<br>
<tab>Chi-Square: select top 8 accroding to scores from 13 features<br>
<tab>RFECV: select all features from 13 features<br>

hyperparameter.py: for tuning the hyperparameter of classifier<br>
<tab>RandomizedSearchCV: for large range of parameters or time limit, search fast, roughly<br>
<tab>GridSearchCV: for small range of parameters, precisely<br>

main.py: for running the classifiers and generate the predict results<br>
<tab>the recent one is using MentalHealthyDataset_test and it will generate diretory "predicton_best" for the predict results which using the best parameter got from the hyperparameter.py<br>