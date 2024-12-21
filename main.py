import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("./train.csv")
print(df.shape)
#drop 無用的feature
df = df.drop(columns=["Name","id","CGPA"])
print(df.shape)
#合併pressure and Satisfaction
df["Pressure"] = df["Academic Pressure"].fillna(df["Work Pressure"])
df["Satisfaction"] = df['Study Satisfaction'].combine_first(df['Job Satisfaction'])
df["Profession"] = df["Profession"].fillna(df["Working Professional or Student"])
df = df.drop(columns=["Academic Pressure","Work Pressure","Study Satisfaction","Job Satisfaction","Working Professional or Student"])
#確認missing value
print(df["Pressure"].isna().sum())
print(df["Satisfaction"].isna().sum())
#drop 缺失值
df.dropna(subset=["Pressure", "Satisfaction"],inplace=True)
print(df.shape)

#處理category feature
print(df["Gender"].isna().sum())
df["Gender"] = df["Gender"].map({'Male': 0, 'Female': 1})
print(df["Gender"])

# df["Working Professional or Student"] = df["Working Professional or Student"].map({"Working Professional": 0, "Student": 1})
# print(df["Working Professional or Student"])

print(df["Dietary Habits"].isna().sum())
df.dropna(subset=["Dietary Habits"], inplace=True)
print(df.shape)
df["Dietary Habits"] = df["Dietary Habits"].map({"Unhealthy": 0, "Moderate": 1, "Healthy": 2})
print(df["Dietary Habits"])

print(df["Have you ever had suicidal thoughts ?"].isna().sum())
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"No": 0, "Yes": 1})
print(df["Have you ever had suicidal thoughts ?"])

print(df["Family History of Mental Illness"].isna().sum())
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"No": 0, "Yes": 1})
print(df["Family History of Mental Illness"])

#use Label encoder
label_encoder = LabelEncoder()
df = df[df['City'] != "3.0"]
print(df.shape)
# df['City'] = label_encoder.fit_transform(df['City'])
# print(label_encoder.classes_)

# df["Profession"] = label_encoder.fit_transform(df["Profession"])
# print(label_encoder.classes_)

# df["Sleep Duration"] = label_encoder.fit_transform(df["Sleep Duration"])
# print(label_encoder.classes_)

# df["Degree"] = label_encoder.fit_transform(df["Degree"])
# print(label_encoder.classes_)

print(df["Profession"])
# #處理異常值
valid_city_values = [
    'Agra',
    'Ahmedabad',
    'Bangalore',
    'Bhopal',
    'Chennai',
    'Delhi',
    'Faridabad',
    'Ghaziabad',
    'Gurgaon',
    'Hyderabad',
    'Indore',
    'Jaipur',
    'Kalyan',
    'Kanpur',
    'Khaziabad',
    'Kolkata',
    'Lucknow',
    'Ludhiana',
    'Meerut',
    'Morena',
    'Mumbai',
    'Nagpur',
    'Nashik',
    'Patna',
    'Pune',
    'Rajkot',
    'Srinagar',
    'Surat',
    'Thane',
    'Vadodara',
    'Varanasi',
    'Vasai-Virar',
    'Visakhapatnam'
]


valid_profession_values = [
    'Academic',
    'Accountant',
    'Analyst',
    'Architect',
    'Business Analyst',
    'Chef',
    'Chemist',
    'City Manager',
    'Civil Engineer',
    'Consultant',
    'Content Writer',
    'Customer Support',
    'Data Scientist',
    'Dev',
    'Digital Marketer',
    'Doctor',
    'Educational Consultant',
    'Electrician',
    'Entrepreneur',
    'Family Consultant',
    'Financial Analyst',
    'Graphic Designer',
    'HR Manager',
    'Investment Banker',
    'Judge',
    'Lawyer',
    'Manager',
    'Marketing Manager',
    'Mechanical Engineer',
    'Medical Doctor',
    'Pharmacist',
    'Pilot',
    'Plumber',
    'Research Analyst',
    'Researcher',
    'Sales Executive',
    'Software Engineer',
    'Teacher',
    'Travel Consultant',
    'UX/UI Designer',
    'Student'
]

# 6-7只有4筆
valid_sleep_duration_values = [
    'Less than 5 hours', '5-6 hours', '7-8 hours', 
    'More than 8 hours'
]

valid_degree_values = [
    'ACA',
    'B B.Com',
    'B BA',
    'B.Arch',
    'B.B.Arch',
    'B.Com',
    'B.Ed',
    'B.Pharm',
    'B.Sc',
    'B.Tech',
    'BA',
    'BArch',
    'BBA',
    'BCA',
    'BE',
    'BEd',
    'BHM',
    'BPA',
    'BPharm',
    'BSc',
    'E.Tech',
    'K.Ed',
    'L.Ed',
    'LHM',
    'LL B.Ed',
    'LL.Com',
    'LLB',
    'LLBA',
    'LLCom',
    'LLEd',
    'LLM',
    'LLS',
    'M.Arch',
    'M.Com',
    'M.Ed',
    'M.Pharm',
    'M.S',
    'M.Tech',
    'MA',
    'MBA',
    'MBBS',
    'MCA',
    'MD',
    'ME',
    'MEd',
    'MHM',
    'MPA',
    'MPharm',
    'MSc',
    'MTech',
    'M_Tech',
    'N.Pharm',
    'P.Com',
    'P.Pharm',
    'PhD',
    'S.Arch',
    'S.Pharm',
    'S.Tech'
]

df = df[df['City'].isin(valid_city_values)]
print(df.shape)
df = df[df['Profession'].isin(valid_profession_values)]
print(df.shape)
df = df[df['Sleep Duration'].isin(valid_sleep_duration_values)]
print(df.shape)
df = df[df['Degree'].isin(valid_degree_values)]
print(df.shape)

df = df[df['Work/Study Hours'] != 0]
print(df.shape)
#移除極少值
cityCnt = df["City"].value_counts()
city = cityCnt[cityCnt >= 10].index
df = df[df['City'].isin(city)]
print(df.shape)
jobCnt = df["Profession"].value_counts()
job = jobCnt[jobCnt >= 10].index
df = df[df['Profession'].isin(job)]
print(df.shape)
df['City'] = label_encoder.fit_transform(df['City'])
print(label_encoder.classes_)

df["Profession"] = label_encoder.fit_transform(df["Profession"])
print(label_encoder.classes_)

df["Sleep Duration"] = df["Sleep Duration"].map({"Less than 5 hours": 0, "5-6 hours":1, "7-8 hours": 2, "More than 8 hours":3})

df["Degree"] = label_encoder.fit_transform(df["Degree"])
print(label_encoder.classes_)

#age bining
df['Age'] = pd.cut(df['Age'], bins=[18, 45, 60], labels=[0, 1])
df.dropna(subset=["Age"],inplace=True)
print(df.shape)

from sklearn.feature_selection import SelectKBest, chi2

# 假設特徵列和目標列分別如下：
features = df.drop(columns=['Depression'])
print(features)
label = df['Depression']

selector = SelectKBest(score_func=chi2, k="all")
selector.fit(features.fillna(0), label)

# 查看每個特徵的分數
scores = selector.scores_
columns = features.columns
feature_scores = pd.DataFrame({"Feature": columns, "Score": scores})
print(feature_scores.sort_values(by="Score", ascending=False))
# 選擇 K 個最佳特徵
selector = SelectKBest(score_func=chi2, k=8)
selected_features = selector.fit_transform(features.fillna(0), label)

# 選出的特徵
targets_selected = list(features.columns[selector.get_support()])
# targets_myself = ["gender","age","city"]
targets_export = ["Profession", "Pressure", "Satisfaction", "Sleep Duration",
                   "Have you ever had suicidal thoughts ?", "Work/Study Hours",
                   "Financial Stress", "Family History of Mental Illness"]
print("Selected Features:", targets_selected)
df.to_csv('output.csv')