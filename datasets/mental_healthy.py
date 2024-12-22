from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
class MentalHealthyDataset(Dataset):
    dataset = None
    def __init__(self, targets, data_type):
        self.feature ,self.label= MentalHealthyDataset.get_X_y(targets, data_type)
        self.type = data_type
    def __getitem__(self, item):
        input = {}
        for k in self.feature.keys():
            input[k] = self.feature[k][item]
        
        return input, self.label[item]
    def __len__(self):
        return len(self.feature)
    
    @staticmethod
    def get_data(data_type):
        if data_type == "train":
            df = pd.read_csv("./train.csv")
        elif data_type == "test":
            df = pd.read_csv("./test.csv")
        # print(df.shape)
        #drop 無用的feature
        df = df.drop(columns=["Name","id","CGPA"])
        # print(df.shape)
        #合併pressure and Satisfaction
        df["Pressure"] = df["Academic Pressure"].fillna(df["Work Pressure"])
        df["Satisfaction"] = df['Study Satisfaction'].combine_first(df['Job Satisfaction'])
        df["Profession"] = df["Profession"].fillna(df["Working Professional or Student"])
        df = df.drop(columns=["Academic Pressure","Work Pressure","Study Satisfaction","Job Satisfaction","Working Professional or Student"])
        #確認missing value
        # print(df["Pressure"].isna().sum())
        # print(df["Satisfaction"].isna().sum())
        #drop 缺失值
        df.dropna(subset=["Pressure", "Satisfaction", "Financial Stress"],inplace=True)
        # print(df.shape)

        #處理category feature
        # print(df["Gender"].isna().sum())
        df["Gender"] = df["Gender"].map({'Male': 0, 'Female': 1})
        # print(df["Gender"])

        # df["Working Professional or Student"] = df["Working Professional or Student"].map({"Working Professional": 0, "Student": 1})
        # # print(df["Working Professional or Student"])

        # print(df["Dietary Habits"].isna().sum())
        # print(df.shape)
        df["Dietary Habits"] = df["Dietary Habits"].map({"Unhealthy": 0, "Moderate": 1, "Healthy": 2})
        df.dropna(subset=["Dietary Habits"], inplace=True)
        # print(df["Dietary Habits"])

        # print(df["Have you ever had suicidal thoughts ?"].isna().sum())
        df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"No": 0, "Yes": 1})
        # print(df["Have you ever had suicidal thoughts ?"])

        # print(df["Family History of Mental Illness"].isna().sum())
        df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"No": 0, "Yes": 1})
        # print(df["Family History of Mental Illness"])

        #use Label encoder
        label_encoder = LabelEncoder()
        df = df[df['City'] != "3.0"]
        # print(df.shape)
        # df['City'] = label_encoder.fit_transform(df['City'])
        # # print(label_encoder.classes_)

        # df["Profession"] = label_encoder.fit_transform(df["Profession"])
        # # print(label_encoder.classes_)

        # df["Sleep Duration"] = label_encoder.fit_transform(df["Sleep Duration"])
        # # print(label_encoder.classes_)

        # df["Degree"] = label_encoder.fit_transform(df["Degree"])
        # # print(label_encoder.classes_)

        # print(df["Profession"])
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
        # print(df.shape)
        df = df[df['Profession'].isin(valid_profession_values)]
        # print(df.shape)
        df = df[df['Sleep Duration'].isin(valid_sleep_duration_values)]
        # print(df.shape)
        df = df[df['Degree'].isin(valid_degree_values)]
        # print(df.shape)

        df = df[df['Work/Study Hours'] != 0]
        # print(df.shape)
        #移除極少值
        cityCnt = df["City"].value_counts()
        city = cityCnt[cityCnt >= 10].index
        df = df[df['City'].isin(city)]
        # print(df.shape)
        jobCnt = df["Profession"].value_counts()
        job = jobCnt[jobCnt >= 10].index
        df = df[df['Profession'].isin(job)]
        # print(df.shape)
        df['City'] = label_encoder.fit_transform(df['City'])
        # print(label_encoder.classes_)

        df["Profession"] = label_encoder.fit_transform(df["Profession"])
        # print(label_encoder.classes_)

        df["Sleep Duration"] = df["Sleep Duration"].map({"Less than 5 hours": 0, "5-6 hours":1, "7-8 hours": 2, "More than 8 hours":3})

        df["Degree"] = label_encoder.fit_transform(df["Degree"])
        # print(label_encoder.classes_)

        #age bining
        df['Age'] = pd.cut(df['Age'], bins=[18, 45, 60], labels=[0, 1])
        df.dropna(subset=["Age"],inplace=True)
        # print(df.shape)
        # 最後再確認是否有缺失值，並列出有缺失值的欄位
        missing_values = df.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        if not missing_columns.empty:
            print("資料集中存在缺失值的欄位及其數量:")
            print(missing_columns)
            print("正在移除包含缺失值的行...")
            df = df.dropna()
            print("移除後資料集大小:", df.shape)
        else:
            print("資料集中沒有缺失值。")

        df.to_csv('output.csv')

        MentalHealthyDataset.dataset = df
        return MentalHealthyDataset.dataset.copy()
    
    @staticmethod
    def get_X_y(targets, data_type):
        df = MentalHealthyDataset.get_data(data_type)
        
        X = df[targets]
        if data_type == "train":
            y = df["Depression"]
            return X, y
        elif data_type == "test":
            y = None
            return X, y
class MentalHealthyDataset_test(Dataset):
    dataset = None
    def __init__(self, targets, data_type="train"):
        self.feature, self.label = MentalHealthyDataset_test.get_X_y(targets, data_type)

    def __getitem__(self, item):
        input_data = {}
        for k in self.feature.keys():
            input_data[k] = self.feature[k][item]
        return input_data, self.label[item] if self.label is not None else None

    def __len__(self):
        return len(self.feature)

    @staticmethod
    def get_data(data_type="train"):
        if data_type == "train":
            df = pd.read_csv("./train.csv")
        elif data_type == "test":
            df = pd.read_csv("./test.csv")

        # --- 1) 移除確定不用的欄位 (這不會影響 row 數量) ---
        df = df.drop(columns=["Name","id","CGPA"])
        
        # --- 2) 合併 Pressure 與 Satisfaction 欄位 ---
        #     將 Academic Pressure 的缺失值用 Work Pressure 來補，反之亦然
        df["Pressure"] = df["Academic Pressure"].fillna(df["Work Pressure"])
        df["Satisfaction"] = df["Study Satisfaction"].combine_first(df["Job Satisfaction"])
        df["Profession"] = df["Profession"].fillna(df["Working Professional or Student"])
        
        # 這三個欄位都已合併到新欄位，移除原欄位
        df = df.drop(columns=[
            "Academic Pressure",
            "Work Pressure",
            "Study Satisfaction",
            "Job Satisfaction",
            "Working Professional or Student"
        ])

        # --- 3) 缺失值填補 (Imputation)，取代原本的 dropna ---
        # 例如，用中位數或平均值來填補數值欄位
        # 這邊示範用中位數來填補 Pressure, Satisfaction
        if df["Pressure"].isna().sum() > 0:
            mean_pressure = df["Pressure"].mean()
            df["Pressure"] = df["Pressure"].fillna(mean_pressure)

        if df["Satisfaction"].isna().sum() > 0:
            median_satisfaction = df["Satisfaction"].median()
            df["Satisfaction"] = df["Satisfaction"].fillna(median_satisfaction)
        if df["Financial Stress"].isna().sum() > 0:
            mean_stress = df["Financial Stress"].mean()
            df["Financial Stress"] = df["Financial Stress"].fillna(mean_stress)
        # --- 4) 類別型欄位缺失值填補 ---
        # 例如 Dietary Habits, Gender, ... 都可以加一個未知類別
        df["Gender"] = df["Gender"].map({'Male': 0, 'Female': 1})
        df["Gender"] = df["Gender"].fillna(-1)  # -1 代表 Unknown/未指定

        df["Dietary Habits"] = df["Dietary Habits"].map({
            "Unhealthy": 0,
            "Moderate": 1,
            "Healthy": 2
        })
        df["Dietary Habits"] = df["Dietary Habits"].fillna(-1)  # -1 代表 Unknown

        # suicidal thoughts
        df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"No": 0, "Yes": 1})
        df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].fillna(-1)

        # family history
        df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"No": 0, "Yes": 1})
        df["Family History of Mental Illness"] = df["Family History of Mental Illness"].fillna(-1)

        # --- 5) 對 City, Profession, Sleep Duration, Degree 等欄位做「未知類別」映射 ---
        #     先定義好有效清單
        valid_city_values = [
            'Agra','Ahmedabad','Bangalore','Bhopal','Chennai','Delhi','Faridabad','Ghaziabad','Gurgaon',
            'Hyderabad','Indore','Jaipur','Kalyan','Kanpur','Khaziabad','Kolkata','Lucknow','Ludhiana',
            'Meerut','Morena','Mumbai','Nagpur','Nashik','Patna','Pune','Rajkot','Srinagar','Surat',
            'Thane','Vadodara','Varanasi','Vasai-Virar','Visakhapatnam'
        ]
        valid_profession_values = [
            'Academic','Accountant','Analyst','Architect','Business Analyst','Chef','Chemist','City Manager',
            'Civil Engineer','Consultant','Content Writer','Customer Support','Data Scientist','Dev',
            'Digital Marketer','Doctor','Educational Consultant','Electrician','Entrepreneur','Family Consultant',
            'Financial Analyst','Graphic Designer','HR Manager','Investment Banker','Judge','Lawyer','Manager',
            'Marketing Manager','Mechanical Engineer','Medical Doctor','Pharmacist','Pilot','Plumber',
            'Research Analyst','Researcher','Sales Executive','Software Engineer','Teacher','Travel Consultant',
            'UX/UI Designer','Student'
        ]
        valid_sleep_duration_values = [
            'Less than 5 hours','5-6 hours','7-8 hours','More than 8 hours'
        ]
        valid_degree_values = [
            'ACA','B B.Com','B BA','B.Arch','B.B.Arch','B.Com','B.Ed','B.Pharm','B.Sc','B.Tech','BA','BArch','BBA',
            'BCA','BE','BEd','BHM','BPA','BPharm','BSc','E.Tech','K.Ed','L.Ed','LHM','LL B.Ed','LL.Com','LLB',
            'LLBA','LLCom','LLEd','LLM','LLS','M.Arch','M.Com','M.Ed','M.Pharm','M.S','M.Tech','MA','MBA','MBBS',
            'MCA','MD','ME','MEd','MHM','MPA','MPharm','MSc','MTech','M_Tech','N.Pharm','P.Com','P.Pharm','PhD',
            'S.Arch','S.Pharm','S.Tech'
        ]

        # 將原本會直接過濾掉的條件，改成：如果不在 valid 裡，就設定為 "OTHER"。
        # City = "3.0" 也改成 OTHER。
        df["City"] = df["City"].replace("3.0", np.nan)  # 先視為遺失值
        df["City"] = df["City"].apply(lambda x: x if x in valid_city_values else "OTHER")

        df["Profession"] = df["Profession"].apply(lambda x: x if x in valid_profession_values else "OTHER")
        
        # Sleep Duration 不在 valid list 時，設為 OTHER
        df["Sleep Duration"] = df["Sleep Duration"].apply(
            lambda x: x if x in valid_sleep_duration_values else "OTHER"
        )

        # Degree 不在 valid list 時，設為 OTHER
        df["Degree"] = df["Degree"].apply(lambda x: x if x in valid_degree_values else "OTHER")

        # Work/Study Hours == 0 時，視為無效，可用中位數或平均值來填補
        if (df["Work/Study Hours"] == 0).sum() > 0:
            median_hours = df.loc[df["Work/Study Hours"] != 0, "Work/Study Hours"].median()
            df.loc[df["Work/Study Hours"] == 0, "Work/Study Hours"] = median_hours

        # --- 6) 移除「過少樣本」前，先考慮：若 city / profession 出現次數少於 10，就改成 OTHER ---
        #     （這樣就不用刪除 row，而是把它合併到 OTHER 類別）
        city_count = df["City"].value_counts()
        rare_cities = city_count[city_count < 10].index
        # 注意：已經有 "OTHER" 這個類別，所以這邊要再把小於 10 的都塞到 "OTHER"
        df.loc[df["City"].isin(rare_cities), "City"] = "OTHER"

        job_count = df["Profession"].value_counts()
        rare_jobs = job_count[job_count < 10].index
        df.loc[df["Profession"].isin(rare_jobs), "Profession"] = "OTHER"

        # --- 7) Label Encoding（包含把 "OTHER" 也一起編碼） ---
        le_city = LabelEncoder()
        df["City"] = le_city.fit_transform(df["City"])

        le_profession = LabelEncoder()
        df["Profession"] = le_profession.fit_transform(df["Profession"])

        # Sleep Duration：原本有 4 種，就再加一個 "OTHER" => 5 種
        # 用 map 前，先改成對應字典
        sleep_map = {
            "Less than 5 hours": 0,
            "5-6 hours": 1,
            "7-8 hours": 2,
            "More than 8 hours": 3,
            "OTHER": 4  # 新增一個 4
        }
        df["Sleep Duration"] = df["Sleep Duration"].map(sleep_map)

        # Degree：先做 LabelEncoder，但要注意 "OTHER" 也要一起
        le_degree = LabelEncoder()
        df["Degree"] = le_degree.fit_transform(df["Degree"])

        # --- 8) Age binning ---
        #    如果年齡超過 60，或小於 18，原本可能被丟棄；現在可以把它們歸成特別區段
        #    例如把 18 以下都當 0 區段，>60 歸 2 區段。
        #    這裡假設原本只想區分: [18, 45), [45, 60), [60, ∞)
        df["Age"] = df["Age"].apply(lambda x: float(x) if pd.notnull(x) else np.nan)
        
        def bin_age(age):
            if pd.isna(age):
                return -1  # 不知道
            elif age < 18:
                return -1  # 未成年，不在資料範圍，當 Unknown
            elif age < 45:
                return 0
            elif age < 60:
                return 1
            else:
                return 2  # 大於等於 60

        df["Age"] = df["Age"].apply(bin_age)
        df.to_csv(f"output_{data_type}.csv", index=False)

        # 將結果存到 class 變數
        MentalHealthyDataset_test.dataset = df
        return df.copy()  # 回傳一份拷貝

    @staticmethod
    def get_X_y(targets, data_type="train"):
        df = MentalHealthyDataset_test.get_data(data_type)
        
        X = df[targets]  # 取得 features
        if data_type == "train":
            y = df["Depression"]  # 只有 train.csv 才有 label
            return X, y
        else:  # test
            y = None
            return X, y
