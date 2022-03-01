import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import KNNImputer, SimpleImputer


class LauraProcess:

    def __init__(self):
        self.k = 0

    def process(self):
        files = ['location_v2.csv','satisfaction.csv','services_mean.csv','status_v2.csv']
        Train_IDs_df = pd.read_csv(r'src\Train_IDs.csv')
        Test_IDs_df = pd.read_csv(r'src\Test_IDs.csv')
        population_df = pd.read_csv(r'src\population.csv')

        df = pd.read_csv(r'src\demographics_full.csv')
        for file in files:
            other = pd.read_csv(r'src\\'+file)
            if 'Count' in other.columns:
                other = other.drop(['Count'], axis=1)
            col_file = file.replace('.csv','')
            df = df.merge(other, how='outer', left_on='Customer ID', right_on='Customer ID',suffixes=(None, '_'+col_file))
            if file =='location_v2.csv':
                df = df.merge(population_df, how='left', left_on='Zip Code', right_on='Zip Code',suffixes=(None, '_'+col_file))

        cleanup_class = {"Referred a Friend": {"No": 0, "Yes": 1},
                         "Phone Service": {"No": 0, "Yes": 1},
                         "Multiple Lines": {"No": 0, "Yes": 1},
                         "Internet Service": {"No": 0, "Yes": 1},
                         "Online Security": {"No": 0, "Yes": 1},
                         "Online Backup": {"No": 0, "Yes": 1},
                         "Device Protection Plan": {"No": 0, "Yes": 1},
                         "Premium Tech Support": {"No": 0, "Yes": 1},
                         "Streaming TV": {"No": 0, "Yes": 1},
                         "Streaming Movies": {"No": 0, "Yes": 1},
                         "Streaming Music": {"No": 0, "Yes": 1},
                         "Unlimited Data": {"No": 0, "Yes": 1},
                         "Paperless Billing": {"No": 0, "Yes": 1},}
        df = df.replace(cleanup_class)

        data_dum = df.iloc[:, 1:]
        data_dum = data_dum.drop(['Count','Zip Code','Latitude','Longitude','Lat Long','ID'], axis=1)
        data_dum = pd.get_dummies(data_dum)
        data_dum['Customer ID'] = df.iloc[:,0]
        # print(data_dum)
        # print(list(data_dum.columns.values))

        scale_columns = ['Age',
                            'Number of Dependents', 'Satisfaction Score',
                            'Number of Referrals', 'Tenure in Months',
                            'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download',
                            'Monthly Charge', 'Total Charges', 'Total Refunds',
                            'Total Extra Data Charges', 'Total Long Distance Charges',
                            'Total Revenue', 'Population']
        #train
        train_df = Train_IDs_df.merge(data_dum, how='left', left_on='Customer ID', right_on='Customer ID',suffixes=(None, '_'+col_file))
        train_raw_X = train_df.drop(['Customer ID','Churn Category'], axis=1)
        train_raw_X = train_raw_X.fillna(0)
        train_raw_y = train_df['Churn Category']
        train_raw_y = train_raw_y.fillna(-1)
        train_raw_y = train_raw_y.astype(int)

        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler = min_max_scaler.fit(train_raw_X[scale_columns])
        train_raw_X.loc[:,scale_columns] = min_max_scaler.transform(train_raw_X[scale_columns])
        train_raw_X['Customer ID'] = train_df['Customer ID']
        # print(train_raw_X)
        # print(np.shape(train_raw_X))

        # test
        test_df = Test_IDs_df.merge(data_dum, how='left', left_on='Customer ID', right_on='Customer ID',suffixes=(None, '_'+col_file))
        test_raw_X = test_df.drop(['Customer ID','Churn Category'], axis=1)
        test_raw_X = test_raw_X.fillna(0)

        test_raw_X.loc[:,scale_columns] = min_max_scaler.transform(test_raw_X[scale_columns])
        test_raw_X['Customer ID'] = test_df['Customer ID']
        # print(train_raw_X)
        # print(np.shape(test_raw_X))

        return train_raw_X, train_raw_y, test_raw_X

    def process_impute_knn(self):
        files = ['location_v2.csv','satisfaction.csv','services_mean.csv','status_v2.csv']
        Train_IDs_df = pd.read_csv(r'src\Train_IDs.csv')
        Test_IDs_df = pd.read_csv(r'src\Test_IDs.csv')
        population_df = pd.read_csv(r'src\population.csv')

        df = pd.read_csv(r'src\demographics_logic_only.csv')
        for file in files:
            other = pd.read_csv(r'src\\'+file)
            if 'Count' in other.columns:
                other = other.drop(['Count'], axis=1)
            col_file = file.replace('.csv','')
            df = df.merge(other, how='outer', left_on='Customer ID', right_on='Customer ID',suffixes=(None, '_'+col_file))
            if file =='location_v2.csv':
                df = df.merge(population_df, how='left', left_on='Zip Code', right_on='Zip Code',suffixes=(None, '_'+col_file))

        cleanup_class = {"Referred a Friend": {"No": 0, "Yes": 1},
                         "Phone Service": {"No": 0, "Yes": 1},
                         "Multiple Lines": {"No": 0, "Yes": 1},
                         "Internet Service": {"No": 0, "Yes": 1},
                         "Online Security": {"No": 0, "Yes": 1},
                         "Online Backup": {"No": 0, "Yes": 1},
                         "Device Protection Plan": {"No": 0, "Yes": 1},
                         "Premium Tech Support": {"No": 0, "Yes": 1},
                         "Streaming TV": {"No": 0, "Yes": 1},
                         "Streaming Movies": {"No": 0, "Yes": 1},
                         "Streaming Music": {"No": 0, "Yes": 1},
                         "Unlimited Data": {"No": 0, "Yes": 1},
                         "Paperless Billing": {"No": 0, "Yes": 1},}
        df = df.replace(cleanup_class)

        data_dum = df.iloc[:, 1:]
        data_dum = data_dum.drop(['Count','Zip Code','Latitude','Longitude','Lat Long','ID'], axis=1)
        data_dum = pd.get_dummies(data_dum)
        data_dum['Customer ID'] = df.iloc[:,0]
        # print(data_dum)
        # print(list(data_dum.columns.values))

        scale_columns = ['Age',
                            'Number of Dependents', 'Satisfaction Score',
                            'Number of Referrals', 'Tenure in Months',
                            'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download',
                            'Monthly Charge', 'Total Charges', 'Total Refunds',
                            'Total Extra Data Charges', 'Total Long Distance Charges',
                            'Total Revenue', 'Population']
        #train
        train_df = Train_IDs_df.merge(data_dum, how='left', left_on='Customer ID', right_on='Customer ID',suffixes=(None, '_'+col_file))
        train_raw_X = train_df.drop(['Customer ID','Churn Category'], axis=1)
        train_raw_y = train_df['Churn Category']
        train_raw_y = train_raw_y.fillna(-1)
        train_raw_y = train_raw_y.astype(int)

        imp_mean = KNNImputer()
        imp_mean.fit(train_raw_X[scale_columns])
        train_raw_X.loc[:,scale_columns] = imp_mean.transform(train_raw_X[scale_columns])
        train_raw_X = train_raw_X.fillna(0)

        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler = min_max_scaler.fit(train_raw_X[scale_columns])
        train_raw_X.loc[:,scale_columns] = min_max_scaler.transform(train_raw_X[scale_columns])
        train_raw_X['Customer ID'] = train_df['Customer ID']
        # print(train_raw_X)
        # print(np.shape(train_raw_X))

        # test
        test_df = Test_IDs_df.merge(data_dum, how='left', left_on='Customer ID', right_on='Customer ID',suffixes=(None, '_'+col_file))
        test_raw_X = test_df.drop(['Customer ID','Churn Category'], axis=1)
        test_raw_X.loc[:, scale_columns] = imp_mean.transform(test_raw_X[scale_columns])
        test_raw_X = test_raw_X.fillna(0)

        test_raw_X.loc[:,scale_columns] = min_max_scaler.transform(test_raw_X[scale_columns])
        test_raw_X['Customer ID'] = test_df['Customer ID']
        # print(train_raw_X)
        # print(np.shape(test_raw_X))

        return train_raw_X, train_raw_y, test_raw_X

    def process_impute_simple(self):
        files = ['location_v2.csv','satisfaction.csv','services_mean.csv','status_v2.csv']
        Train_IDs_df = pd.read_csv(r'src\Train_IDs.csv')
        Test_IDs_df = pd.read_csv(r'src\Test_IDs.csv')
        population_df = pd.read_csv(r'src\population.csv')

        df = pd.read_csv(r'src\demographics_logic_only.csv')
        for file in files:
            other = pd.read_csv(r'src\\'+file)
            if 'Count' in other.columns:
                other = other.drop(['Count'], axis=1)
            col_file = file.replace('.csv','')
            df = df.merge(other, how='outer', left_on='Customer ID', right_on='Customer ID',suffixes=(None, '_'+col_file))
            if file =='location_v2.csv':
                df = df.merge(population_df, how='left', left_on='Zip Code', right_on='Zip Code',suffixes=(None, '_'+col_file))

        cleanup_class = {"Referred a Friend": {"No": 0, "Yes": 1},
                         "Phone Service": {"No": 0, "Yes": 1},
                         "Multiple Lines": {"No": 0, "Yes": 1},
                         "Internet Service": {"No": 0, "Yes": 1},
                         "Online Security": {"No": 0, "Yes": 1},
                         "Online Backup": {"No": 0, "Yes": 1},
                         "Device Protection Plan": {"No": 0, "Yes": 1},
                         "Premium Tech Support": {"No": 0, "Yes": 1},
                         "Streaming TV": {"No": 0, "Yes": 1},
                         "Streaming Movies": {"No": 0, "Yes": 1},
                         "Streaming Music": {"No": 0, "Yes": 1},
                         "Unlimited Data": {"No": 0, "Yes": 1},
                         "Paperless Billing": {"No": 0, "Yes": 1},}
        df = df.replace(cleanup_class)

        data_dum = df.iloc[:, 1:]
        data_dum = data_dum.drop(['Count','Zip Code','Latitude','Longitude','Lat Long','ID'], axis=1)
        data_dum = pd.get_dummies(data_dum)
        data_dum['Customer ID'] = df.iloc[:,0]
        # print(data_dum)
        # print(list(data_dum.columns.values))

        scale_columns = ['Age',
                            'Number of Dependents', 'Satisfaction Score',
                            'Number of Referrals', 'Tenure in Months',
                            'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download',
                            'Monthly Charge', 'Total Charges', 'Total Refunds',
                            'Total Extra Data Charges', 'Total Long Distance Charges',
                            'Total Revenue', 'Population']
        binary_columns = ['Under 30', 'Senior Citizen', 'Dependents', 'Referred a Friend', 'Phone Service',
                          'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup',
                          'Device Protection Plan',
                          'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music',
                          'Unlimited Data', ]

        #train
        train_df = Train_IDs_df.merge(data_dum, how='left', left_on='Customer ID', right_on='Customer ID',suffixes=(None, '_'+col_file))
        train_raw_X = train_df.drop(['Customer ID','Churn Category'], axis=1)
        train_raw_y = train_df['Churn Category']
        train_raw_y = train_raw_y.fillna(-1)
        train_raw_y = train_raw_y.astype(int)

        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(train_raw_X[scale_columns])
        train_raw_X.loc[:,scale_columns] = imp_mean.transform(train_raw_X[scale_columns])
        imp = SimpleImputer(strategy="most_frequent")
        imp.fit(train_raw_X[binary_columns])
        train_raw_X.loc[:,binary_columns] = imp.transform(train_raw_X[binary_columns])
        train_raw_X = train_raw_X.fillna(0)

        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler = min_max_scaler.fit(train_raw_X[scale_columns])
        train_raw_X.loc[:,scale_columns] = min_max_scaler.transform(train_raw_X[scale_columns])
        train_raw_X['Customer ID'] = train_df['Customer ID']
        # print(train_raw_X)
        # print(np.shape(train_raw_X))

        # test
        test_df = Test_IDs_df.merge(data_dum, how='left', left_on='Customer ID', right_on='Customer ID',suffixes=(None, '_'+col_file))
        test_raw_X = test_df.drop(['Customer ID','Churn Category'], axis=1)
        test_raw_X.loc[:, scale_columns] = imp_mean.transform(test_raw_X[scale_columns])
        test_raw_X = test_raw_X.fillna(0)

        test_raw_X.loc[:,scale_columns] = min_max_scaler.transform(test_raw_X[scale_columns])
        test_raw_X['Customer ID'] = test_df['Customer ID']
        # print(train_raw_X)
        # print(np.shape(test_raw_X))

        return train_raw_X, train_raw_y, test_raw_X


if __name__ == '__main__':
    laura = LauraProcess()
    # X_train, y_train, X_test = laura.process()
    # X_train, y_train, X_test = laura.process_impute_knn()
    X_train, y_train, X_test = laura.process_impute_simple
    # s = pd.Series(y_train, dtype="category")
    # print(s)
    print('X_train=====================')
    print(X_train)
    # print('y_train=====================')
    # print(y_train)
    # print('X_test======================')
    # print(X_test)
