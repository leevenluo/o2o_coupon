"""
o2o data mining competition
"""
#!/usr/bin/python
# -*- coding:utf-8 -*-

from datetime import date
from sklearn.metrics import auc, roc_curve

import numpy as np
import pandas as pd

WEEKDAYCOLS = ['weekday_' + str(i) for i in range(1, 8)]


def get_discount_type(row):
    """convert Discount_rate and Distance"""
    if row == 'null':
        return 'null'
    elif ':' in row:
        return 1
    return 0


def convert_rate(row):
    """Convert discount to rate"""
    if row == 'null':
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    return float(row)


def get_discount_man(row):
    """Convert discount to man"""
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    return 0


def get_discount_jian(row):
    """Convert discount to jian"""
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    return 0


def get_weekday(row):
    """handle Date_received and Date"""
    if row == 'null':
        return row
    return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


def process_data(df_data):
    """process data"""
    # convert discunt_rate
    df_data['discount_rate'] = df_data['Discount_rate'].apply(convert_rate)
    df_data['discount_man'] = df_data['Discount_rate'].apply(get_discount_man)
    df_data['discount_jian'] = df_data['Discount_rate'].apply(
        get_discount_jian)
    df_data['discount_type'] = df_data['Discount_rate'].apply(
        get_discount_type)
    print(df_data['discount_rate'].unique())
    # convert distance
    df_data['distance'] = df_data['Distance'].replace('null', -1).astype(int)
    print(df_data['distance'].unique())
    # Date_received and Date
    df_data['weekday'] = df_data['Date_received'].astype(
        str).apply(get_weekday)
    # weekday_type :  周六和周日为1，其他为0
    df_data['weekday_type'] = df_data['weekday'].apply(
        lambda x: 1 if x in [6, 7] else 0)
    # change weekday to one-hot encoding
    tmp_df_data = pd.get_dummies(df_data['weekday'].replace('null', np.nan))
    tmp_df_data.columns = WEEKDAYCOLS
    df_data[WEEKDAYCOLS] = tmp_df_data
    return df_data


def label(row):
    """product sample"""
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        pd_date = pd.to_datetime(row['Date'], format='%Y%m%d')
        pd_date_rec = pd.to_datetime(row['Date_received'], format='%Y%m%d')
        td_minus = pd_date - pd_date_rec
        if td_minus <= pd.Timedelta(15, 'D'):
            return 1
    return 0


def logistic_regression_classifier(train_x, train_y):
    """Logistic Regression Classifier"""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


def process_user_data(df_data):
    """docstring for process_user_data"""
    user_key = df_data[['User_id']].copy().drop_duplicates()
    # count
    user_f1 = df_data[df_data['Date_received'] != 'null'][['User_id']].copy()
    user_f1['u_coupon_count'] = 1
    user_f1 = user_f1.groupby(['User_id'], as_index=False).count()

    user_f2 = df_data[df_data['Date'] != 'null'][['User_id']].copy()
    user_f2['u_buy_count'] = 1
    user_f2 = user_f2.groupby(['User_id'], as_index=False).count()

    user_f3 = df_data[(df_data['Date'] != 'null') &
                      (df_data['Date_received'] != 'null')][['User_id']].copy()
    user_f3['u_buy_with_coupon_count'] = 1
    user_f3 = user_f3.groupby(['User_id'], as_index=False).count()

    user_f4 = df_data[df_data['Date'] != 'null'][[
        'User_id', 'Merchant_id']].copy()
    user_f4.drop_duplicates(inplace=True)
    user_f4 = user_f4.groupby(['User_id'], as_index=False).count()
    user_f4.rename(columns={'Merchant_id': 'u_merchant_count'}, inplace=True)

    # distance
    user_dis_tmp = df_data[(df_data['Date'] != 'null') &
                           (df_data['Date_received'] != 'null')][['User_id', 'distance']].copy()
    user_dis_tmp.replace(-1, np.nan, inplace=True)
    user_f5 = user_dis_tmp.groupby(['User_id'], as_index=False).min()
    user_f5.rename(columns={'distance': 'u_distance_min'}, inplace=True)

    user_f6 = user_dis_tmp.groupby(['User_id'], as_index=False).max()
    user_f6.rename(columns={'distance': 'u_distance_max'}, inplace=True)

    user_f7 = user_dis_tmp.groupby(['User_id'], as_index=False).median()
    user_f7.rename(columns={'distance': 'u_distance_median'}, inplace=True)

    user_f8 = user_dis_tmp.groupby(['User_id'], as_index=False).mean()
    user_f8.rename(columns={'distance': 'u_distance_mean'}, inplace=True)

    # merge all feature
    user_feature = pd.merge(user_key, user_f1, on='User_id', how='left')
    user_feature = pd.merge(user_feature, user_f2, on='User_id', how='left')
    user_feature = pd.merge(user_feature, user_f3, on='User_id', how='left')
    user_feature = pd.merge(user_feature, user_f4, on='User_id', how='left')
    user_feature = pd.merge(user_feature, user_f5, on='User_id', how='left')
    user_feature = pd.merge(user_feature, user_f6, on='User_id', how='left')
    user_feature = pd.merge(user_feature, user_f7, on='User_id', how='left')
    user_feature = pd.merge(user_feature, user_f8, on='User_id', how='left')

    # rate
    user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon_count'].astype(float) / \
        user_feature['u_coupon_count'].astype(float)
    user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon_count'].astype(float) / \
        user_feature['u_buy_count'].astype(float)
    user_feature.fillna(0, inplace=True)

    return user_feature


# sample path
PREFIX_PATH = '/Users/leevenluo/Desktop/o2o_coupon/'
OFF_PATH = PREFIX_PATH + 'ccf_offline_stage1_TRAIN.csv'
TEST_PATH = PREFIX_PATH + 'ccf_offline_stage1_test_revised.csv'

DFOFF = pd.read_csv(OFF_PATH, keep_default_na=False)
DFTEST = pd.read_csv(TEST_PATH, keep_default_na=False)
# DFon = pd.read_csv(prefix_pat + 'ccf_online_stage1_TRAIN.csv')

# handle Discount_rate and Distance
DFOFF = process_data(DFOFF)
DFTEST = process_data(DFTEST)
print('DFOFF.shape:' + str(DFOFF.shape))

# handle user feature
USER_DFOFF_FEATURE = process_user_data(DFOFF)
print('USER_DFOFF_FEATURE.shape:' + str(USER_DFOFF_FEATURE.shape))

# join feature
DFOFF_FEATURE = pd.merge(DFOFF, USER_DFOFF_FEATURE, on='User_id', how='left')
print('DFOFF_FEATURE.shape:' + str(DFOFF_FEATURE.shape))

DFTEST_FEATURE = pd.merge(DFTEST, USER_DFOFF_FEATURE,
                          on='User_id', how='left')
DFTEST_FEATURE = DFTEST_FEATURE.fillna(0)
print('DFTEST_FEATUR.shape:' + str(DFTEST_FEATURE.shape))

# product sample
# DFOFF['label'] = DFOFF.apply(label, axis=1)
DFOFF_FEATURE['label'] = DFOFF_FEATURE.apply(label, axis=1)

# data split
# DF = DFOFF[DFOFF['label'] != -1].copy()
DF = DFOFF_FEATURE[DFOFF_FEATURE['label'] != -1].copy()
TRAIN = DF[(DF['Date_received'] < '20160516')].copy()

VALID = DF[(DF['Date_received'] >= '20160516') & (
    DF['Date_received'] <= '20160615')].copy()
print(TRAIN['label'].value_counts())
print(VALID['label'].value_counts())

# feature
ORIGINAL_FEATURE = ['discount_rate', 'discount_type', 'discount_man',
                    'discount_jian', 'distance', 'weekday', 'weekday_type'] + WEEKDAYCOLS

# user feature
USER_FEATURE_COLUMNS = ['u_coupon_count', 'u_buy_count', 'u_buy_with_coupon_count', 'u_merchant_count', 'u_distance_min',
                        'u_distance_max', 'u_distance_median', 'u_distance_mean', 'u_use_coupon_rate', 'u_buy_with_coupon_rate']

ORIGINAL_FEATURE = ORIGINAL_FEATURE + USER_FEATURE_COLUMNS
print(len(ORIGINAL_FEATURE), ORIGINAL_FEATURE)

# LR Model TRAIN
LR_MODEL = logistic_regression_classifier(TRAIN[ORIGINAL_FEATURE],
                                          TRAIN['label'])

# VALID predict
Y_VALID_PRED = LR_MODEL.predict_proba(VALID[ORIGINAL_FEATURE])
VALID1 = VALID.copy()
VALID1['pred_prob'] = Y_VALID_PRED[:, 1]

# aVGAUC calculation
VG = VALID1.groupby(['Coupon_id'])
AUCS = []
for i in VG:
    tmpDF = i[1]
    if len(tmpDF['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpDF['label'],
                                     tmpDF['pred_prob'],
                                     pos_label=1)
    AUCS.append(auc(fpr, tpr))
print(np.average(AUCS))

# test prediction for submission
Y_VALID_PRED = LR_MODEL.predict_proba(DFTEST_FEATURE[ORIGINAL_FEATURE])
DFTEST1 = DFTEST_FEATURE[['User_id', 'Coupon_id', 'Date_received']].copy()
DFTEST1['label'] = Y_VALID_PRED[:, 1]
DFTEST1.to_csv('submit1.csv', index=False, header=False)
DFTEST1.head()
