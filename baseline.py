#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from datetime import date
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve

weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]

# convert Discount_rate and Distance
def getDiscountType(row):
    if row == 'null':
        return 'null'
    elif ':' in row:
        return 1
    else:
        return 0


def convertRate(row):
    """Convert discount to rate"""
    if row == 'null':
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    else:
        return float(row)


def getDiscountMan(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0


def getDiscountJian(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


#handle Date_received and Date
def getWeekday(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


def processData(df):
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print(df['discount_rate'].unique())
    # convert distance
    df['distance'] = df['Distance'].replace('null', -1).astype(int)
    print(df['distance'].unique())
    # Date_received and Date
    df['weekday'] = df['Date_received'].astype(str).apply(getWeekday)
    # weekday_type :  周六和周日为1，其他为0
    df['weekday_type'] = df['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
    # change weekday to one-hot encoding
    tmpdf = pd.get_dummies(df['weekday'].replace('null', np.nan))
    tmpdf.columns = weekdaycols
    df[weekdaycols] = tmpdf
    return df


# product sample
def label(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


#sample path
off_path = '/Users/leeven/PycharmProjects/tensorflow/tianchi/ccf_offline_stage1_train.csv'
test_path = '/Users/leeven/PycharmProjects/tensorflow/tianchi/ccf_offline_stage1_test_revised.csv'

dfoff = pd.read_csv(off_path, keep_default_na=False)
dftest = pd.read_csv(test_path, keep_default_na=False)
#dfon = pd.read_csv('/Users/leeven/PycharmProjects/tensorflow/tianchi/ccf_online_stage1_train.csv')

#handle Discount_rate and Distance
dfoff = processData(dfoff)
dftest = processData(dftest)

print(dfoff.head(5))

# product sample
dfoff['label'] = dfoff.apply(label, axis = 1)

# data split
df = dfoff[dfoff['label'] != -1].copy()
train = df[(df['Date_received'] < '20160516')].copy()
valid = df[(df['Date_received'] >= '20160516') & (df['Date_received'] <= '20160615')].copy()
print(train['label'].value_counts())
print(valid['label'].value_counts())

# feature
original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
print(len(original_feature),original_feature)

# LR Model Train
lr_model = logistic_regression_classifier(train[original_feature], train['label'])

# valid predict
y_valid_pred = lr_model.predict_proba(valid[original_feature])
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:, 1]

# avgAUC calculation
vg = valid1.groupby(['Coupon_id'])
aucs = []
for i in vg:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))

'''
# test prediction for submission
y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['label'] = y_test_pred[:,1]
dftest1.to_csv('submit1.csv', index=False, header=False)
dftest1.head()
'''