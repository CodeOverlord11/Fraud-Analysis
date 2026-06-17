import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings 
import datetime as dt
from math import radians,sin,cos,sqrt,atan2

sns.set_style('whitegrid')
plt.rcParams['figure.figsize']=(12,6)

df=pd.read_csv('fraudTrain.csv')
#print(df.describe())

fraud_rate=df['is_fraud'].value_counts(normalize=True) * 100
#print(fraud_rate)

sns.countplot(x='is_fraud',data=df)
plt.title("Fraud vs Non - fraud distribution")
#plt.show()

sns.histplot(data=df,hue='is_fraud',bins=50,x='amt',kde=True,log_scale=True)
plt.title("Transaction amount distribution")
#plt.show()

#print("Fraud transaction statistic:")
#print(df.groupby('is_fraud')['amt'].describe())

#print(df['amt'].mean())


bins = [0, 1000, 5000, 10000, 20000, 50000, 100000, float('inf')]
labels = ['0-1k', '1k-5k', '5k-10k', '10k-20k', '20k-50k', '50k-1L', '1L+']

df['amount_bucket'] = pd.cut(df['amt'], bins=bins, labels=labels, right=False)

analysis = df.groupby('amount_bucket').agg(
    total_txns=('is_fraud', 'count'),
    fraud_txns=('is_fraud', 'sum'),
    fraud_rate_pct=('is_fraud', lambda x: round(x.mean()*100, 3)),
    total_amount=('amt', 'sum'),
    fraud_amount=('amt', lambda x: x[df['is_fraud']==1].sum())
).round(2)

print(analysis)

fraud_rate_by_cat = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False) * 100
#print(fraud_rate_by_cat.round(3))
#We see that shopping_net has the highest percentage of frauds(1.743%), followed by misc_nets , grocery_pos . It is more than 3x the avg fraud rate . 
#To fix the fraud rate , implement 2 factor authetication , verify validity of vendor , have banks issue an SMS of deposit of money to customer , and allow for immediate freezing of bank account incase of an unauthroized transaction
#fraud_rate=df['is_fraud'].mean() * 100
#print(fraud_rate)

#So now we have
#Category wise fraud analysis

#Phase 2:EDa


# Basic stats
#print(df.groupby('is_fraud')['amt'].describe())

#Time-based analysis(at what time of the day does transaction occur the most)
df['trans_date_trans_time']=pd.to_datetime(df['trans_date_trans_time'].str.strip(),format="mixed",dayfirst=True)
df['hours']=df['trans_date_trans_time'].dt.hour
df['day_of_the_week']=df['trans_date_trans_time'].dt.day_name()
df['month']=df['trans_date_trans_time'].dt.month

# plt.figure(figsize=(10,6))
# sns.lineplot(x='hours',y='is_fraud',data=df,estimator='mean')
# plt.xlabel('Hours')
# plt.ylabel('Fraud')
# plt.xticks(range(0,24))
# plt.show()

#Based on the line plot generate , fraud happens the most between 0 to 3 am , with a substantial drop followed by a sharp rise after 9 pm to 11 pm
#To fix this , we recommend stricter rule for fraud checking during this time period , for example limiting fast and frequent transactions , improved security checking

def distance_analysis(lat1,long1,lat2,long2):
    R=6371
    lat1,lat2,long1,long2=map(radians,[lat1,lat2,long1,long2])
    dlat=lat2-lat1
    dlong=long2-long1
    a=sin(dlat/2)**2 + cos(lat1)*cos(lat2) * sin(dlong/2)**2
    
    c=2 * atan2(sqrt(a),sqrt(1-a))
    return c * R 

df['distance_km']=df.apply(lambda row: distance_analysis(
    row['lat'],row['long'],
    row['merch_lat'],row['merch_long']
),axis=1)

fraud_distance=df[df['is_fraud']==1].groupby('cc_num')['distance_km'].mean()




#print(fraud_distance.head())

#Average distance between 2 fraud transactions happens at 76 km

#mean       76.237524
# std        28.698942
# min         0.738769
# 25%        55.623008
# 50%        77.869499
# 75%        98.372391
# max       144.522410
# Name: distance_km, dtype: float64



# count    1.042569e+06
# mean     7.610140e+01
# std      2.911500e+01
# min      2.225452e-02
# 25%      5.533037e+01
# 50%      7.821246e+01
# 75%      9.848084e+01
# max      1.521172e+02

#We can see that the avg distance between transactions between transaction between fraud and non fraud is not much , hence this is not a good predictor of a fraud transactin

#grouping cc num of fraud transactions , and identifying factors such as time gap btwn consecutive tnsns , 
# 1. Sort the data
df = df.sort_values(['cc_num', 'trans_date_trans_time']).reset_index(drop=True)

# 2. Calculate time difference between consecutive frauds per credit card
df['fraud_time_diff'] = df[df['is_fraud'] == 1].groupby('cc_num')['trans_date_trans_time'].diff()

# Optional: Convert to hours for easier understanding
df['fraud_time_diff_hours'] = df['fraud_time_diff'].dt.total_seconds() / 3600

# Check results
#print(df[['cc_num', 'trans_date_trans_time', 'is_fraud', 'fraud_time_diff', 'fraud_time_diff_hours']].head(20))
# 1. See only rows where fraud actually happened
fraud_only = df[df['is_fraud'] == 1][['cc_num', 'trans_date_trans_time', 'is_fraud', 'fraud_time_diff', 'fraud_time_diff_hours']]

#print("First 15 Fraud Transactions with Time Difference:")
#print(fraud_only.head(15))

# 2. Summary Statistics of time between frauds
#print("\n=== Time Between Consecutive Frauds (in Hours) ===")
#print(df['fraud_time_diff_hours'].describe())

# 3. How many frauds have a previous fraud on same card?
#print("\nNumber of frauds that have previous fraud on same card:", 
df['fraud_time_diff_hours'].notna().sum()

print(df['fraud_time_diff_hours'].mean())

#Analyzing the time difference between consecutive fraud transactions , we see that several transactions take place within one day 
#We see that multiple transactions happen in an extremely short interval as well , for eg. between tn no120 and 121 happen within 7 minutes . This is a classic case of velocity transactions which are often seen if a card is compromised
#Recommendations : Flag accounts making multiple transactions within a short timespan . 



df['high_risk_category'] = df['category'].isin(['shopping_net', 'misc_net', 'grocery_pos']).astype(int)

# 2. High Value Transaction Flag
df['high_value_txn'] = df['amount_bucket'].isin(['20k-50k', '50k-1L', '1L+']).astype(int)

# 3. Night Transaction
df['night_txn'] = df['hours'].isin(range(0,6)).astype(int)

# 4. Velocity Features (Very Powerful)
df = df.sort_values(['cc_num', 'trans_date_trans_time'])
df['txns_last_1h'] = df.groupby('cc_num').rolling('1h', on='trans_date_trans_time').count()['amt'].reset_index(0, drop=True)
df['txns_last_24h'] = df.groupby('cc_num').rolling('24h', on='trans_date_trans_time').count()['amt'].reset_index(0, drop=True)

# 5. Distance Flag
df['far_transaction'] = (df['distance_km'] > 80).astype(int)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Select features
features = ['amt', 'high_risk_category', 'high_value_txn', 'night_txn', 
            'distance_km', 'far_transaction', 'txns_last_1h', 'txns_last_24h', 'hours']

X = df[features]
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost
pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
model = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42,eval_metric='auc')  # handles imbalance
model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]
df['fraud_probability']=(model.predict_proba(X)[:,1]*100).round(2)
print(classification_report(y_test, pred))
print("AUC-ROC:", roc_auc_score(y_test, prob))

import shap

explainer=shap.TreeExplainer(model)
shap_values=explainer(X_test)

#shap.summary_plot(shap_values,X_test)

powerbi_df=df[[
    'trans_date_trans_time',
    'category',
    'amt',
    'is_fraud',
    'hours',
    'amount_bucket',
    'distance_km',
    'txns_last_1h',
    'txns_last_24h',
    'night_txn',
    'high_risk_category',
    'fraud_probability',
    'fraud_time_diff_hours'
]]
import os
print("Current directory:" ,os.getcwd())
powerbi_df.to_csv('fraud_powerbi_df.csv')
print("fraud_time_diff_hours" in powerbi_df.columns)


