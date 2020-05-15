#!/usr/bin/env python
# coding: utf-8

# In[311]:


import pandas as pd
from pandasql import sqldf
pd.set_option('display.max_columns', None)


# In[312]:


csvc = pd.read_pickle("customer_service_reps")
engagement = pd.read_pickle("engagement")
subs = pd.read_csv("subs.csv")
adspend = pd.read_excel("adspend.xlsx")


# In[3]:


subs.head()


# In[4]:


csvc.head()


# In[5]:


engagement.head()


# In[6]:


csvc.num_trial_days.value_counts()


# In[7]:


csvc7 = csvc[csvc['num_trial_days']==7]
csvc14 = csvc[csvc['num_trial_days']==14]
csvc0 = csvc[csvc['num_trial_days']==0]


# In[8]:


subs.plan_type.value_counts()


# In[9]:


subs['plan_type_days'] = 14

subs.loc[subs['plan_type']=='base_uae_no_trial_7_day_guarantee','plan_type_days']= 7
subs.loc[subs['plan_type']=='low_uae_no_trial','plan_type_days']= 0
subs.loc[subs['plan_type']=='low_sar_no_trial','plan_type_days']= 0
subs.loc[subs['plan_type']=='low_eur_no_trial','plan_type_days']= 0
subs.loc[subs['plan_type']=='low_eur_no_trial','plan_type_days']= 0


# In[10]:


subs.head()


# In[11]:


subs14 = subs[subs['plan_type_days']==14]


# # AB TESTING

# In[13]:


q = '''select count(distinct(subid)) as no_of_subid_7day_trial
from csvc7'''

total_7day_trial = sqldf(q)
total_7day_trial


# In[14]:


q1 = '''select count(distinct(subid)) as no_7day_converted
from csvc7
where payment_period = 1'''

no_7day_converted = sqldf(q1)
no_7day_converted


# In[15]:


# need to check if all customers in subscriber set was covered in customer service dataset

# Products in cpi holding that are not in cpi mapping
#set1 = set(cpiholding['Product_Code'].unique().tolist())
#set2 = set(cpimapping['Product_Code'].unique().tolist())
#diff = set1.difference(set2)
#missing_cpi_prod = list(diff)

# people in subscriber set that are not in csvc14
set1 = set(subs14['subid'].unique().tolist())
set2 = set(csvc14['subid'].unique().tolist())
diff = set1.difference(set2)
missing_subs_notcounted = list(diff)
len(missing_subs_notcounted)


# In[16]:


q2 = '''select count(distinct(subid)) as no_of_subid_14day_trial
from csvc14'''

total_14day_trial = sqldf(q2)
total_14day_trial


# In[17]:


q3 = '''select count(distinct(subid)) as no_14day_converted
from csvc14
where payment_period = 1'''

no_14day_converted = sqldf(q3)
no_14day_converted


# In[18]:


leftoutsubs = subs14[~subs14['subid'].isin(csvc14['subid'])]


# In[19]:


leftoutsubs.head()


# In[21]:


q4 = '''select count(distinct(subid)) as no_converted_14day_subs
from leftoutsubs
where (refund_after_trial_TF = 0) and (cancel_before_trial_ends = 0)'''

no_converted_14day_subs = sqldf(q4)
no_converted_14day_subs


# In[22]:


q5 = '''select count(distinct(subid)) as no_14day_subs_notin_csvc14
from leftoutsubs'''

no_14day_subs_notin_csvc14 = sqldf(q5)
no_14day_subs_notin_csvc14


# # adspend

# In[314]:


#cust_npb['FirstContactDay']=pd.to_datetime(cust_npb['FirstContactDay'],format='%Y%m%d')
subs['account_creation_date'] = pd.to_datetime(subs['account_creation_date'])


# In[315]:


#df['Dates'] = pd.to_datetime(df['date']).dt.date
#df['Time'] = pd.to_datetime(df['date']).dt.time

subs['Date'] = subs['account_creation_date'].dt.date


# In[316]:


subs.head()


# In[317]:


subs_new = subs.drop(['account_creation_date'],axis=1)


# In[318]:


subs_new.head()


# In[319]:


subs_new.dtypes


# In[320]:


subs_new['Date'] = pd.to_datetime(subs_new['Date'], format='%Y/%m/%d')


# In[321]:


subs_new.dtypes


# In[322]:


subs_new['Date'].dtype


# In[323]:


subs_new.head()


# In[324]:


adspend


# In[325]:


subs_new['adspend_month'] = "9"

startdate01 = pd.to_datetime("2019-06-30").date()
enddate01 = pd.to_datetime("2019-07-31").date()

subs_new.loc[(subs_new['Date'] >= startdate01)&(subs_new['Date'] < enddate01), 'adspend_month'] = '0'

startdate12 = pd.to_datetime("2019-07-31").date()
enddate12 = pd.to_datetime("2019-08-31").date()

subs_new.loc[(subs_new['Date'] >= startdate12)&(subs_new['Date'] < enddate12), 'adspend_month'] = '1'

startdate23 = pd.to_datetime("2019-08-31").date()
enddate23 = pd.to_datetime("2019-09-30").date()

subs_new.loc[(subs_new['Date'] >= startdate23)&(subs_new['Date'] < enddate23), 'adspend_month'] = '2'

startdate34 = pd.to_datetime("2019-09-30").date()
enddate34 = pd.to_datetime("2019-10-31").date()

subs_new.loc[(subs_new['Date'] >= startdate34)&(subs_new['Date'] < enddate34), 'adspend_month'] = '3'

startdate45 = pd.to_datetime("2019-10-31").date()
enddate45 = pd.to_datetime("2019-11-30").date()

subs_new.loc[(subs_new['Date'] >= startdate45)&(subs_new['Date'] < enddate45), 'adspend_month'] = '4'

startdate56 = pd.to_datetime("2019-11-30").date()
enddate56 = pd.to_datetime("2019-12-31").date()

subs_new.loc[(subs_new['Date'] >= startdate56)&(subs_new['Date'] < enddate56), 'adspend_month'] = '5'

startdate67 = pd.to_datetime("2019-12-31").date()
enddate67 = pd.to_datetime("2020-01-31").date()

subs_new.loc[(subs_new['Date'] >= startdate67)&(subs_new['Date'] < enddate67), 'adspend_month'] = '6'

startdate78 = pd.to_datetime("2020-01-31").date()
enddate78 = pd.to_datetime("2020-02-29").date()

subs_new.loc[(subs_new['Date'] >= startdate78)&(subs_new['Date'] < enddate78), 'adspend_month'] = '7'

startdate89 = pd.to_datetime("2020-02-29").date()
enddate89 = pd.to_datetime("2020-03-31").date()

subs_new.loc[(subs_new['Date'] >= startdate89)&(subs_new['Date'] < enddate89), 'adspend_month'] = '8'


# In[326]:


adspend


# In[327]:


a = [0,1,2,3,4,5,6,7,8,9]
adspend['adspend_month'] = a


# In[328]:


adspend


# In[26]:


adspend.sum(axis=1)


# In[329]:


def attribution_calc_1(channel):
    if channel not in ['facebook','email','search','brand sem intent google','affiliate','email_blast','pinterest','referral']:
        return 'compensate'
    else:
        return channel


# In[330]:


def attribution_calc_2(row):
    if (row.attribution_calc == 'compensate') & (row.attribution_survey in  ['facebook','email','search','brand sem intent google','affiliate','email_blast','pinterest','referral']):
        label = row.attribution_survey
    else:
        label = row.attribution_technical
    return label


# In[331]:


subs_new['attribution_calc'] = subs_new['attribution_technical'].apply(attribution_calc_1)

subs_new['attribution_calc'] = subs_new.apply(attribution_calc_2,axis=1)


# In[332]:


subs_new.head()


# In[31]:


len(subs_new['subid'].unique().tolist())


# In[32]:


subs_new.attribution_calc.value_counts()


# In[333]:


q6 = '''select adspend_month, sum(case when attribution_calc = 'facebook' then 1 else 0 end) as facebook,
sum(case when attribution_calc = 'email' then 1 else 0 end) as email,
sum(case when attribution_calc = 'search' then 1 else 0 end) as search,
sum(case when attribution_calc = 'brand sem intent google' then 1 else 0 end) as brand_sem_intent_google,
sum(case when attribution_calc = 'affiliate' then 1 else 0 end) as affiliate,
sum(case when attribution_calc = 'email_blast' then 1 else 0 end) as email_blast,
sum(case when attribution_calc = 'pinterest' then 1 else 0 end) as pinterest,
sum(case when attribution_calc = 'referral' then 1 else 0 end) as referral
from subs_new
group by adspend_month
order by adspend_month'''

attrition_conversion = sqldf(q6)
attrition_conversion


# In[334]:


adspend


# In[335]:


adspend = adspend.rename(columns={"brand sem intent google": "brand_sem_intent_google"})


# In[336]:


# calaculate CAC for each of the paid channels and store it in a dataframe
cac = pd.DataFrame()

cac['Month'] = adspend['date']

for col in ['facebook', 'email', 'search', 'brand_sem_intent_google',
       'affiliate', 'email_blast', 'pinterest', 'referral']:
    cac[col] = adspend[col]/attrition_conversion[col]


# In[337]:


cac = cac.drop(9)


# In[338]:


average_cac = cac.mean()


# In[339]:


average_cac


# In[340]:


#pd.DataFrame({'email':sf.index, 'list':sf.values})
average_cac = pd.DataFrame({'channel':average_cac.index, 'average_cac':average_cac.values})


# In[341]:


average_cac


# In[42]:


average_cac.mean()


# # prediction model

# In[43]:


# build prediction model to calculate CLV of individual customer
# we need individual CAC values to calculate the CLV
# so we can only use customers whom we have average CAC value


# In[44]:


q7 = '''select *
from subs_new
inner join average_cac
on average_cac.channel = subs_new.attribution_calc'''

subs_new_withcac = sqldf(q7)
subs_new_withcac


# In[45]:


# from above table we see there are 183625 unique customers whom we have CAC value
len(subs_new_withcac.subid.unique().tolist())


# In[46]:


# now i want to build a prediction model where churn = customer moves to payment 1 from trial (payment 0)
# so i need to select the customers who have moved on to payment 1
q8 = '''select *
from subs_new_withcac
inner join csvc
on subs_new_withcac.subid = csvc.subid
where payment_period = 0 or payment_period = 1
'''

test = sqldf(q8)
test


# In[49]:


# having the above dataset, i want to select some data columns that I think will be useful
# (ie drop those i think are useless)
# then join the smaller set with the engagement dataset
# so that I can see who are more likely to churn
q9 = '''select subs.subid, subs.package_type, subs.num_weekly_services_utilized, subs.preferred_genre,
subs.intended_use, subs.weekly_consumption_hour, subs.num_ideal_streaming_services, subs.age, subs.male_TF, subs.country,
subs.op_sys, subs.plan_type, subs.creation_until_cancel_days, subs.refund_after_trial_TF, subs.payment_type, 
subs.cancel_before_trial_ends, subs.adspend_month, subs.attribution_calc, c.trial_completed_TF, c.billing_channel, 
c.payment_period
from subs_new_withcac as subs
inner join csvc as c
on subs.subid = c.subid
where payment_period = 0 or payment_period = 1
'''

temp = sqldf(q9)
temp


# In[50]:


#because the temp dataset has many columns, and repeat values due to payment periods 0 and 1, i want to 'aggregate'
#this value so that each customer is represented by 1 row

q10 = '''select subid, package_type, num_weekly_services_utilized, preferred_genre, intended_use, weekly_consumption_hour,
num_ideal_streaming_services, age, male_TF, country, op_sys, plan_type, creation_until_cancel_days, refund_after_trial_TF, 
payment_type, cancel_before_trial_ends, adspend_month, attribution_calc, trial_completed_TF, billing_channel, sum(payment_period)
from temp
group by subid, package_type, num_weekly_services_utilized, preferred_genre, intended_use, weekly_consumption_hour,
num_ideal_streaming_services, age, male_TF, country, op_sys, plan_type, creation_until_cancel_days, refund_after_trial_TF, 
payment_type, cancel_before_trial_ends, adspend_month, attribution_calc, trial_completed_TF, billing_channel
'''

selected_subjoincsvc = sqldf(q10)
selected_subjoincsvc


# In[51]:


selected_subjoincsvc.isna().any()


# In[52]:


engagement.head(50).sort_values('subid')


# In[53]:


engagement.payment_period.value_counts()


# In[54]:


# here, I want to get the engagement information of a customer's trial period to determine their activity level if they will churn
# i aggregated their activity using sum and average
# i also focus on payment period 0 to find out their trial period activity

q11 = '''select subid, count(date) as number, sum(app_opens) as sum_app_opens, sum(cust_service_mssgs) as sum_cust_service_mssgs,
sum(num_videos_completed) as sum_num_videos_completed, sum(num_videos_more_than_30_seconds) as sum_vid_more_30, sum(num_videos_rated) as sum_videos_rated,
sum(num_series_started) as sum_series_started, avg(app_opens) as avg_app_opens, avg(cust_service_mssgs) as avg_cust_service_mssgs,
avg(num_videos_completed) as avg_num_videos_completed, avg(num_videos_more_than_30_seconds) as avg_vid_more_30, avg(num_videos_rated) as avg_videos_rated,
avg(num_series_started) as avg_series_started
from engagement
where payment_period = 0
group by subid'''

engagement0_agg = sqldf(q11)
engagement0_agg


# In[55]:


# join the two datasets together to find the intersect of customers
# intersect of subscriber, customer service payment period 0 and 1, and engagement data
q12 = '''select * from selected_subjoincsvc inner join engagement0_agg on selected_subjoincsvc.subid = engagement0_agg.subid'''

df = sqldf(q12)
df


# In[56]:


df.refund_after_trial_TF.value_counts()


# In[57]:


temprefund = df[df['refund_after_trial_TF']==1]


# In[58]:


# using the columns I have, create the dependent variable y as churn
# defined by 1) customer not completing trial,
# or customer completed trial but asked for refund

# 0 = no churn
# 1 = churn

df['churn'] = 0
df.loc[df['trial_completed_TF'] == 0,'churn'] = 1
df.loc[(df['refund_after_trial_TF'] == 1)&(df['trial_completed_TF'] == 1), 'churn'] = 1


# In[59]:


df


# In[60]:


# after I created churn feature, i want to drop the variables that are directly correlated to churn, and other variables which i think
# is not important

df = df.drop(['refund_after_trial_TF','cancel_before_trial_ends','trial_completed_TF','creation_until_cancel_days','adspend_month','billing_channel','sum(payment_period)'],axis=1)


# In[61]:


df


# In[62]:


# here, I remove the duplicate df.subid column due to sql joins
df = df.loc[:,~df.columns.duplicated()]


# In[63]:


df


# In[64]:


copy = df.copy()


# In[65]:


# chech which columns have missing values
df.isna().any()


# In[67]:


import numpy as np
# fill in some of the missing values in categorical columns with proportional sampling 
def proportional_fillna(df,col):
    s = df[col].value_counts(normalize=True)
    missing = df[col].isnull()
    df.loc[missing,col] = np.random.choice(s.index, size=len(df[missing]),p=s.values)
    return df

# apply the proportional fillna function
proportional_fillna(df,'package_type')
proportional_fillna(df,'preferred_genre')
proportional_fillna(df,'intended_use')
proportional_fillna(df,'op_sys')
proportional_fillna(df,'payment_type')


# In[68]:


df.isna().any()


# In[69]:


df.num_weekly_services_utilized.describe()


# In[70]:


df.num_weekly_services_utilized.isna().sum()/109161


# In[71]:


df.weekly_consumption_hour.isna().sum()/109161


# In[72]:


df.num_ideal_streaming_services.isna().sum()/109161


# In[73]:


df.age.isna().sum()/109161


# In[74]:


# i decide to drop num weekly services because the number of missing values is quite high
df_final = df.drop(['num_weekly_services_utilized','num_ideal_streaming_services'],axis=1)


# In[75]:


copy = df_final.copy()


# In[76]:


# now my columns with missing values left with weekly consumption hour and age

df_final.weekly_consumption_hour.describe()


# In[77]:


#len(df[(df['A']>0) & (df['B']>0) & (df['C']>0)])
len(df_final[df_final['weekly_consumption_hour']<=0])


# In[78]:


len(df_final.subid.unique().tolist())


# In[79]:


# drop the specific columns with weekly consumption hours with negative values

indexNames = df_final[df_final['weekly_consumption_hour']<0].index
df_final.drop(indexNames, inplace=True)


# In[80]:


df_final


# In[81]:


copy = df_final.copy()


# In[82]:


df_final.weekly_consumption_hour.describe()


# In[83]:


df_final = df_final.fillna({'weekly_consumption_hour': 27.8300})


# In[84]:


df_final.isna().any()


# In[85]:


#age missing values
df_final.age.max()


# In[87]:


import numpy as np
import matplotlib.pyplot as plt
df_final.hist(column='age')


# In[88]:


# count number of people with reasonable age
len(df_final[(df_final['age']>10)&(df_final['age']<=100)])


# In[89]:


temptemp = df_final[(df_final['age']>10)&(df_final['age']<=100)]


# In[342]:


temptemp.head()


# In[90]:


temptemp.shape


# In[91]:


temptemp.age.describe()


# In[92]:


df_final = df_final.fillna({'age': 45})


# In[93]:


df_final.shape


# In[94]:


len(df_final[(df_final['age']>10)&(df_final['age']<=100)])


# In[95]:


109152 - 108802


# In[96]:


# we observe about 350 customers with unreasonable ages
# remove them from df_final as number is small


# In[97]:


copy = df_final.copy()


# In[ ]:


Python
# Get names of indexes for which column Age has value 30
indexNames = dfObj[ dfObj['Age'] == 30 ].index
# Delete these row indexes from dataFrame
dfObj.drop(indexNames , inplace=True)
1
2
3
4
5
# Get names of indexes for which column Age has value 30
indexNames = dfObj[ dfObj['Age'] == 30 ].index
 
# Delete these row indexes from dataFrame
dfObj.drop(indexNames , inplace=True)


# In[98]:


indexNames = df_final[df_final['age'] < 10].index
df_final.drop(indexNames, inplace=True)


# In[99]:


df_final.shape


# In[100]:


copy = df_final.copy()


# In[101]:


indexNames = df_final[df_final['age'] > 100].index
df_final.drop(indexNames, inplace=True)


# In[102]:


df_final.shape


# In[103]:


copy = df_final.copy()


# In[104]:


df_final.isna().any()


# In[105]:


# here we have replaced all missing values from the dataset
df_final


# In[106]:


df_final.reset_index(drop=True, inplace=True)


# In[107]:


df_final


# In[108]:


df_final_copy = df_final.copy()


# # FINALLY I BUILD MY PREDICTION MODEL WAHHHHH

# In[ ]:


# Separating with continuous and categorical variables. 
# 'Customer_id','Cust_Segment'
#cat = ['PBK_Ind', 'OccuCode', 'Gender','IncomeLevel', 'Marital_Status',
#       'Number_Children', 'Education_Level']
#num = ['age','Home_Ownership','FirstContactDay_re','Recency', 'Frequency', 'Monetary', 'pctg_online',
#       'pctg_overseas', 'pctg_alipay', 'pctg_tenpay', 'pctg_applepay',
#       '1.spending', '2.cash advance', '3.installment', '4.repayment', '5.fee',
#       '6.interest', '7.others', '8.Cash rebate']
#X_num = df[num]
#X_cat = df[cat]

# Creating dummy variable dataframe from categorical variables.
#df_X = X_num.join(pd.get_dummies(X_cat))


# # LOGISTIC REGRESSION

# In[124]:


# I need to separate continuous and categorical variables


# In[125]:


df_final.country.value_counts()


# In[126]:


df_final = df_final.drop(['country'],axis=1)


# In[127]:


cat = ['package_type','preferred_genre','intended_use','op_sys','plan_type','payment_type','attribution_calc']
num = ['weekly_consumption_hour', 'age', 'male_TF','number', 'sum_app_opens','sum_cust_service_mssgs', 
       'sum_num_videos_completed', 'sum_vid_more_30','sum_videos_rated', 'sum_series_started', 'avg_app_opens',
       'avg_cust_service_mssgs', 'avg_num_videos_completed', 'avg_vid_more_30','avg_videos_rated', 'avg_series_started']
X_num = df_final[num]
X_cat = df_final[cat]

#creating dummy variable dataframe
df_X = X_num.join(pd.get_dummies(X_cat))


# In[128]:


df_X


# In[129]:


df_y = df_final['churn']


# In[130]:


from sklearn.model_selection import train_test_split
X = df_X
y = df_y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 0)


# In[131]:


# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train[num])
X_test_scale = scaler.transform(X_test[num])


# In[132]:


# feature scaling
X_train[num]= X_train_scale
X_test[num] = X_test_scale


# In[133]:


# feature selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 15)
fit = rfe.fit(X_train, y_train)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))


# In[134]:


X_test.columns


# In[135]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

logmodel = LogisticRegression(C=0.5, max_iter=120, penalty='l1')
logmodel.fit(X_train,y_train)
pred_log = logmodel.predict(X_test)
prob_log = logmodel.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test,pred_log))
print(classification_report(y_test,pred_log))
print('Test accuracy:{:.2f}'.format(logmodel.score(X_test, y_test)))
print('Training accuracy:', logmodel.score(X_train, y_train))
print('Coefficient of each feature:', logmodel.coef_)

roc_value = roc_auc_score(y_test, prob_log)
fpr, tpr, thresholds = roc_curve(y_test,prob_log)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_value)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# # CATBOOST

# In[136]:


forcatboost = df_final


# In[137]:


forcatboost


# In[138]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier


# In[139]:


X1 = forcatboost.drop(['subid','churn'],axis=1)
y1 = forcatboost['churn']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=.3, random_state = 0)


# In[140]:


categorical_features_indices = np.where(X1.dtypes != np.float)[0]


# In[141]:


model = CatBoostClassifier()
model.fit(X_train1, y_train1,cat_features=categorical_features_indices,eval_set=(X_test1, y_test1))


# In[142]:


preds = model.predict(X_test1)


# In[143]:


preds


# In[144]:


pred_proba = model.predict_proba(X_test1)


# In[148]:


print(pred_proba)


# In[146]:


print('Accuracy of CatBoost classifier on training set: {:.2f}'
     .format(model.score(X_train1, y_train1)))
print('Accuracy of CatBoost classifier on test set: {:.2f}'
     .format(model.score(X_test1, y_test1)))


# In[147]:


from sklearn import metrics
model_matrix = metrics.confusion_matrix(y_test1, preds)
print(model_matrix)
print(metrics.f1_score(y_test1,preds))


# In[149]:


model.get_feature_importance()


# In[150]:


X1.columns


# In[151]:


# from the above feature importance, we see that some columns might not be important in determining churn or not
# hence we set up a new model without these features


# In[231]:


forcatboost2 = df_final.copy()


# In[197]:


forcatboost2


# In[153]:


X2 = forcatboost2.drop(['subid','churn','plan_type','sum_app_opens','sum_videos_rated','sum_series_started','avg_videos_rated'],axis=1)
y2 = forcatboost2['churn']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=.3, random_state = 0)


# In[154]:


categorical_features_indices2 = np.where(X2.dtypes != np.float)[0]


# In[155]:


model2 = CatBoostClassifier()
model2.fit(X_train2, y_train2,cat_features=categorical_features_indices2,eval_set=(X_test2, y_test2))


# In[156]:


print('Accuracy of CatBoost classifier on training set: {:.2f}'
     .format(model2.score(X_train2, y_train2)))
print('Accuracy of CatBoost classifier on test set: {:.2f}'
     .format(model2.score(X_test2, y_test2)))


# In[157]:


model2.get_feature_importance()


# In[158]:


X2.columns


# In[183]:


X_test2.shape


# In[184]:


y_test2.shape


# In[185]:


X_test2.head()


# In[186]:


y_test2.head()


# In[187]:


# I want to check the prediction confusion matrix under different thresholds, so i want to use the test set results


# In[215]:


model_evaluation = X_test2.copy()


# In[216]:


model_evaluation.head()


# In[217]:


model_evaluation['actual_churn_result'] = y_test2


# In[218]:


model_evaluation


# In[219]:


model_evaluation['copy_index']=model_evaluation.index


# In[220]:


model_evaluation


# In[221]:


model2.predict(X_test2)


# In[222]:


pred_proba2 = model2.predict_proba(X_test2)
pred_proba2


# In[223]:


model = model_evaluation.reset_index()


# In[224]:


proba_temp = pd.DataFrame(data = pred_proba2, columns=['prob_convert','prob_churn'])


# In[225]:


proba_temp


# In[226]:


model_eval_join_prob = pd.concat([model,proba_temp],axis=1)
model_eval_join_prob


# In[227]:


forcatboost2.head()


# In[232]:


forcatboost2['index_copy']=forcatboost2.index


# In[235]:


for column in model_eval_join_prob.columns:
    print(column)


# In[237]:


q15 = '''select m.*, f.subid
from model_eval_join_prob as m
inner join forcatboost2 as f
on f.index_copy = m.copy_index'''

final_model = sqldf(q15)
final_model

#model_eval_final = forcatboost2[['subid']].merge(model_eval_join_prob,left_index=True,right_index='index')


# In[238]:


q16 = '''select final_model.*, subs_new_withcac.average_cac
from final_model
inner join subs_new_withcac
on final_model.subid = subs_new_withcac.subid'''

final_results = sqldf(q16)
final_results


# In[239]:


final_table = final_results.drop(['index','copy_index','prob_convert'],axis=1)


# # setting different threshold values

# In[241]:


final_table['threshold50_prediction'] = 0

final_table.loc[final_table['prob_churn'] >= 0.50,'threshold50_prediction'] = 1


# In[242]:


final_table


# In[243]:


final_table['threshold66_prediction'] = 0

final_table.loc[final_table['prob_churn'] >= 0.667,'threshold66_prediction'] = 1


# In[244]:


final_table['threshold90_prediction'] = 0

final_table.loc[final_table['prob_churn'] >= 0.90,'threshold90_prediction'] = 1


# In[245]:


final_table


# In[246]:


final_table.to_csv("final_table.csv",index=False)


# # confusion matrix

# In[257]:


q17 = '''select actual_churn_result, sum(case when threshold50_prediction = 0 then 1 else 0 end) as 'predict_0', sum(case when threshold50_prediction = 1 then 1 else 0 end) as 'predict_1'
from final_table
group by actual_churn_result'''

threshold50 = sqldf(q17)
threshold50


# In[249]:


q18 = '''select actual_churn_result, sum(case when threshold66_prediction = 0 then 1 else 0 end) as '0', sum(case when threshold66_prediction = 1 then 1 else 0 end) as '1'
from final_table
group by actual_churn_result'''

threshold66 = sqldf(q18)
threshold66


# In[250]:


q19 = '''select actual_churn_result, sum(case when threshold90_prediction = 0 then 1 else 0 end) as '0', sum(case when threshold90_prediction = 1 then 1 else 0 end) as '1'
from final_table
group by actual_churn_result'''

threshold90 = sqldf(q19)
threshold90


# In[ ]:





# # clv

# In[258]:


final_table


# In[259]:


# attach discount price to final table

q20 = '''select final_table.*, subs_new_withcac.discount_price
from final_table
left join subs_new_withcac
on final_table.subid = subs_new_withcac.subid'''

final_table_withprice = sqldf(q20)
final_table_withprice


# In[260]:


final_table_withprice.discount_price.value_counts()


# In[261]:


final_table_withprice['payment'] = final_table_withprice['discount_price'] *4


# In[262]:


final_table_withprice.head()


# In[263]:


final_table_withprice_prenewal = final_table_withprice.copy()


# In[264]:


final_table_withprice_prenewal['prob_renewal'] = 1 - final_table_withprice_prenewal['prob_churn']


# In[265]:


final_table_withprice_prenewal.head()


# In[266]:


final_table_forclv = final_table_withprice_prenewal.copy()


# In[299]:


temp = final_table_withprice_prenewal.copy()


# In[300]:


temp['multiply'] = 1.0333 / (1.0333 - temp['prob_renewal'])


# In[301]:


temp.head()


# In[302]:


temp['clv'] = temp['payment']*temp['multiply'] - temp['average_cac']


# In[303]:


temp.head()


# In[304]:


final_table_withclv = temp.copy()


# In[305]:


final_table_withclv.hist(column='clv')


# In[306]:


final_table_withclv.clv.describe()


# In[307]:


final_table_withclv[final_table_withclv['clv']<0].shape


# In[308]:


final_table_withclv['actual_churn_result'].value_counts()


# In[292]:


a = ['package_type', 'preferred_genre', 'intended_use','weekly_consumption_hour', 'age', 'male_TF', 'op_sys', 'payment_type','attribution_calc', 'number', 'sum_cust_service_mssgs','sum_num_videos_completed', 'sum_vid_more_30', 'avg_app_opens','avg_cust_service_mssgs', 'avg_num_videos_completed', 'avg_vid_more_30','avg_series_started']


# In[293]:


featimp = pd.DataFrame(a,columns=['features_used'])


# In[294]:


featimp


# In[295]:


b = [ 1.15794103,  1.37477094,  2.79604379, 27.96525905,  7.26360823,1.25938928,  2.10811148,  1.9827975 ,  2.30884875, 11.33741931,9.25362194,  2.3886531 , 11.04706171,  3.63975918,  4.47700531,3.18214397,  4.76963204,  1.68793341]


# In[296]:


featimp['feat_importance'] = b


# In[297]:


featimp


# In[310]:


final_table_withclv[(final_table_withclv['clv']>0)&(final_table_withclv['clv']<25)].shape


# In[344]:


subs.male_TF.value_counts()


# In[345]:


subs.shape


# In[346]:


subs.preferred_genre.value_counts()


# In[ ]:




