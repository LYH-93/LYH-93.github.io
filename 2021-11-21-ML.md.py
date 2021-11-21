# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:34:46 2021

@author: 피파허접
"""
import pandas as pd 
import os 
import matplotlib.pyplot as plt # 별칭 \
import numpy as np 
from pandas import DataFrame as df
import plotly.express as px
import seaborn as sns
from scipy import stats
import sklearn
import scipy.optimize as op
from sklearn.svm import SVR
from scipy import stats

# 파일 불러오기
os.chdir('C:\\ITWILL\\4_Python-II/data')
st = pd.read_csv('car_final_3.csv',  encoding='utf-8')
st.info()
# 전기차 불러오기
ev = st[st['Fuel'] == 'Electric']


# 특정칼럼만 불러오기
ev1 = ev.iloc[:, [1,4,5,6,10,13,14,15,20,21]]
ev1.info()
print(ev1.columns)

# ----- 형변환 및 표준화 진행 -----

# 단위 뒤 공백 발견 -> 제거 조치
ev1['Batterytype'] = ev1['Battery_Type'].str.replace("[ ]|[  ]","")

# MPGE 단위 지우고 형 변환
ev1['Fuelef'] = ev1['FuelEficiency'].str.replace("[A-Z]","")
ev1["Fuelef"]=pd.to_numeric(ev1['Fuelef'], errors='coerce')

ev1["BatteryCAPA"]=pd.to_numeric(ev1['Battery_capacity'], errors='coerce')
ev1.info()
print(ev1.columns)
evmap = ev1[['Manufacturer', 'Model_Year', 'CarPrice',
       'Total_Power', '240v_Charge Time',
       'Category', 'Zero(s)', 'Batterytype', 'Fuelef', 'BatteryCAPA']]

realev= ev1[['Model_Year','CarPrice','BatteryCAPA','Total_Power','Fuelef','240v_Charge Time','Zero(s)']]


############################## 정규화 전 ###################################
import pandas as pd # csv file 읽기
from sklearn.linear_model import LinearRegression # model 생성
from sklearn.metrics import mean_squared_error, r2_score 
# ㄴ> model 평가도구 제공, 평가점수
from sklearn.model_selection import train_test_split 
# ㄴ> 훈련셋, 검정셋을 나눠주기
from sklearn.preprocessing import minmax_scale # 정규화(0~1)-x변수
from scipy.stats import zscore # 표준화(mu=0,std=1)-y변수

from xgboost import XGBRegressor #  Tree model 
import xgboost as xgb
from xgboost import plot_importance # 중요변수 시각화 
from sklearn.model_selection import train_test_split # split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import squarederror


X, y = realev.iloc[:,[0,2,3,4,5,6]],realev.iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

type(X)
type(y)


X
y.describe()
y.plot()


df = X.copy()
df['y'] = y
df.info()

new_df = df[df.y <= 10000]
new_df.info()

cols = list(new_df.columns)

train, test = train_test_split(new_df, test_size = 0.3)



obj = xgb.XGBRegressor()
model = obj.fit(train[cols[:-1]], train['y'])


y_pred = model.predict(test[cols[:-1]])

score = r2_score(test['y'], y_pred)
print(score)


################## 정규화 후 ################################
new_df = df[df.y <= 10000] # 이상치 제거 
new_df.info()


corr = new_df.corr() # 상관계수 
corr['y']




'''
Model_Year          0.298161 : 제거 
BatteryCAPA         0.720561
Total_Power         0.752223
Fuelef             -0.315941 : 제거 
240v_Charge Time    0.297092 : 제거 
Zero(s)            -0.621350 : 제거 
''' 

# NEW df 
new_df = new_df[['BatteryCAPA','Total_Power','y']]
cols = list(new_df.columns)


# X,Y 정규화 
new_arr = minmax_scale(new_df)

X = new_arr[:, :-1]
y = new_arr[:, -1]

# 70:30 split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# model
obj = xgb.XGBRegressor()
model = obj.fit(X_train, y_train)

# train set
model.score(X_train, y_train) # 0.9985782303212807


# 예측치 
y_pred = model.predict(X_test)

# 평가 
scoreXGB = r2_score(y_test, y_pred)
print(scoreXGB)






################### 히트맵 시각화 ##########################
evmap = ev1[['Manufacturer', 'Model_Year', 'CarPrice',
       'Total_Power', '240v_Charge Time',
       'Category', 'Zero(s)', 'Batterytype', 'Fuelef', 'BatteryCAPA']]
cols1 = ['CarPrice','Model_Year','BatteryCAPA','Total_Power','Fuelef','240v_Charge Time','Zero(s)']
cols = [ 'Model_Year',
       'Total_Power', '240v_Charge Time',
       'Zero(s)',  'Fuelef', 'BatteryCAPA','CarPrice']
cols_view = ['M_Y','T_P','240','zero','F_E','B_C','C_P']
sns.set(font_scale=1.5)
# corr에 칼럼을 넣어서 셋팅
corr = evmap[cols].corr(method='pearson')
# 이제 히트맵 Run 시킨다.
hm = sns.heatmap(corr.values,
                 cbar = True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size':15},
                 yticklabels=cols_view,
                 xticklabels=cols_view)
plt.tight_layout()
plt.show()


##########################################################
# 랜덤 포레스트 다시 돌려보기

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import os # file path
from sklearn.ensemble import RandomForestRegressor # model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split # split
from xgboost import plot_importance 

from sklearn.preprocessing import minmax_scale # 정규화(0~1)-x변수
from scipy.stats import zscore # 표준화(mu=0,std=1)-y변수

rf = RandomForestRegressor()

model_rf = rf.fit(X, y)

y_pred = model_rf.predict(X_test) # 예측치
len(y_pred) # 16

R2_score = r2_score(y_test, y_pred)
print('R2_score', R2_score) # R2_score 0.9448277018550238


print('중요변수 점수:', model_rf.feature_importances_)

# 과적합 여부
model_rf.score(X_train, y_train) # 0.9149268734951215


def importances_plot(model) :
    x_size = 2 # x변수 갯수
    plt.barh(y=range(x_size), width=model_rf.feature_importances_)
    # y축 눈금 - feature_names
    plt.yticks(range(x_size), X)
    plt.xlabel('importances')
    plt.show()

importances_plot(model_rf)

##########################################################
# 다중회귀분석
from statsmodels.formula.api import ols # 다중회귀분석 model 생성
import os
import numpy as np
lr = LinearRegression()
model = lr.fit(X=X_train, y=y_train)

# ['FuelEficiency_MPGE','Engine_Size_L','Total_Power_kW','Battery_capacity_kWh','Charge_Time']기울기
# [-0.17318697, -1.13973356,  5.63659513, -0.48954452,  0.50306904]

##### 5. model 평가 : test set 이용 (x,y변수의 정규화 여부 확인필요)
# 둘다 정규화되어있기 때문에 MSE 이용할수 있음
y_pred = model.predict(X = X_test)
# y_true = y_test 미리 지정을 해놓아도 되고..

mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
print('mse =', mse) #  mse = 0.024881312053091936작을수록 실제와 예측이 적은것

score = r2_score(y_true=y_test, y_pred=y_pred)
print(score) # 0.6355530486971175

# 과적합 여부 확인
model.score(X_train, y_train) # 0.5800994491142045
model.score(X_test, y_test) # 0.6355530486971175
# 통계적 의미성
print(model.summary(X_test,y_test))
###########################################################
print(new_df)

formula = 'y ~ BatteryCAPA + Total_Power'
obj = ols( formula = formula , data = new_df)
model = obj.fit()
print(model.summary())



###########################################################
EV_score_RF = 0.976892340375510
EV_score_lr = 0.6355530486971175
EV_score_XGB = 0.8057302221944355

HEV_score_RF = 0.851049
HEV_score_lr = 0.6130441645235152
HEV_score_XGB = 0.6828660443993657

PHEV_score_RF = 0.8786
PHEV_score_lr = 0.6423
PHEV_score_XGB = 0.8680

sns.lmplot(x='Model_Year',y='CarPrice',ci=95,data=realev,hue='Batterytype')
plt.show()

R2_score_xg = [EV_score_XGB, HEV_score_XGB, PHEV_score_XGB ]
R2_score_lr = [EV_score_lr, HEV_score_lr, PHEV_score_lr ]
R2_score_rf = [EV_score_RF, HEV_score_RF, PHEV_score_RF ]
label = ['EV', 'HEV', 'PHEV']

plt.plot(label, R2_score_xg)
plt.plot(label, R2_score_lr)
plt.plot(label, R2_score_rf)
plt.xlabel('car_category')
plt.ylabel('Score')
plt.title('score Compare')
plt.legend(['XGBoost', 'LinearRegression', 'RandomForest'])
plt.show()



