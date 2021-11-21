# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""-------------------------------------------------------------------------------------------------------------------------------------------------------------- """

"""
자료구조 : 1. sequence 자료구조 (순서있음)
          2. 딕셔너리, 세트 (순서없음)

인덱싱 : 리스트에서 하나의 요소를 인덱스 연산자를 통하여 참조(접근)하는 것

슬라이싱 : 리스트 안에서 범위를 지정하여서 원하는 요소들을 선택하는 연산

리스트 : 여러 개의 데이터가 저장되어 있는 자료구조
리스트가 필요한 이유 : 여러 개의 데이터가 저장되어 있는 자료구조
"""

# 문자열 인덱싱 & 슬라이싱

text = "IT Will is power."
print(text[:-2]) # IT Will is powe
print(text[:8], text[-1])

# 인덱싱
flist = ["apple", "banana", "tomato", "peach", "pear" ]
print(flist[0], flist[3], flist[-1])

# 슬라이싱
a = [0,1,4,9,16,25,36,49]
a[3:6]

# 리스트 (append로 추가해서 리스트를 만들어보는 것)

scores = [ ]
for i in range(10):
    scores.append(int(input("성적을 입력하시오:")))
print(scores)

scores[0] = 80

scores[i] = 10;
scores[i+2] = 20;

# 리스트의 요소갯수만큼 리스트가 반복되어 출력
for element in scores:
    print(scores)

# 복잡한 리스트
list1 = [12,"dog",180.14] # 혼합자료형
list2 = [["Seoul", 10], ["Paris", 12], ["London", 50]] # 내장 리스트

# 리스트 기초연산
marvel_heroes = [ "스파이더맨", "헐크", "아이언맨" ]
dc_heroes = [ "슈퍼맨", "배트맨", "원더우먼" ]

heros = marvel_heroes + dc_heroes
heros
# 리스트에 곱하기
values = [1,2,3]*3
values # [1, 2, 3, 1, 2, 3, 1, 2, 3]
len(values)
 
# 요소추가하기
developteam = []
developteam.append("이유현")
developteam.append("성민승")
developteam.append("김동완")

developteam

# 리스트 안의 내용 조회
if "이유현" in developteam:
    print("내부인원")
    
# 리스트 인덱스 확인
developteam.index("이유현") # 0
developteam.index("김동완") # 2

# 리스트 최소값 최대값
values = [ 100, 20, 31, 45, 15, 6, 7, 8, 9, 10 ]
min(values)
max(values)

# 리스트에서 sort 쓰기 정렬~~~
values.sort()
value2=sorted(values)
print(value2)

# 리스트 내의 조건문
list1 = [3,4,5]
list2 = [x*2 for x in list1]
print(list2)

# 2차원 리스트
s = [ 
[ 1, 2, 3, 4, 5 ] ,
[ 6, 7, 8, 9, 10 ], 
[11, 12, 13, 14, 15 ] 
]

print(s)

# 동적으로 2차원 리스트 생성

rows = 3
cols = 5

s = []
for row in range(rows):
    s += [[3]*cols]
    
print("s=",s) # s= [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]]

rows = len(s)
cols = len(s[0])
cols
for r in range(rows):
    for c in range(cols):
        print(s[r][c], end=",")
    print()

"""
    tuple!! 
    튜플은 변경될 수 없는 리스트 + 순서가 없다!!
    
    tuple('listname') 
    : 리스트를 튜플로 변경한다.
"""
# tuple을 변경하려고 해보자

t1 = (1,2,3,4,5);
t2 = (1,2,3,4,5);

t1[0] =100; # TypeError: 'tuple' object does not support item assignment

# 튜플 대입 연산
student1 = ("철수",19,"CS")
(name,age,major) = student1
name # '철수'

"""
    set!!
    세트는 중복되지 않은 항목들이 모인것 + 순서가 없다!!
    
"""

numbers = {1,2,2,3,3,3,4}

numbers # {1, 2, 3, 4}

# 요소 추가
numbers.add(5)


"""

    dictionary
    딕셔너리는 키(key)와 값(value)의 쌍을 저장할 수 있는 객체
    

"""

# 형태
dictionary = {'name':'이유현','phone':'01091597160','score':100}
dictionary['score']
# 추가하기

dictionary['speed'] = '1000'
print(dictionary)

# 항목 순회하며 출력하기.

for item in dictionary.items():
    print(item)
"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    plot 문제로 정리하기
    1. matplotlib
        1) 한글 및 음수 부호 지원
        2) 기본차트 시각화 - 기본 선 스타일과 색상, x축 y축 스타일&색상, color와 marker 이용
        3) 산점도, 히스토그램, 상자그래프
        3) 이산형 변수 시각화 - 가로막대 그래프, 세로막대 차트, 원차트
        4) subplot 차트
        5) 시계열차트
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

data3= np.random.randn(50) # 난수
data4= np.random.randn(50).cumsum() # 난수

chart.plot(data3, color='r', label='step', 
drawstyle="steps-post")

chart.plot(data4, color='g', label='line')
plt.legend(loc='best')
plt.ylabel('y label')
plt.xlabel('x label')
plt.title('chart title')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sn

# 문3) seaborn의  titanic 데이터셋을 이용하여 다음과 같이 단계별로 시각화하시오.
titanic = sn.load_dataset('titanic')
print(titanic.info())

#  <단계1> 'total_bill','tip','sex','size' 칼럼으로 서브셋 만들기  
titanic_df = titanic[['survived','pclass', 'age','fare']]
print(titanic_df.info())
#sn.pairplot(data=DataFrame, hue='집단변수', kind='scatter')

# <단계2> 성별(sex) 칼럼을 집단변수로 산점도행렬 시각화
sn.pairplot(data=titanic_df, hue='survived')
plt.show()

# <단계3> 산점도행렬의 시각화 결과 해설하기
'''
pclass : 3등석 일수록 사망비율 매우 높음 
pclass vs fare : 1등석 일수록 고 요금 
age : 25~50세 사이에서 가장 높은 빈도, 사망과 생존 비율 비슷 
age vs fare : 대체적으로 나이가 많고, 요금이 낮은 경우 사망 비율 높음
fare : 비용이 저렴한 경우가 상대적으로 많은 분포
fare vs age : 대체적으로 요금이 낮고, 나이가 50대 이상인 경우 사망 비율 높음    
'''

# survived vs age    
# 연령대 생존비율 : 20~40 
titanic[titanic['survived'] == 1].age.plot(kind = 'hist', color = 'blue')
# 연령대 사망비율 : 20~40
titanic[titanic['survived'] == 0].age.plot(kind = 'hist', color = 'green')


# 문4) seaborn의 tips 데이터셋을 이용하여 다음과 같이 단계별로 시각화하시오.
tips = sn.load_dataset('tips')
print(tips.info())

# <단계1> 'total_bill','tip','sex','size' 칼럼으로 서브셋 만들기
tips_df = tips[['total_bill','tip','sex','size']]

# <단계2> 성별(sex) 칼럼을 집단변수로 산점도행렬 시각화 
sn.pairplot(data=tips_df, hue='sex')
plt.show()

# <단계3> 산점도행렬의 시각화 결과 해설하기
"""
total_bill : 총금액 15~20 사이가 가장 높은 빈도, 금액이 클 수록 남자 지불  
total_bill vs tip  : 대체적으로 비례관계, 총금액과 팁이 많은 경우 남자 지불
tip : 팁은 1~5 사이가 가장 높은 빈도, 팁 금액이 클 수록 남자 지불 
total_bill vs size : 행사규모가 작은 경우 여성 지불
size : 행사규모 2가 가장 높은 빈도, 특히 4일때 남자 지불 
size vs total_bill : 대체적으로 비례관계, 규모가 큰 경우 총금액이 많음  

"""
import pandas as pd # object
import numpy as np # dataset
import matplotlib.pyplot as plt # plt.show() 

# 1. 기본 차트 시각화 
ser = pd.Series(np.random.randn(10)) # 1d 
print(ser)

# 1차원 객체 : 기본차트 - 선 그래프 
ser.plot(color='g')
plt.show()

# 2차원 객체 
df = pd.DataFrame(np.random.randn(10, 4),
                  columns=('one','two','three','fore'))

print(df)

# 기본차트 : 선 그래프
df.plot()  
plt.show()

# 막대차트 : 세로 
df.plot(kind='bar', title = 'bar chart')
plt.show()


# 막대차트 : 가로 
df.plot(kind='barh', title = 'bar chart')
plt.show()


# 막대차트 : 가로, 누적형  
df.plot(kind='barh', title = 'barh chart', stacked=True)
plt.show() 


# 2. dataset 이용 
import os

os.chdir('C:/ITWILL/4_Python-II/data')
tips = pd.read_csv('tips.csv')
print(tips.info())

# 교차분할표 : 집단변수 이용 
# 요일(day):행 vs 규모(size):열
tips['day'].unique() # ['Sun', 'Sat', 'Thur', 'Fri']
tips['size'].unique()# [2, 3, 4, 1, 6, 5]

tab = pd.crosstab(index=tips['day'], columns=tips['size'])
print(tab)

# 테이블 정보 
tab.shape # (4, 6)
tab.index # 행 이름 
tab.columns # 열 이름 

#tab.index = 수정 이름 
type(tab) # pandas.core.frame.DataFrame

help(tab.plot)

# size : 1, 6 제외 -> subset
#obj.loc[행, 열]
new_tab = tab.loc[:, 2:5]
print(new_tab)


new_tab.plot(kind='barh', stacked=True,
         title = 'day and size')
plt.show()



"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot 예제 끝~~
"""

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

1. Group by : 머 대강 칼럼들을 특정 변수의 조건에 맞게 그룹화 시키는거다
2. apply :
3. Pivot table ---> df.pivot(index='x',columns='y',values='z')
        - 좌측 index(행)를 어떤 칼럼으로 설정할지
        - 칼럼(열)쪽을 y칼럼으로 세우고
        - 안에 채울 values 값 z로 설정


---------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

"""

NUMPY!!!!!!!!!!!!!
    1. 배열 생성 array()
    - arange() : 배열 객체 반환
    - linspace() : 시작점과 끝점을 균일 간격으로 나눈 점들을 생성
    - reshape() : 행수와 열수를 조절
    2. 특수 행렬 생성 zeros() : 0으로 채워진 배열, ones() : 1로 채워진 배열 ,eye() : 항등행렬
    
    3. 난수 생성  numpy.random.normal(size = 개수)
    4. 산술연산 

"""










"""
# Series Pandas의 Series는 1차원 데이터를 다루는 데 효과적인 자료구조
values 속성을 호출하면 데이터의 배열 원소가 리턴
index 속성을 호출하면 인덱스의 정보가 리턴
각각의 데이터는 [인덱스]를 이용해서 접근이 가능
numpy 함수 사용 가능
dict 객체로 생성 가능
"""
# code 1

from pandas import Series, DataFrame
import numpy as np
import pandas as pd

price = Series([4000, 3000, 3500, 2000])
print(price)
print(price.index)
print(price.values)
print('====================')
fruit = Series([4000, 3000, 3500, 2000], 
index=['apple', 'mellon','orange', 'kiwi'])
print(fruit)
print(fruit[0]) # 순번 이용 데이터 접근
print(fruit['apple']) # 인덱스 이용 데이터 접근
print(fruit[fruit>3000]) # 부울리언 식
print("=================")

# code 2
from pandas import Series, DataFrame
import numpy as np
import pandas as pd

good1 = Series([4000,3500,None,2000],index = ['apple','mango','orange','kiwi'])

good2 = Series([3000,3000,3500,2000],index = ['apple','mango','orange','kiwi'])

print(pd.isnull(good1)) # NaN값 검출 none이면 True를 반환
print (good1+good2)


# DataFrame

"""
# DataFrame은 행과 열로 구성된 2차원 데이터를 다루는 데 효과적인 자료구조
일반적으로 딕셔너리 활용해서 생성
입력가능한 데이터
1. 2차원 ndarray
2. 리스트, 튜플,dict, Series의 dict
3. dict, Series의 list
4. 리스트, 튜플의 리스트
5. 칼럼 뽑는 방법
    1) data.iloc[[행]],[열]  # 행은 index

"""
# code 1
from pandas import Series, DataFrame

items = {'code': [1,2,3,4,5,6],
'name': ['apple','watermelon','oriental melon', 'banana', 'lemon', 'mango'],
'manufacture': ['korea', 'korea', 'korea','philippines','korea', 'taiwan'],
'price':[1500, 15000,1000,500,1500,700]}

data = DataFrame(items)
print(data)

# 특정 컬럼만 뽑기
data1 = DataFrame(items, columns = ['code', 'price'])
print(data1)
print(data.loc[0]) # 0 행 출력 가로로 나옴
print(data.loc[:0]) # 0 행 출력 세로로 나옴
print(data.loc[[2],['name']])


# 데이터 프레임 변경해보기
data.index = np.arange(1,7,1) # 1~6까지 인덱스 설정
data.columns
data = data.reindex(['1','2','3','4','5','7'],columns = ['code', 'name', 'manufacture', 'price'])
print(data.index)
print(data);


















"""-------------------------------------------------------------------------------------------------------------------------------------------------------------- """
# 정규분포 생성 알고리즘
""" 정규분포란? 가우시안 정규 분포 :
    자연 현상에서 나타나는 숫자를 확률 모형으로 모형화할 때 가장 많이 사용되는 모형
    표준편차 1배안에 전체 데이터의 약 70% 이상이 몰려있고
    1.96배 안에 95% 이상이 분포된 경우
    """
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc 
from scipy import stats
import scipy as sp 
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
mu = 0 
std = 1
rv = sp.stats.norm(mu, std) 
xx = np.linspace(-5, 5, 100)
plt.plot(xx, rv.pdf(xx))
plt.ylabel("확률")
plt.title("정규분포곡선")
plt.show() 
x = rv.rvs(100) # rvs 메서드로 시뮬레이션해 샘플을 얻는것

print(x)

# 검정통계 , 유의확률(p-value)
"""
    검정(testing)은  데이터  뒤에  숨어있는  확률  변수의  분포와  모수에  대한  가설의  진 위를  정량적(quantitatively)으로  증명하는  작업
    
    실제 모집단에서 표본 몇 십 개 또는 몇 백 개를 추출해서 그것의 분산(표본분산)이나 평균(표본평균)을 사용해야 하는 경우가 대 부분입니다.
   표본수가  크지  않을  때  표본분산(표본표준편차)을  사용한  테스트는  t-분포를  이용 한다고  해서  T-test라고  합니다.
   가설  증명  즉  검정의  기본적인  논리는  다음과  같습니다.
   만약  가설이  맞다면  즉, 모수  값이  특정한  조건을  만족한다면  해당  확률  변수로부터  만 들어진  표본(sample) 데이터들은  어떤  규칙을  따르게  된다.
    해당 규칙에 따라 표본 데이터 집합에서 어떤 숫자를 계산하면 계산된 숫자는 특정한 확률 분포를 따르게 된다. 이 숫자를 검정 통계치(test statistics)라고 하며 확률 분포를 검정 통계 분포(test statistics distribution)라고 한다. 검정 통계 분포의 종류 및 모수의 값은 처음에 정한 가설에 의해 결정된다. 이렇게 검정 통계 분포를 결정하는 최초의 가 설을 귀무 가설(Null hypothesis)이라고 한다.
   데이터에  의해서  실제로  계산된  숫자, 즉, 검정  통계치가  해당  검정  통계  분포에서  나올 수  있는  확률을  계산한다. 이를  유의  확률(p-value)라고  한다.
   만약  유의  확률이  미리  정한  특정한  기준  값보다  작은  경우를  생각하자. 이  기준  값을 유의  수준(significance level)이라고  하는  데  보통  1% 혹은  5% 정도의  작은  값을  지정한 다. 유의  확률이  유의  수준으로  정한  값(예  1%)보다도  작다는  말은  해당  검정  통계  분포 에서  이  검정  통계치가  나올  수  있는  확률이  아주  작다는  의미이므로  가장  근본이  되는 가설  즉, 귀무  가설이  틀렸다는  의미이다. 따라서  이  경우에는  귀무  가설을  기각(reject) 한다.
   만약 유의 확률이 유의 수준보다 크다면 해당 검정 통계 분포에서 이 검정 통계치가 나 오는 것이 불가능하지만은 않다는 의미이므로 귀무 가설을 기각할 수 없다. 따라서 이 경우에는 귀무 가설을 채택(accept)한다.
    
    linspace 함수는 numpy모듈의 1차원 배열 만드는 함수
    x = np.linspace(start,stop,num) # num은 요소의 개수 그 사이에 몇개를 만들것인가.
    np.random.randit : 균일 분포의 정수 난수 1개 생서
    np.random.rand : 0부터 1사이의 난수 matrix array 생성
    np.random.randn : 가우시안 표준 정규 분포에서 난수 matrix array 생성
    plt.fill_between() : 두 수평 방향의 곡선 사이를 채웁니다.
    plt.fill_betweenx() : 두 수평 방향의 곡선 사이를 채웁니다.
    plt.fill() : 다각형 영역을 채웁니다.
    
    
    파이썬 플랏 상세 설명 : https://blog.naver.com/nach3012/222419686483
    
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc 
from scipy import stats
import scipy as sp

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name() 
rc('font', family=font_name)

xx1 = np.linspace(-4, 4, 100) 
xx2 = np.linspace(-4, -2, 100) 
xx3 = np.linspace(2, 4, 100)

plt.fill_between(xx1, sp.stats.norm.pdf(xx1), facecolor='green', alpha=0.1)
plt.fill_between(xx2, sp.stats.norm.pdf(xx2), facecolor='blue', alpha=0.35)
plt.fill_between(xx3, sp.stats.norm.pdf(xx3), facecolor='blue', alpha=0.35)
plt.text(-3, 0.1, "p-value=%5.3f" % (2*sp.stats.norm.cdf(-2)), horizontalalignment='center')
plt.title("유의확률:0.046") 
plt.show()
























