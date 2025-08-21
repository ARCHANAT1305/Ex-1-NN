
<H3>EX. NO.1</H3>
<H3>DATE : 19/08/25</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

### NAME : ARCHANA T
### REG NO : 212223240013
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd                  
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn_Modelling.csv")
print(df)

x = df.iloc[:, :-1].values
x

y = df.iloc[:, -1].values
y

print(df.isnull().sum())
df.duplicated()
df.describe()

df = df.drop(['Surname', 'Geography', 'Gender'], axis=1)
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train)
print(len(x_train))

print(x_test)
print(len(x_test))

```


## OUTPUT:
<img width="750" height="224" alt="image" src="https://github.com/user-attachments/assets/b5912446-c6af-4b46-8fad-29b4e37ab9ea" />
<img width="657" height="227" alt="image" src="https://github.com/user-attachments/assets/7c6ef193-ca80-4e31-b6ec-f97e9b13a0cf" />
<img width="344" height="516" alt="image" src="https://github.com/user-attachments/assets/1f2438a3-1784-43ab-b6bb-d49f8add7d8a" />
<img width="644" height="230" alt="image" src="https://github.com/user-attachments/assets/2c80bea3-93b5-438b-a023-0ef6d18297ff" />
<img width="513" height="321" alt="image" src="https://github.com/user-attachments/assets/fc43db7e-3619-4ed5-abc1-d45a45d39ad9" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


