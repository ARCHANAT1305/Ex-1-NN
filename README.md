
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

df = pd.read_csv("heart.csv")
print("Initial Dataset:\n", df.head())

print("\nMissing Values:\n", df.isnull().sum())
df = df.fillna(df.mean())

df = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding:\n", df.head())

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("\nNormalized Data:\n", df_scaled.head())

x = df_scaled.drop('target', axis=1).values
y = df_scaled['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nTraining set size:", len(x_train))
print("Testing set size:", len(x_test))

```


## OUTPUT:
<img width="715" height="243" alt="image" src="https://github.com/user-attachments/assets/89d908b4-a1f0-49f1-be56-8b19a3fe35c4" />
<img width="160" height="308" alt="image" src="https://github.com/user-attachments/assets/f4ee8163-fe22-4e61-9980-f2ad47f300a3" />
<img width="648" height="262" alt="image" src="https://github.com/user-attachments/assets/b6e69373-8f5e-4521-b121-2e7e3d4f70e8" />
<img width="262" height="55" alt="image" src="https://github.com/user-attachments/assets/c0963119-143b-4ae3-908a-11023f90299e" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


