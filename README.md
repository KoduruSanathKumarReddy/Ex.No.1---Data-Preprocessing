# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
~~~
Developed by: Koduru Sanath Kumar Reddy
Reg no: 212221240024
~~~
~~~
import pandas as pd
df=pd.read_csv("/content/Churn_Modelling.csv")
df.head()
df.isnull().sum()
df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)
print(df)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x)
print(y)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain)
print(len(xtrain))
print(xtest)
print(len(xtest))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1 = sc.fit_transform(df)
print(df1)
~~~

## OUTPUT:
<img width="1132" alt="Screenshot 2023-08-26 at 9 27 12 AM" src="https://github.com/KoduruSanathKumarReddy/Ex.No.1---Data-Preprocessing/assets/69503902/50395e18-8508-47ed-a4f3-14be5d049439">
<img width="1132" alt="Screenshot 2023-08-26 at 9 27 20 AM" src="https://github.com/KoduruSanathKumarReddy/Ex.No.1---Data-Preprocessing/assets/69503902/c953c4f8-1910-442e-84cf-7c20d291b9dc">



<img width="1119" alt="Screenshot 2023-08-26 at 9 27 59 AM" src="https://github.com/KoduruSanathKumarReddy/Ex.No.1---Data-Preprocessing/assets/69503902/133e9ef7-fdea-4a47-b586-748a15f69407">
<img width="1121" alt="Screenshot 2023-08-26 at 9 28 09 AM" src="https://github.com/KoduruSanathKumarReddy/Ex.No.1---Data-Preprocessing/assets/69503902/f7470fa3-93ff-4642-ae2b-64fdce2ff1b0">

## RESULT
Therefore the given data is successfully explorated using preprocessing techniques in neural networks.
