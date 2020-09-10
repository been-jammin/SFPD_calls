# SFPD_calls
AI/ML exercises on calls/responses to the san francisco fire department based on UC davis course. the first part will closely follow the instructions given in the course lectures. then, i will divert and use some of my own code to go deeper and improve on the instructions given in the course

## Introduction


## Data Extraction and Prep  (working alongside course instruction)
first, we link to our data source. the data's source is https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3/data

luckily, the course already condensed the data from this URL into a convenient parquet format called ```fireCallsClean```. more on how we got from hosted data to an Apache parquet file later.

the point is, we now have a parquet file which is faster to interact with, which has all the columns of the original, so let's load that in.

```
USE DATABRICKS;

CREATE TABLE IF NOT EXISTS fireCallsClean
USING parquet
OPTIONS (
  path "/mnt/davis/fire-calls/fire-calls-clean.parquet"
)
```
![histogram] (D:\github_portfolio\fireCalls\screenshots\Annotation 2020-09-01 161135.jpg)

```SQL
DESCRIBE fireCallsClean
```

next we can do a little SQL data cleaning and simple math to create a column to hold the "time delay", or the time between when the call was received and when the officers arrived

```sql
CREATE OR REPLACE VIEW time AS (
  SELECT *, unix_timestamp(Response_DtTm, "MM/dd/yyyy hh:mm:ss a") as ResponseTime, 
            unix_timestamp(Received_DtTm, "MM/dd/yyyy hh:mm:ss a") as ReceivedTime
  FROM fireCallsClean
)

CREATE OR REPLACE VIEW timeDelay AS (
  SELECT *, (ResponseTime - ReceivedTime)/60 as timeDelay
  FROM time
)
```
finally, we envoke Apache Arrow to read the SQL table into a Pandas dataframe. in the lecture, they only extract a few columns. also, they exclude erroneous records and outliers by filtering the table and only taking records whose timeDelay is between 0 and 15 minutes. this results in an unrealistically asthetic distribution of timeDelays, but it will work for now. i'd like to start deviating from the course instructions here, but we'll continue...

```
%python
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

pdDF = sql("""SELECT timeDelay, Call_Type, Fire_Prevention_District, `Neighborhooods_-_Analysis_Boundaries`, Number_of_Alarms, Original_Priority, Unit_Type
              FROM timeDelay 
              WHERE timeDelay < 15 AND timeDelay > 0""").toPandas()
```
next, we will use an 80/20 train/test split, using all the features of the data.
```
%python
from sklearn.model_selection import train_test_split

X = pdDF.drop("timeDelay", axis=1)
y = pdDF["timeDelay"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

next, we develop what the course calls a "baseline model" basically, how do we know that the linear regression model is actually doing anything? we need something to compare it to. so the course instructs us to develop a predictive model, but one where the prediction is always just the arithmetic mean of timeDelays of all the calls in the training data set. we accomplish that using the simple code below, which creates a variable of the same dimensions as y_test, and fills it with the arithmetic mean of y_train.
```
%python
from sklearn.metrics import mean_squared_error
import numpy as np

avgDelay = np.full(y_test.shape, np.mean(y_train), dtype=float)
rsme_baseline = np.sqrt(mean_squared_error(y_test, avgDelay))

print("RMSE is {0}".format(rsme_baseline)

Out[58]: array([3.36445485, 3.36445485, 3.36445485, ..., 3.36445485, 3.36445485,
       3.36445485])
```
we can see that rsme_baseline just predicts 3.3644... every time

the 2nd to last line uses scikit-learn's built in function mean_squared_error() to evaluate the average mean squared error between the prediction avgDelay and the y_test data. this is the number we are trying to beat with the linear regression model. so now we can easily build it.

note the use of one hot encoding for every feature, the enable of the normalization on the training set, and the use of the pipeilne functionality
```
%python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

ohe = ("ohe", OneHotEncoder(handle_unknown="ignore"))
lr = ("lr", LinearRegression(fit_intercept=True, normalize=True))

pipeline = Pipeline(steps = [ohe, lr]).fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```
[show histogram of predictions

now we can see that the model is predicting timeDelays between 3 and 4 minutes from the features in the X_test. 
```
print("RMSE is {0}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
RMSE is 1.724841956799332
```
and we can see that the RMSE of the prediction has improved...but not by very much. we can probably do better.

