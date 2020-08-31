# SFPD_calls
AI/ML exercises on calls/responses to the san francisco fire department based on UC davis course

## Data Extraction and Prep
first, we link to our data source. the data's source is https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3/data

luckily, the course already condensed the data from this URL into a convenient parquet format called ```fireCallsClean```. more on how we got from hosted data to an Apache parquet file later.

the point is, we now have a parquet file which is faster to interact with, which has all the columns of the original
[show image of notebook showing describe]

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
finally, we Apache Arrow to read selected columns directly into a Pandas dataframe. 

```
%python
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

pdDF = sql("""SELECT timeDelay, Call_Type, Fire_Prevention_District, `Neighborhooods_-_Analysis_Boundaries`, Number_of_Alarms, Original_Priority, Unit_Type
              FROM timeDelay 
              WHERE timeDelay < 15 AND timeDelay > 0""").toPandas()
```              

## Data scrubbing...switch to Spyder IDE for this
