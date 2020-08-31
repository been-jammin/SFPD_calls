-- Databricks notebook source
-- MAGIC 
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px; height: 163px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Applying ML with UDFs
-- MAGIC ## Module 4, Lesson 6
-- MAGIC 
-- MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this notebook you:<br>
-- MAGIC * Apply a pre-trained Linear Regression model to predict response times
-- MAGIC * Identify which types of calls or neighborhoods are anticipated to have the longest response time

-- COMMAND ----------

-- MAGIC %run ../Includes/Classroom-Setup

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Create UDF
-- MAGIC 
-- MAGIC MLflow can create a User Defined Function for us to use in PySpark or SQL.  This allows for custom code (that is, functionality not in core Spark) to be run on Spark.
-- MAGIC 
-- MAGIC You can use `spark.udf.register` to register this Python UDF in the SQL namespace and call it `predictUDF`.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC try:
-- MAGIC   import mlflow
-- MAGIC   from mlflow.pyfunc import spark_udf
-- MAGIC 
-- MAGIC   model_path = "/dbfs/mnt/davis/fire-calls/models/firecalls_pipeline"
-- MAGIC   predict = spark_udf(spark, model_path, result_type="string")
-- MAGIC 
-- MAGIC   spark.udf.register("predictUDF", predict)
-- MAGIC except:
-- MAGIC   print("ERROR: This cell did not run, likely because you're not running the correct version of software. Please use a cluster with `DBR 5.5 ML` rather than `DBR 5.5` or a different cluster version.")

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Import the Data
-- MAGIC 
-- MAGIC Create a temporary view called `fireCallsParquet`

-- COMMAND ----------

CREATE OR REPLACE TEMPORARY VIEW fireCallsParquet
USING Parquet 
OPTIONS (
    path "/mnt/davis/fire-calls/fire-calls-1p.parquet"
  )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Save Predictions
-- MAGIC 
-- MAGIC We are going to save our predictions to a table called `predictions`.

-- COMMAND ----------

USE Databricks;
DROP TABLE IF EXISTS predictions;

CREATE TEMPORARY VIEW predictions AS (
  SELECT cast(predictUDF(Call_Type, Fire_Prevention_District, `Neighborhooods_-_Analysis_Boundaries`, 
                Number_of_Alarms, Original_Priority, Unit_Type) as double) as prediction, *
  FROM fireCallsParquet
  LIMIT 10000)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Average Prediction by Neighborhood
-- MAGIC 
-- MAGIC Let's see which district in San Francisco has the highest predicted average response time! Do you remember why we are setting the shuffle partitions here?

-- COMMAND ----------

SET spark.sql.shuffle.partitions=8;

-- COMMAND ----------

SELECT avg(prediction) as avgPrediction, `Neighborhooods_-_Analysis_Boundaries`
FROM predictions
GROUP BY `Neighborhooods_-_Analysis_Boundaries`
ORDER BY avgPrediction DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## San Francisco Districts
-- MAGIC 
-- MAGIC ![](https://files.training.databricks.com/images/eLearning/ucdavis/sfneighborhoods.gif)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Standard Deviation on Prediction by Neighborhood

-- COMMAND ----------

SELECT stddev(prediction) as stddevPrediction, `Neighborhooods_-_Analysis_Boundaries`
FROM predictions
GROUP BY `Neighborhooods_-_Analysis_Boundaries`
ORDER BY stddevPrediction DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Average Prediction by Call Type

-- COMMAND ----------

SELECT avg(prediction) as avgPrediction, Call_Type
FROM predictions
GROUP BY Call_Type
ORDER BY avgPrediction DESC


-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
