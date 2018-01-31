import findspark

# Or the following command
findspark.init("/usr/local/spark/spark-2.2.1-bin-hadoop2.7")

from pyspark import SparkContext, SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import *

# Import `DenseVector`
from pyspark.ml.linalg import DenseVector


spark = SparkSession.builder \
   .master("local") \
   .appName("Linear Regression Model") \
   .config("spark.executor.memory", "1gb") \
   .getOrCreate()


# start spark session

rdd = spark.sparkContext.textFile('/home/kunal/Downloads/CaliforniaHousing/cal_housing.data')
header = spark.sparkContext.textFile('/home/kunal/Downloads/CaliforniaHousing/cal_housing.domain')
# read in the files

header.collect()
rdd = rdd.map(lambda line: line.split(",")) # each line just a string of values split by commas, use line.split to create list of lists

df = rdd.map(lambda line: Row(longitude=line[0], 
                              latitude=line[1], 
                              housingMedianAge=line[2],
                              totalRooms=line[3],
                              totalBedRooms=line[4],
                              population=line[5], 
                              households=line[6],
                              medianIncome=line[7],
                              medianHouseValue=line[8])).toDF()

# schema of dataframe above and converted into dataframe


# all columns of type string,
# Write a custom function to convert the data type of DataFrame columns, just cast each column to a float
def convertColumn(df, names, newType):
    for name in names: 
        df = df.withColumn(name, df[name].cast(newType))
    return df 

# Assign all column names to `columns`
columns = ['households', 'housingMedianAge', 'latitude', 'longitude', 'medianHouseValue', 'medianIncome', 'population', 'totalBedRooms', 'totalRooms']

# Conver the `df` columns to `FloatType()`
df = convertColumn(df, columns, FloatType())



df = df.withColumn("medianHouseValue", col("medianHouseValue")/100000)
# standardize median house value as value range too large


# create new variables

# rooms per household
roomsPerHousehold = df.select(col("totalRooms")/col("households"))

#`population` by `households`
populationPerHousehold = df.select(col("population")/col("households"))

# `totalBedRooms` by `totalRooms`
bedroomsPerRoom = df.select(col("totalBedRooms")/col("totalRooms"))



# Add the new columns to `df`
df = df.withColumn("roomsPerHousehold", col("totalRooms")/col("households")) \
   .withColumn("populationPerHousehold", col("population")/col("households")) \
   .withColumn("bedroomsPerRoom", col("totalBedRooms")/col("totalRooms"))


# choose only a selected few columns, order so label is first column so makes it easier to visualize split
df = df.select("medianHouseValue", 
              "totalBedRooms", 
              "population", 
              "households", 
              "medianIncome", 
              "roomsPerHousehold", 
              "populationPerHousehold", 
              "bedroomsPerRoom")


# Define the `input_data` 
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:]))) # create a tupple with label/target in first index with features in second index



# Replace `df` with the new DataFrame
df = spark.createDataFrame(input_data, ["label", "features"]) # store in new dataframe with label holding house values and features held in features

from pyspark.ml.feature import StandardScaler

# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

scaler = standardScaler.fit(df)

# Transform the data in `df` with the scaler
scaled_df = scaler.transform(df)


# Split the data into train and test sets, 80% train, 20% test
train_data, test_data = scaled_df.randomSplit([.8,.2],seed=1234)


# Import `LinearRegression`
from pyspark.ml.regression import LinearRegression

# Initialize `lr`
lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8) 

# Fit the training data to the model
linearModel = lr.fit(train_data)

# Generate predictions of houe median values for test data
predicted = linearModel.transform(test_data)

# Extract the predictions and the actual results
predictions = predicted.select("prediction").rdd.map(lambda x: x[0]) # get test predictions
labels = predicted.select("label").rdd.map(lambda x: x[0]) # get test actual values

# Zip `predictions` and `labels` into a list
predictionAndLabel = predictions.zip(labels).collect()


# Get the RMSE
# linearModel.summary.rootMeanSquaredError

# Get the R2, R2 common lin reg metric, lower it the worse, higher it is the better.
print (linearModel.summary.r2)
print ('done')
