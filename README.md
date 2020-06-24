# ClassificationModelsinPySpark
## Comparing performance of various classification models in PySpark library ML

**Advisor: Prof. Joseph Rosen**<br>                                                

The aim of this project was to get hands-on experince in building machine learning models using big data tool, understand the difference between modeling python package like sklearn and spark ML package. <br>
I decided to solve the age old problem statement of building a classification model that predict if a credit card user would default. I used the Kaggle dataset 'Give me some credit' for this purpose. <br>

**A brief on the dataset:** <br>

The dataset originally consists of two sections training and test dataset with 150K and 102K rows respectively with 11 columns each. Due to restriction on task size at node to 100KB, I have reduced the dataset size to around 450 rows (stratified sampling) which is around 30 KB. This restriction does defeat the purpose of using a big data tool in the first place but serves well to my aim of just being able to explore Spark ML structure and functionality.

---

**The major steps involved are as follows:** <br>

(1) Initializing a Spark Session and scp data to remote <br>
(2) Exploartory Data Analysis <br>
(3) Data processing and transformation <br>
(4) Building Classification Models <br>
(5) Evaluation <br>

---

## (1) Initializing a Spark Session and scp data to remote

SCP file from local to remote
```
scp -P 2222m local_file_location/filename.csv username@localhost:/home/dev
```

Spark Session
```
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-bank').getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
RDD = spark.read.csv('filename.csv', header = True, inferSchema = True)
RDD.printSchema()

```

## (2) Exploartory Data Analysis
I also performed Exploratory Data Analysis on the dataset to get a thorough understanding of the data. It was challenging to work in PySpark when the interface is a terminal, to perform EDA I explored Pygal library to make interactive plot in the Hadoop sandbox terminal and imported the svg files in my system.	

Histogram plot

``` Spark
Import pygal
Default_60_89ds = pygal.Bar(x_label_rotation=45)
Default_60_89ds.title = ' Default across NumberOfTime60-89DaysPastDueNotWorse'
Default_60_89ds.x_labels = map(str, df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts().index)
Default_60_89ds.add('Non Defaulter' ,  df[df['SeriousDlqin2yrs'] == 0.0 ]['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())
Default_60_89ds.add('Defaulter',  df[df['SeriousDlqin2yrs'] == 1.0 ]['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())
Default_60_89ds.render_to_file('Default_60_89ds.svg')

```
<center> <img src="https://github.com/SanchayniB/ClassificationModelsinPySpark/blob/master/SubImages/EDA/60_89_SD.PNG" alt="histplot" width="300"> </center> 

Boxplot
``` Spark
BP_age_default = pygal.Box(box_mode="1.5IQR")
BP_age_default.title = 'Age variation across Default'
BP_age_default.add('Defaulter', df[df['SeriousDlqin2yrs'] == 1.0].age)
BP_age_default.add('Non Defaulter', df[df['SeriousDlqin2yrs'] == 0.0].age)
BP_age_default.render_to_file('BP_age_default.svg')
``` 

<img src="https://github.com/SanchayniB/ClassificationModelsinPySpark/blob/master/SubImages/EDA/Age_SD.PNG" alt="boxplot" width="300"> 

## (3) Data processing and transformation 
### Imputation
There was NA present in two columns of the dataset, rather than replacing by zero or removing the entry I used Imputer from ml.feature  <br>

``` 
from pyspark.ml.feature import Imputer
imputer = Imputer(inputCols = cols, outputCols = cols)
imputer_model = imputer.fit(data)
imputed_data = imputer_model.transform(data)
``` 
### Standardization
As I was planning on builidng Logistic Regression model I did standardizing using StandardScaler() function. Before standardizing the data needs to be vectorized for easier access to the variables while modelling and faster transformation

```
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = cols, outputCol = 'features')
data = assembler.transform(data)
data.select("features").show(truncate = False)
```

```
from pyspark.ml.feature import StandardScaler
standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
data = standardscaler.fit(data).transform(data)
```

## (4) Building Classification Models
### Stratified sampling

As we have imbalanced data, I used stratified sampling for splitting into train and test dataset rather than random sampling. We can perform this by using sampleBy function and providing the fractions parameter equal weights for both classes.

```
train_fractions = {0: 0.70, 1: 0.70}
train = data.sampleBy('SeriousDlqin2yrs',fractions = train_fractions ,seed= 1232)
test = data.subtract(train) 
```


TBC...


