# -*- coding: utf-8 -*-
"""
@author: Sanchayni

# CSP 554 | BIG DATA
# Prof. Joseph Rosen
"""

# Type this in your pyspark session to execute this file: 
# execfile('/filepath/Classification_CS.py')

print(' \n \n *** Building a classification model for predicting the probability of a user defaulting on credit card payment *** \n ')
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-bank').getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")


print(' ----------------------------- Creating my own function for calculating evaluation metrices --------------------------')

# Function for extracting different metrices

def Evaluation_metrics(predictions_train,predictions_test, target_col  , prediction_col ):
    # Accuracy on training dataset
    accuracy_train = np.round(100 - (sum(np.abs(predictions_train[prediction_col] - 
                    predictions_train[target_col]))*100/ len(predictions_train)),3)
    # Accuracy on test dataset
    accuracy_test = np.round(100 - (sum(np.abs(predictions_test[prediction_col] - 
                    predictions_test[target_col]))*100/ len(predictions_test)),3)
    
    # Confusion Metric
    
    TP_test =  float(sum((predictions_test[target_col] == 0) & (predictions_test[prediction_col] == 0)))
    FP_test =  float(sum((predictions_test[target_col] == 1) & (predictions_test[prediction_col] == 0)))
    TN_test =  float(sum((predictions_test[target_col] == 1) & (predictions_test[prediction_col] == 1)))
    FN_test =  float(sum((predictions_test[target_col] == 0) & (predictions_test[prediction_col] == 1)))
    
    Recall_test = np.round(TP_test/ (TP_test+ FN_test),3)
    Precision_test =  np.round(TP_test / (TP_test + FP_test ),3)
    
    metrics = dict()
    metrics['Accuracy'] = [accuracy_train, accuracy_test]
    metrics['Precision'] = Precision_test
    metrics['Recall'] = Recall_test
    
    return(metrics)

print(' ------------------------------------ Reading the data --------------------------')

RDD_01 = spark.read.csv('New_cs_data.csv', header = True, inferSchema = True)
RDD_01.printSchema()
RDD_01.show(5,truncate=False)

df = RDD_01.select("*").toPandas()
df = df.drop(['_c0'],axis=1)
df = df.drop(['Unnamed: 0'],axis=1)

print(df.columns)

import numpy as np
import pygal
import pandas as pd

print(' ------------------------------------ Finding NA values --------------------------')
df = df.apply(lambda x: x.replace('NA',np.nan)) 
df = df.apply(lambda x: x.astype(float))

RDD_02 = sqlContext.createDataFrame(df)
from pyspark.ml.feature import Imputer

print(' ------------------------------------ Imputing using Imputer() NA values --------------------------')
# We don't want to impute the target variable
cols=RDD_02.columns
cols.remove('SeriousDlqin2yrs')

imputer=Imputer(inputCols= cols, outputCols = cols)
imputer_model=imputer.fit(RDD_02)
RDD_02= imputer_model.transform(RDD_02)
RDD_02.show(5, truncate=False)


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols= cols, outputCol= 'features')
RDD_02= assembler.transform(RDD_02)
RDD_02.select("features").show(truncate=False)

print(' ------------------------------------ Standardizing our feature variables --------------------------')

# Standardizing the features column

from pyspark.ml.feature import StandardScaler
standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
RDD_02=standardscaler.fit(RDD_02).transform(RDD_02)
RDD_02.select("features","Scaled_features").show(5, truncate=False)

print(' ------------------------------------ Stratified sampling | Train and Test split --------------------------')

# Splitting in train and test
# Stratified sampling

train_fractions = {0.0: 0.700, 1.0: 0.700}
train = RDD_02.sampleBy('SeriousDlqin2yrs',fractions = train_fractions ,seed= 45671)
test = RDD_02.subtract(train) 

 
total_train = float(train.select("SeriousDlqin2yrs").count())
defaults_train = float(train.select("SeriousDlqin2yrs").where('SeriousDlqin2yrs == 1').count())
non_defaults_train  = float(train.select("SeriousDlqin2yrs").where('SeriousDlqin2yrs == 0').count())


total_test = float(test.select("SeriousDlqin2yrs").count())
defaults_test = float(test.select("SeriousDlqin2yrs").where('SeriousDlqin2yrs == 1').count())
non_defaults_test  = float(test.select("SeriousDlqin2yrs").where('SeriousDlqin2yrs == 0').count())


print('Number of enteries in train dataset = {}'.format(total_train))
print('Number of defaulters in train dataset = {}'.format(defaults_train))
print('Number of non defaulters in train dataset = {}\n'.format(non_defaults_train))

print('Number of enteries in test dataset = {}'.format(total_test))
print('Number of defaulters in test dataset = {}'.format(defaults_test))
print('Number of non defaulters in test dataset = {} \n'.format(non_defaults_test))


BalancingRatio= non_defaults_train / total_train
print('BalancingRatio in train = {}'.format(BalancingRatio))

BalancingRatio_test= non_defaults_test / total_test
print('BalancingRatio in test = {}\n'.format(BalancingRatio_test))

# Class Weight
from pyspark.sql.functions import when
train=train.withColumn("classWeights", when(train.SeriousDlqin2yrs == 1,BalancingRatio).otherwise(1-BalancingRatio))


print(' ------------------------------------ Logistic Regression --------------------------')
from pyspark.ml.classification import LogisticRegression
logR = LogisticRegression(labelCol="SeriousDlqin2yrs", featuresCol="Scaled_features" 
                          ,weightCol="classWeights",maxIter=7)
logR_m = logR.fit(train)

# Model diagnostics
print("Coefficients: " + str(logR_m.coefficients))
print("Intercept: \n" + str(logR_m.intercept))

trainingSummary = logR_m.summary
roc = trainingSummary.roc.toPandas()

from pygal.style import LightColorizedStyle
ROC_curve = pygal.Line(fill=True,  style=LightColorizedStyle, 
                       y_label_rotation=45, x_label_rotation=45)
ROC_curve.title = 'The ROC curve for Logistic Regression'
ROC_curve.x_labels = map(str,np.round(roc['FPR'],3))
ROC_curve.add('TPR vs FPR', roc['TPR'])
ROC_curve.render_to_file('Classification_plots/ROC_curve_LR.svg')


# print('Training Summary: \n',trainingSummary)

# Predictions
predict_train_LR= logR_m.transform(train)
predict_test_LR= logR_m.transform(test)

predict_train_LR.select("SeriousDlqin2yrs","rawPrediction","prediction","probability").show(5,truncate=False)
predict_test_LR.select("SeriousDlqin2yrs","rawPrediction","prediction","probability").show(5,truncate=False)

predict_train_LR_pd = predict_train_LR.toPandas()
predict_test_LR_pd = predict_test_LR.toPandas()
     

    
KPI_LR = Evaluation_metrics(predict_train_LR_pd, predict_test_LR_pd,
                   target_col = "SeriousDlqin2yrs", prediction_col = "prediction" )

print('\n My metrics for Logistic Regression: {} \n'.format(KPI_LR))


print("The Accuracy for Logistic (Train) = {}".format(KPI_LR['Accuracy'][0]))
print("The Accuracy for Logistic (Test) = {} \n".format(KPI_LR['Accuracy'][1]))

# Confusion Matrix 
print('\nConfusion Matrix Train:')
print(pd.crosstab(index=predict_train_LR_pd["prediction"], 
                  columns=predict_train_LR_pd["SeriousDlqin2yrs"]))
print('\nConfusion Matrix Test:')
print(pd.crosstab(index=predict_test_LR_pd["prediction"], 
                  columns=predict_test_LR_pd["SeriousDlqin2yrs"]))


print("Recall for Logistic Regression = {}".format(KPI_LR['Recall']))
print("Precision for Logistic Regression = {} \n".format(KPI_LR['Precision']))

# AUC values
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator_AUC = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',
                                        labelCol='SeriousDlqin2yrs')

print("The AUC for Logistic (Train) = {}".format(np.round(evaluator_AUC.evaluate(predict_train_LR),3)))
print("The AUC for Logistic (Test) = {} \n".format(np.round(evaluator_AUC.evaluate(predict_train_LR),3)))


'''
print('------------------------------------ GLR --------------------------')

from pyspark.ml.regression import GeneralizedLinearRegression
glr = GeneralizedLinearRegression(labelCol="SeriousDlqin2yrs", featuresCol="Scaled_features" 
                          ,weightCol="classWeights", family="binomial", link="logit", 
                          maxIter=10, regParam=0.0)
model = glr.fit(train)
summary = model.summary

print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))

'''
print('------------------------------------ 2nd Iteration- Kfold --------------------------')



# K fold Cross Validation

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = ParamGridBuilder()\
    .addGrid(logR.aggregationDepth,[2,5,10])\
    .addGrid(logR.elasticNetParam,[0.0, 0.5, 1.0])\
    .addGrid(logR.fitIntercept,[False, True])\
    .addGrid(logR.maxIter,[10, 100])\
    .addGrid(logR.regParam,[0.01, 0.5, 2.0]) \
    .build()

CV = CrossValidator(estimator= logR, estimatorParamMaps=paramGrid, evaluator=evaluator_AUC, numFolds= 5)
CVModel = CV.fit(train)

# Best Model 

Best_Logm = CVModel.bestModel


print(Best_Logm.coefficients)
print(Best_Logm.intercept)

predict_train_cv=CVModel.transform(train)
predict_test_cv=CVModel.transform(test)

predict_train_cv_pd = predict_train_cv.toPandas()
predict_test_cv_pd = predict_test_cv.toPandas()


# Accuracy Calculation on final model

KPI_LR_final = Evaluation_metrics(predict_train_cv_pd, predict_test_cv_pd,
                   target_col = "SeriousDlqin2yrs", prediction_col = "prediction" )

print('\n My metrics for final Logistic Regression model: {}'.format(KPI_LR_final))


print("The Accuracy for final Logistic (Train) = {}".format(KPI_LR_final['Accuracy'][0]))
print("The Accuracy for final Logistic (Test) = {} \n".format(KPI_LR_final['Accuracy'][1]))

# Confusion Matrix 
print('\nConfusion Matrix Train :')
print(pd.crosstab(index=predict_train_cv_pd["prediction"], 
                  columns=predict_train_cv_pd["SeriousDlqin2yrs"]))
print('\nConfusion Matrix Test: \n')
print(pd.crosstab(index=predict_test_cv_pd["prediction"], 
                  columns=predict_test_cv_pd["SeriousDlqin2yrs"]))


print("Recall of final Logistic Regression model = {}".format(KPI_LR_final['Recall']))
print("Precision of final Logistic Regression model = {} \n".format(KPI_LR_final['Precision']))

print("The AUC for final Logistic (Train) = {}".format(evaluator_AUC.evaluate(predict_train_cv)))
print("The AUC for final Logistic (Test) = {} \n".format(evaluator_AUC.evaluate(predict_test_cv)))




print(' ------------------------------------ Naive Bayes --------------------------')

from pyspark.ml.classification import NaiveBayes

# NB needs a column called label
# with Column adds a column in an RDD

NB = NaiveBayes(featuresCol="Scaled_features", labelCol="SeriousDlqin2yrs" , smoothing=1.0 ,
                weightCol = "classWeights")
NB_m = NB.fit(train)

# Predictions
predict_train_NB= NB_m.transform(train)
predict_test_NB = NB_m.transform(test)

predict_train_NB_pd = predict_train_NB.toPandas()
predict_test_NB_pd = predict_test_NB.toPandas()

KPI_NB = Evaluation_metrics(predict_train_NB_pd, predict_test_NB_pd,
                   target_col = "SeriousDlqin2yrs", prediction_col = "prediction")

# Confusion Matrix 
print('\nConfusion Matrix Train:')
print(pd.crosstab(index=predict_train_NB_pd["prediction"], 
                  columns=predict_train_NB_pd["SeriousDlqin2yrs"]))
print('\nConfusion Matrix Test:')
print(pd.crosstab(index=predict_test_NB_pd["prediction"], 
                  columns=predict_test_NB_pd["SeriousDlqin2yrs"]))

print('\n My metrics for Naive Bayes:{} \n'.format(KPI_NB))


print("The Accuracy on train set is = {}".format(KPI_NB['Accuracy'][0]))
print("The Accuracy on test set is = {} \n".format(KPI_NB['Accuracy'][1]))


print("Recall for Naive Bayes = {}".format(KPI_NB['Recall']))
print("Precision for Naive Bayes = {} \n".format(KPI_NB['Precision']))


print("The AUC for Naive Bayes (Train) = {}".format(evaluator_AUC.evaluate(predict_train_NB)))
print("The AUC for Naive Bayes (Test) = {}".format(evaluator_AUC.evaluate(predict_test_NB)))


print('------------------------------------ Decision Trees --------------------------')
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorIndexer,StringIndexer
from pyspark.ml import Pipeline


labelIndexer = StringIndexer(inputCol="SeriousDlqin2yrs", outputCol="indexedLabel").fit(RDD_02)
featureIndexer = VectorIndexer(inputCol="Scaled_features", outputCol="indexedFeatures", 
                               maxCategories=4).fit(RDD_02)

DT = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
# Default parameters:
# maxDepth=5, maxBins=32

pipeline_dt = Pipeline(stages=[labelIndexer, featureIndexer, DT])

DT_m = pipeline_dt.fit(train)

treeModel = DT_m.stages[2]
print(treeModel.toDebugString)

# Important Variables
print('Important Features: {} \n'.format(treeModel.featureImportances))

# Predictions
predict_train_DT= DT_m.transform(train)
predict_test_DT = DT_m.transform(test)

predict_train_DT_pd = predict_train_DT.toPandas()
predict_test_DT_pd = predict_test_DT.toPandas()


KPI_DT = Evaluation_metrics(predict_train_DT_pd, predict_test_DT_pd,
                   target_col = "SeriousDlqin2yrs", prediction_col = "prediction" )

# Confusion Matrix 
print('\nConfusion Matrix Train:')
print(pd.crosstab(index=predict_train_DT_pd["prediction"], 
                  columns=predict_train_DT_pd["SeriousDlqin2yrs"]))
print('\nConfusion Matrix Test:')
print(pd.crosstab(index=predict_test_DT_pd["prediction"], 
                  columns=predict_test_DT_pd["SeriousDlqin2yrs"]))

print('\n My metrics for Decision Tree: {} \n'.format(KPI_DT))


print("The Accuracy on train set is = {}".format(KPI_DT['Accuracy'][0]))
print("The Accuracy on test set is = {} \n".format(KPI_DT['Accuracy'][1]))


print("Recall for Decision Tree = {}".format(KPI_DT['Recall']))
print("Precision for Decision Tree = {} \n".format(KPI_DT['Precision']))



print("The AUC for Decision Tree (Train) = {}".format(evaluator_AUC.evaluate(predict_train_DT)))
print("The AUC for Decision Tree (Test) = {}".format(evaluator_AUC.evaluate(predict_test_DT)))



print(' ------------------------------------ Random Forest --------------------------')

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString

RF_t1 = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", 
                                     numTrees=20, featureSubsetStrategy="auto",
                                     impurity='gini')
# Default parameters:
# maxDepth=5, maxBins=32

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

pipeline_RF = Pipeline(stages=[labelIndexer, featureIndexer, RF_t1, labelConverter])
RF_t1_m = pipeline_RF.fit(train)


RFtreeModel = RF_t1_m.stages[2]
# print(RFtreeModel.toDebugString)
# This plot is pretty interesting 
# Not printing it right now because it has 10 trees


# Important Variables
print('Important Features: {} \n'.format(RFtreeModel.featureImportances))

# Predictions
predict_train_RF = RF_t1_m.transform(train)
predict_test_RF = RF_t1_m.transform(test)

predict_train_RF_pd = predict_train_DT.toPandas()
predict_test_RF_pd = predict_test_DT.toPandas()


KPI_RF = Evaluation_metrics(predict_train_RF_pd, predict_test_RF_pd,
                   target_col = "SeriousDlqin2yrs", prediction_col = "prediction" )

# Confusion Matrix 
print('\nConfusion Matrix Train:')
print(pd.crosstab(index=predict_train_RF_pd["prediction"], 
                  columns=predict_train_RF_pd["SeriousDlqin2yrs"]))
print('\nConfusion Matrix Test:')
print(pd.crosstab(index=predict_test_RF_pd["prediction"], 
                  columns=predict_test_RF_pd["SeriousDlqin2yrs"]))

print('\n My metrics for Random Forest: {} \n'.format(KPI_RF))


print("Accuracy on train set is = {}".format(KPI_RF['Accuracy'][0]))
print("Accuracy on test set is = {} \n".format(KPI_RF['Accuracy'][1]))


print("Recall for Random Forest = {}".format(KPI_RF['Recall']))
print("Precision for Random Forest = {} \n".format(KPI_RF['Precision']))


print("The AUC for Random Forest (Train) = {}".format(evaluator_AUC.evaluate(predict_train_RF)))
print("The AUC for Random Forest (Test) = {}".format(evaluator_AUC.evaluate(predict_test_RF)))




print(' ------------------------------------ Gradient Boosting --------------------------')

from pyspark.ml.classification import GBTClassifier

GBT = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)
pipeline_gbt = Pipeline(stages=[labelIndexer, featureIndexer, GBT])

# Default parameters: 
# maxDepth=5, maxBins=32,
GBT_m = pipeline_gbt.fit(train)


GBT_stages = GBT_m.stages[2]
# print(GBT_stages.toDebugString)
# This plot is pretty interesting 
# Not printing it right now because it has 10 trees

# Important Variables
print('Important Features: {} \n'.format(GBT_stages.featureImportances))


# Predictions
predict_train_GBT = GBT_m.transform(train)
predict_test_GBT = GBT_m.transform(test)

predict_train_GBT_pd = predict_train_GBT.toPandas()
predict_test_GBT_pd = predict_test_GBT.toPandas()


KPI_GBT = Evaluation_metrics(predict_train_GBT_pd, predict_test_GBT_pd,
                   target_col = "SeriousDlqin2yrs", prediction_col = "prediction" )

# Confusion Matrix 
print('\nConfusion Matrix Train:')
print(pd.crosstab(index=predict_train_GBT_pd["prediction"], 
                  columns=predict_train_GBT_pd["SeriousDlqin2yrs"]))
print('\nConfusion Matrix Test:')
print(pd.crosstab(index=predict_test_GBT_pd["prediction"], 
                  columns=predict_test_GBT_pd["SeriousDlqin2yrs"]))

print('\n My metrics for Gradient Boosting Trees: {} \n'.format(KPI_GBT))


print("Accuracy on train set is = {}".format(KPI_GBT['Accuracy'][0]))
print("Accuracy on test set is = {} \n".format(KPI_GBT['Accuracy'][1]))


print("Recall for Gradient Boosting Trees = {}".format(KPI_GBT['Recall']))
print("Precision for Gradient Boosting Trees = {} \n".format(KPI_GBT['Precision']))


print("The AUC for Gradient Boosting Trees (Train) = {}".format(evaluator_AUC.evaluate(predict_train_GBT)))
print("The AUC for Gradient Boosting Trees (Test) = {}".format(evaluator_AUC.evaluate(predict_test_GBT)))

# execfile('/home/maria_dev/Classification_CS.py')