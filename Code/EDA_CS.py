# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:20:59 2019

@author: Sanchayni
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-bank').getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

RDD_01 = spark.read.csv('New_cs_data.csv', header = True, inferSchema = True)
RDD_01.printSchema()

df = RDD_01.select("*").toPandas()
df = df.drop(['_c0'],axis=1)
df = df.drop(['Unnamed: 0'],axis=1)

print(df.columns)
import numpy as np
import pygal
import pandas as pd

df = df.apply(lambda x: x.replace('NA',np.nan)) 
df = df.apply(lambda x: x.astype(float))

print(df.isna().sum())


# Plotting 

# --------------- UNIVARIATE plots ---------------

# Age
age_chart = pygal.Bar(x_label_rotation=-90)
age_chart.title = 'Histogram for variable Age'
age_chart.x_labels = map(str,pd.DataFrame(df.groupby('age').age.count()).index)
age_chart.add('age' ,  pd.DataFrame(df.groupby('age').age.count()).age)
age_chart.render_to_file('EDA_svgs/age_chart.svg')

# Dept Ratio
DebtRatio_chart = pygal.Bar(x_label_rotation=45)
DebtRatio_chart.title = 'Histogram for variable Debt Ratio'
DebtRatio_chart.x_labels = map(str, pd.DataFrame(pd.cut(df['DebtRatio'],20).value_counts().sort_index()).index)
DebtRatio_chart.add('DeptRatio' ,  pd.DataFrame(pd.cut(df['DebtRatio'],20).value_counts().sort_index()).DebtRatio)
DebtRatio_chart.render_to_file('EDA_svgs/DebtRatio_chart.svg')

# Monthly Income
Income_chart = pygal.Bar(x_label_rotation=45)
Income_chart.title = 'Histogram for variable Monthly Income'
Income_chart.x_labels = map(str, pd.DataFrame(pd.cut(df['MonthlyIncome'],20).value_counts().sort_index()).index)
Income_chart.add('MonthlyIncome' ,  pd.DataFrame(pd.cut(df['MonthlyIncome'],20).value_counts().sort_index()).MonthlyIncome)
Income_chart.render_to_file('EDA_svgs/Income_chart.svg')


# NumberOfDependents
NOD_chart = pygal.Bar(x_label_rotation=-90)
NOD_chart.title = 'Histogram for variable Number of Dependents'
NOD_chart.x_labels = map(str,pd.DataFrame(df.groupby('NumberOfDependents').NumberOfDependents.count()).index)
NOD_chart.add('NumberOfDependents',  pd.DataFrame(df.groupby('NumberOfDependents').NumberOfDependents.count()).NumberOfDependents)
NOD_chart.render_to_file('EDA_svgs/NOD_chart.svg')

# SeriousDlqin2yrs
SDy_chart = pygal.Bar(x_label_rotation=-90)
SDy_chart.title = 'Histogram for variable SeriousDlqin2yrs'
SDy_chart.x_labels = map(str,pd.DataFrame(df.groupby('SeriousDlqin2yrs').SeriousDlqin2yrs.count()).index)
SDy_chart.add('SeriousDlqin2yrs',  pd.DataFrame(df.groupby('SeriousDlqin2yrs').SeriousDlqin2yrs.count()).SeriousDlqin2yrs)
SDy_chart.render_to_file('EDA_svgs/SDy_chart.svg')



# RevolvingUtilizationOfUnsecuredLines
RUOUL_chart = pygal.Bar(x_label_rotation=-90)
RUOUL_chart.title = 'Histogram for variable RevolvingUtilization Of Unsecured Lines'

RUOUL_chart.x_labels = map(str,pd.DataFrame(pd.cut(df['RevolvingUtilizationOfUnsecuredLines'],20).value_counts().sort_index()).index)
RUOUL_chart.add('RevolvingUtilizationOfUnsecuredLines',  pd.DataFrame(pd.cut(df['RevolvingUtilizationOfUnsecuredLines'],20).value_counts().sort_index()).RevolvingUtilizationOfUnsecuredLines)
RUOUL_chart.render_to_file('EDA_svgs/RUOUL_chart.svg')


# NumberRealEstateLoansOrLines
NRELL_chart = pygal.Bar(x_label_rotation=45)
NRELL_chart.title = 'Histogram for NumberRealEstateLoansOrLines'
NRELL_chart.x_labels = map(str,pd.DataFrame(df.groupby('NumberRealEstateLoansOrLines').NumberRealEstateLoansOrLines.count()).index)
NRELL_chart.add('NumberRealEstateLoansOrLines',  pd.DataFrame(df.groupby('NumberRealEstateLoansOrLines').NumberRealEstateLoansOrLines.count()).NumberRealEstateLoansOrLines)
NRELL_chart.render_to_file('EDA_svgs/NRELL_chart.svg')


# NumberOfOpenCreditLinesAndLoans
NOCLL_chart = pygal.Bar(x_label_rotation=45)
NOCLL_chart.title = 'Histogram for NumberOfOpenCreditLinesAndLoans'
NOCLL_chart.x_labels = map(str,pd.DataFrame(df.groupby('NumberOfOpenCreditLinesAndLoans'). NumberOfOpenCreditLinesAndLoans.count()).index)
NOCLL_chart.add('NumberOfOpenCreditLinesAndLoans',  pd.DataFrame(df.groupby('NumberOfOpenCreditLinesAndLoans'). NumberOfOpenCreditLinesAndLoans.count()).NumberOfOpenCreditLinesAndLoans)
NOCLL_chart.render_to_file('EDA_svgs/NOCLL_chart.svg')


# NumberOfTime30-59DaysPastDueNotWorse
NT30_59PD_chart = pygal.Bar(x_label_rotation=45)
NT30_59PD_chart.title = 'Histogram for variable NumberOfTime30-59DaysPastDueNotWorse'
NT30_59PD_chart.x_labels = map(str,pd.DataFrame(df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts().sort_index()).index)
NT30_59PD_chart.add('NumberOfTime30-59DaysPastDueNotWorse',  pd.DataFrame(df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts().sort_index())['NumberOfTime30-59DaysPastDueNotWorse'])
NT30_59PD_chart.render_to_file('EDA_svgs/NT30_59PD_chart.svg')


# NumberOfTime60-89DaysPastDueNotWorse
NT60_89PD_chart = pygal.Bar(x_label_rotation=45)
NT60_89PD_chart.title = 'Histogram for variable NumberOfTime60-89DaysPastDueNotWorse'
NT60_89PD_chart.x_labels = map(str,pd.DataFrame(df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts().sort_index()).index)
NT60_89PD_chart.add('NumberOfTime60-89DaysPastDueNotWorse',  pd.DataFrame(df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts().sort_index())['NumberOfTime60-89DaysPastDueNotWorse'])
NT60_89PD_chart.render_to_file('EDA_svgs/NT60_89PD_chart.svg')

# NumberOfTimes90DaysLate
NT90DL_chart = pygal.Bar(x_label_rotation=45)
NT90DL_chart.title = 'Histogram for variable NumberOfTimes90DaysLate'
NT90DL_chart.x_labels = map(str,pd.DataFrame(df['NumberOfTimes90DaysLate'].value_counts().sort_index()).index)
NT90DL_chart.add('NumberOfTimes90DaysLate',  pd.DataFrame(df['NumberOfTimes90DaysLate'].value_counts().sort_index()).NumberOfTimes90DaysLate)
NT90DL_chart.render_to_file('EDA_svgs/NT90DL_chart.svg')


#-------------- BOXPLOTS --------------

# Age
BP_age_default = pygal.Box(box_mode="1.5IQR")
BP_age_default.title = 'Age variation across Default'
BP_age_default.add('Defaulter', df[df['SeriousDlqin2yrs'] == 1.0].age)
BP_age_default.add('Non Defaulter', df[df['SeriousDlqin2yrs'] == 0.0].age)
BP_age_default.render_to_file('EDA_svgs/BP_age_default.svg')


# DebtRatio
BP_DR_default = pygal.Box(box_mode="tukey")
BP_DR_default.title = 'Dept Ratio variation across Default'
BP_DR_default.add('Defaulter', df[df['SeriousDlqin2yrs'] == 1.0].DebtRatio)
BP_DR_default.add('Non Defaulter', df[df['SeriousDlqin2yrs'] == 0.0].DebtRatio)
BP_DR_default.render_to_file('EDA_svgs/BP_DR_default.svg')


# MonthlyIncome
# BP_MI_default = pygal.Box(box_mode="1.5IQR", range = (min(df['MonthlyIncome']),max(df['MonthlyIncome']) ) )

BP_MI_default = pygal.Box(box_mode="1.5IQR", range = (-10000,40000 ))

BP_MI_default.title = 'Monthly Income variation across Default'
BP_MI_default.add('Defaulter', df[(df['SeriousDlqin2yrs'] == 1.0) & (df['MonthlyIncome'].notna())].MonthlyIncome)
BP_MI_default.add('Non Defaulter', df[(df['SeriousDlqin2yrs'] == 0.0) & (df['MonthlyIncome'].notna())].MonthlyIncome)
BP_MI_default.render_to_file('EDA_svgs/BP_MI_default.svg')

# NumberOfDependents
BP_NOD_default = pygal.Box(box_mode="tukey", range = (-3,10))
BP_NOD_default.title = 'NumberOfDependents variation across Default'
BP_NOD_default.add('Defaulter', df[ (df['SeriousDlqin2yrs'] == 1.0) & (df['NumberOfDependents'].notna())].NumberOfDependents)
BP_NOD_default.add('Non Defaulter', df[ (df['SeriousDlqin2yrs'] == 0.0) & (df['NumberOfDependents'].notna())].NumberOfDependents)
BP_NOD_default.render_to_file('EDA_svgs/BP_NOD_default.svg')

# RevolvingUtilizationOfUnsecuredLines
BP_RUOUL_default = pygal.Box(box_mode="tukey")
BP_RUOUL_default.title = ' RevolvingUtilizationOfUnsecuredLines variation across Default'
BP_RUOUL_default.add('Defaulter', df[df['SeriousDlqin2yrs'] == 1.0].RevolvingUtilizationOfUnsecuredLines)
BP_RUOUL_default.add('Non Defaulter', df[df['SeriousDlqin2yrs'] == 0.0].RevolvingUtilizationOfUnsecuredLines)
BP_RUOUL_default.render_to_file('EDA_svgs/BP_RUOUL_default.svg')


# NumberOfTimes90DaysLate
Default_90ds = pygal.Bar(x_label_rotation=45)
Default_90ds.title = ' Default across Number of times 90 days late'
Default_90ds.x_labels = map(str, df['NumberOfTimes90DaysLate'].value_counts().sort_index().index)
Default_90ds.add('Non Defaulter' ,  df[df['SeriousDlqin2yrs'] == 0.0 ].NumberOfTimes90DaysLate.value_counts().sort_index())
Default_90ds.add('Defaulter',  df[df['SeriousDlqin2yrs'] == 1.0 ].NumberOfTimes90DaysLate.value_counts().sort_index())

Default_90ds.render_to_file('EDA_svgs/Default_90ds.svg')

# NumberOfTime60-89DaysPastDueNotWorse
Default_60_89ds = pygal.Bar(x_label_rotation=45)
Default_60_89ds.title = ' Default across NumberOfTime60-89DaysPastDueNotWorse'
Default_60_89ds.x_labels = map(str, df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts().sort_index().index)
Default_60_89ds.add('Non Defaulter' ,  df[df['SeriousDlqin2yrs'] == 0.0 ]['NumberOfTime60-89DaysPastDueNotWorse'].value_counts().sort_index())
Default_60_89ds.add('Defaulter',  df[df['SeriousDlqin2yrs'] == 1.0 ]['NumberOfTime60-89DaysPastDueNotWorse'].value_counts().sort_index())

Default_60_89ds.render_to_file('EDA_svgs/Default_60_89ds.svg')

# NumberOfTime30-59DaysPastDueNotWorse
Default_30_59ds = pygal.Bar(x_label_rotation=45)
Default_30_59ds.title = ' Default across NumberOfTime30-59DaysPastDueNotWorse'
Default_30_59ds.x_labels = map(str, df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts().sort_index().index)
Default_30_59ds.add('Non Defaulter' ,  df[df['SeriousDlqin2yrs'] == 0.0 ]['NumberOfTime30-59DaysPastDueNotWorse'].value_counts().sort_index())
Default_30_59ds.add('Defaulter',  df[df['SeriousDlqin2yrs'] == 1.0 ][ 'NumberOfTime30-59DaysPastDueNotWorse'].value_counts().sort_index())

Default_30_59ds.render_to_file('EDA_svgs/Default_30_59ds.svg')

# NumberOfOpenCreditLinesAndLoans
Default_OpenCredit = pygal.Bar(x_label_rotation=45)
Default_OpenCredit.title = ' Default across NumberOfOpenCreditLinesAndLoans'
Default_OpenCredit.x_labels = map(str, df['NumberOfOpenCreditLinesAndLoans'].value_counts().sort_index().index)
Default_OpenCredit.add('Non Defaulter' ,  df[df['SeriousDlqin2yrs'] == 0.0 ].NumberOfOpenCreditLinesAndLoans.value_counts().sort_index())
Default_OpenCredit.add('Defaulter',  df[df['SeriousDlqin2yrs'] == 1.0 ].NumberOfOpenCreditLinesAndLoans.value_counts().sort_index())

Default_OpenCredit.render_to_file('EDA_svgs/Default_OpenCredit.svg')

# NumberRealEstateLoansOrLines
Default_RealEstate= pygal.Bar(x_label_rotation=45)
Default_RealEstate.title = ' Default across NumberRealEstateLoansOrLines '
Default_RealEstate.x_labels = map(str, df['NumberRealEstateLoansOrLines'].value_counts().sort_index().index)
Default_RealEstate.add('Non Defaulter' ,  df[df['SeriousDlqin2yrs'] == 0.0 ].NumberRealEstateLoansOrLines.value_counts().sort_index())
Default_RealEstate.add('Defaulter',  df[df['SeriousDlqin2yrs'] == 1.0 ].NumberRealEstateLoansOrLines.value_counts().sort_index())

Default_RealEstate.render_to_file('EDA_svgs/Default_RealEstate.svg')

print('My work here is done')

# execfile('/home/maria_dev/EDA_CS.py')


