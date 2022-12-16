# Databricks notebook source
import os
life_exp = spark.read.format("csv").option("header","True").option("inferSchema","True").load(f"file:{os.getcwd()}/life_expectancy.csv")

# COMMAND ----------

life_exp.display()

# COMMAND ----------

life_exp.count()

# COMMAND ----------

life_exp = life_exp.dropna()
life_exp.count()

# COMMAND ----------

display(life_exp.groupBy('Status').count().orderBy('Status'))

# COMMAND ----------

independent_variables = ['Adult Mortality','Schooling','Total expenditure','Diphtheria ','GDP','Population']

dependent_variable = ['Life expectancy ']

# COMMAND ----------

life_exp_corr = life_exp.select(independent_variables + dependent_variable )

# COMMAND ----------

for i in life_exp_corr.columns:
        print('Correlation to life expextancy for', i,'is: ', life_exp_corr.stat.corr('Life expectancy ',i))

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

categoricalCols = ['Status']

stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols])

# COMMAND ----------

stringIndexerModel = stringIndexer.fit(life_exp)

life_exp_statusindex = stringIndexerModel.transform(life_exp)

display(life_exp_statusindex)

# COMMAND ----------

life_exp_statusindex.filter(life_exp_statusindex.Country.isin(['Afghanistan','Italy'])).select(['country','Status','StatusIndex']).display()

# COMMAND ----------

feature_columns = ['Year', 'Adult Mortality','infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B','Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure','Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',' thinness  1-19 years', ' thinness 5-9 years','Income composition of resources', 'Schooling']
label_column = 'Life expectancy '

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_columns , outputCol='features')

life_exp_features_label = assembler.transform(life_exp_statusindex).select(['features',label_column])

life_exp_features_label.display()

# COMMAND ----------

train_df, test_df = life_exp_features_label.randomSplit([0.75,0.25],seed=123)

# COMMAND ----------

train_df.count(), test_df.count()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

linear_regression = LinearRegression(featuresCol='features', labelCol=label_column)
linear_regression_model = linear_regression.fit(train_df)

# COMMAND ----------

print('Model coefficients: \n' + str(linear_regression_model.coefficients))

# COMMAND ----------

print('Intercept: \n' + str(linear_regression_model.intercept))

# COMMAND ----------

training_summary = linear_regression_model.summary
print('RMSE: %f' % training_summary.rootMeanSquaredError)
print('R-SQUARED: %f' % training_summary.r2)

# COMMAND ----------

training_summary.residuals.display()

# COMMAND ----------

test_predictions = linear_regression_model.transform(test_df)

print('TEST DATASET PREDICTIONS AGAINST ACTUAL LABEL: ')

test_predictions.select('features','prediction','Life expectancy ').display()

# COMMAND ----------

test_summary = linear_regression_model.evaluate(test_df)
print('RMSE: on test data = %g' % test_summary.rootMeanSquaredError)
print('R-SQUARED on test data %g' % training_summary.r2)

# COMMAND ----------

print(linear_regression.explainParams())

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(linear_regression.regParam,[1.0,0.05,0.01]) \
                              .addGrid(linear_regression.fitIntercept,[True,False]) \
                              .addGrid(linear_regression.elasticNetParam,[0.0,0.5,1.0]) \
                              .build()


# COMMAND ----------

evaluator = RegressionEvaluator(labelCol=label_column)

# COMMAND ----------

tvs = TrainValidationSplit(estimator=linear_regression,estimatorParamMaps=paramGrid,evaluator=evaluator,trainRatio=0.8)

# COMMAND ----------

model = tvs.fit(train_df)

# COMMAND ----------

tuned_prediction = model.transform(test_df)
tuned_prediction.select('features','Life expectancy ','prediction').display()

# COMMAND ----------

r2_score = evaluator.setMetricName('r2').evaluate(tuned_prediction)
print('R-squared on test data = %g' % r2_score)

# COMMAND ----------

print('best regParam: '+ str(model.bestModel._java_obj.getRegParam()) + '\n' + 'Best ElasticNetParam: ' + str(model.bestModel._java_obj.getElasticNetParam()))

# COMMAND ----------


