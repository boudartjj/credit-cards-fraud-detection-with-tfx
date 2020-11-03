# credit-cards-fraud-detection-with-tfx
Machine learning (ML) leverages our capacity to deliver new services or improve existing ones. But there is a gap to overcome between creating great models and make them available to the world. Google has developed TensorFlow Extended (TFX), an open source framework with all the components needed to define, launch and monitor your ML system.

The puropose of this notebook is to demonstrate an end 2 end ML TFX pipeline for payments fraud detection using Kaggle credit card fraud dataset (https://www.kaggle.com/mlg-ulb/creditcardfraud)
- collect data from Kaggle
- clean data
- features engineering
- train model

TODO
- evaluating model
- push model in production

## 1. ML industrialization challenges
### 1.1 Model performance decreases over time
- Challenge: Changes in the environment (new trends, rare conditions)
- Solution:
  - Monitor the model performance
  - Keep the model up to date (daily / hourly…) with new data and/or new design on a monthly / weekly / daily / hourly basis depending the domain
### 1.2 Time to market from development to production
- Challenge: complex activity (collect and clean data, code model, train model, evaluate model, deploy model to production environment)
- Solution: automate the process from data collection to model deployment 
<p align="center">
  <img src="https://uplanet-public.s3.amazonaws.com/GitHub+-+credit-cards-fraud-detection-with-tfx/ML+Activities.png">
</p>

## 2. About TFX
- TFX is a toolkit for building ML pipelines and provides
  - a set of standard components that provide dedicated functionalities (collect data, clean data, train model, evaluate model…) 
  - A metadata store where all the outputs of each components are kept to be reused at a later stage of the pipeline
- TFX is designed to be portable to multiple environments and orchestration frameworks, including Apache Airflow, Apache Beam and Kubeflow
<p align="center">
  <img src="https://uplanet-public.s3.amazonaws.com/GitHub+-+credit-cards-fraud-detection-with-tfx/TFX+Pipeline.png">
</p>

## 3. TFX Components
- When a Pipeline runs a TFX component, the component is executed in three phases:
  - First, the Driver uses the component specification to retrieve the required artifacts from the metadata store and pass them into the component.
  - Next, the Executor performs the component's work.
  - Then the Publisher uses the component specification and the results from the executor to store the component's outputs in the metadata store.
- Most custom component implementations do not require you to customize the Driver or the Publisher. 
<p align="center">
  <img src="https://uplanet-public.s3.amazonaws.com/GitHub+-+credit-cards-fraud-detection-with-tfx/TFX+Component.png">
</p>

## 4. TFX main components
### ExempleGen
The ExampleGen TFX Pipeline component ingests data into TFX pipelines. It consumes external files/services to generate Examples which will be read by other TFX components.
### StatisticsGen
The StatisticsGen TFX pipeline component generates features statistics over both training and serving data, which can be used by other pipeline components.
### SchemaGen
A SchemaGen pipeline component will automatically generate a schema by inferring types, categories, and ranges from the training data.
### Transform
The Transform TFX pipeline component performs feature engineering on tf.Examples emitted from an ExampleGen component
### Trainer
The Trainer TFX pipeline component trains a TensorFlow model.
### Evaluator
The Evaluator TFX pipeline component performs deep analysis on the training results for your models, to help you understand how your model performs on subsets of your data.
### Pusher
The Pusher component is used to push a validated model to a deployment target

## 5. Run the project in Google Colab
- Create a new project in Google Colab (https://colab.research.google.com) by selecting GitHub
- Paste GitHub Url (https://github.com/boudartjj/credit-cards-fraud-detection-with-tfx.git) and select boudartjj/credit-cards-fraud-detection-with-tfx
- Click on the notebook link (tfx_creditcard_fraud.inpynb)
