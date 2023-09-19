# Breast Cancer Detection with Machine Learning




## Overview


This repository contains a machine learning model built using historical breast cancer data to determine if a given breast tissue sample is malignant or benign. The model uses logistic regression for classification and is deployed as an interactive tool using Streamlit. The primary audience for this tool is nursing staff who can input the metrics of breast tissue samples to assess their malignancy.




## Features


Machine Learning Model: A logistic regression model trained on historical breast cancer data to predict malignancy based on input metrics.
Streamlit Web App: An interactive web application created with Streamlit to allow nursing staff to enter sample metrics and receive predictions.
Data Scaling: Data preprocessing includes scaling to ensure the radar chart used in the application is readable and proportional.
Model Persistence: The model is serialized using the Pickle package to facilitate loading and use within the Streamlit application.
CSS Styling: Custom CSS code is applied to enhance user-friendliness and visual appeal of the web application.





## Useage


Clone the repository to your local machine:

 ``` git clone https://github.com/DavidStrickland19/Streamli_Breast_Cancer_Predictor.git```


Install the required dependencies: 

```pip install -r requirements.txt```


Run the Streamlit Application:

```streamlit run app.py```







## Data


This model was built using data from Breast Cancer Wisconsin (Diagnostic) Data Set. This model and app are only intended for an exercise and not for medical use.






## Contact


davidmstrickland19@gmail.com
