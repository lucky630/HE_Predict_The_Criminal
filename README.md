# HE_Predict_The_Criminal
Winner Solution for HackerEarth Predict the Criminal challenge

## About
This solution is after the metric change from Precision to Mathews Correlation Coefficient.
For Starter and Precision solution refer to.

https://github.com/lucky630/HE_CRIMINAL_CL

Problem statement and data can be dowloaded from the competition site
https://www.hackerearth.com/challenge/competitive/predict-the-criminal/problems/

## Solution
Approach is divided into three layer structure.

First layer contain few lightgbm,xgboost,catboost,random forest,extratree,nn models trained on different features.

Second layer contain stack of these first layer model and models from denoising autoencoder data files.

Third layer contain blend of these second layer models and parity bit changer between different solutions.

## Requirement
- lightgbm
- keras
- sklearn
- xgboost
- catboost
- boruta
- rgf

## acknowledgement
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629

https://www.kaggle.com/tilii7/boruta-feature-elimination
