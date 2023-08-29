# insta_fake_detector

# Fake Account Detection on Instagram 

This repository contains a machine-learning pipeline to detect fake Instagram accounts. The dataset consists of various attributes of Instagram profiles, and the goal is to predict whether a given account is fake or real.

The dataset contains the following columns:
profile pic: Binary attribute indicating if an account has a profile picture.
nums/length username: Proportion of numerical characters in the username.
fullname words: Number of words in the account holder's name.
nums/length fullname: Proportion of numerical characters in the full name.
name==username: Binary feature indicating if the account holder's name matches the username.
description length: Length of the profile's description (bio).
external URL: Binary feature indicating if there's an external website link in the profile's bio.
private: Binary attribute showing if the profile is private.
#posts: Number of posts on the profile.
#followers: Follower count for the account.
#follows: Number of accounts the profile is following.
fake: Target variable. 1 if the account is fake, 0 otherwise.

## Exploratory Data Analysis
The EDA process involves:
Checking for missing and duplicated values.
Inspecting data types and basic statistics.
Visualizing the distribution of binary and continuous features.
Observing the relationship between features and the target variable.

## Feature Engineering
Two new features were created:
Activity Ratio: Measures an account's posting activity relative to its follower count.
Followers > #Follows?: Binary feature indicating if an account has more followers than the number of accounts it follows.

## Modeling
Random Forest (used as a baseline)
XGBoost
LGBM
CatBoost
AdaBoost
The models were evaluated using the AUC-ROC score. CatBoost achieved the highest performance.

## Model Explainability
The SHAP library was used to understand the importance of each feature in the model's predictions. This helps in understanding how the model works and which features are most influential in predicting the target variable.

