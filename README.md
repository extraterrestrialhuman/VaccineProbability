### README

# Vaccine Prediction Project

## Overview

This project aims to predict the likelihood of individuals receiving the xyz and seasonal flu vaccines based on a variety of demographic, behavioral, and opinion-based features. The predictions are represented as probabilities for each individual, addressing the multilabel nature of the problem where both vaccine outcomes are predicted independently.

## Dataset

The dataset comprises several features and labels:
- **Features:**
  - Demographic information (age, education, race, sex, income, marital status, etc.)
  - Behavioral features related to flu prevention (face mask usage, hand washing, etc.)
  - Opinions on vaccine effectiveness and risks.
  - Other relevant health and social factors.
  
- **Labels:**
  - `xyz_vaccine`: Whether the respondent received the xyz flu vaccine.
  - `seasonal_vaccine`: Whether the respondent received the seasonal flu vaccine.

## Files

- `training_set_features.csv`: Features for the training set.
- `training_set_labels.csv`: Labels for the training set.
- `test_set_features.csv`: Features for the test set.
- `submission_format.csv`: Format for the submission file.

## Methodology

### Data Preprocessing

1. **Loading Data:**
   - Loaded the training features, training labels, and test features from CSV files.
   
2. **Merging Data:**
   - Merged training features and labels on the `respondent_id` to form the complete training dataset.

3. **Feature Identification:**
   - Identified binary, categorical, and numerical features from the dataset.

4. **Preprocessing Pipelines:**
   - **Numerical Data:** Imputed missing values with the median and scaled features using `StandardScaler`.
   - **Categorical Data:** Imputed missing values with the most frequent value and applied one-hot encoding.

5. **Column Transformer:**
   - Combined numerical and categorical preprocessing steps using `ColumnTransformer`.

### Model Training

1. **Model Definition:**
   - Defined a `RandomForestClassifier` with 100 estimators as the base model.
   - Used `MultiOutputClassifier` to handle the multilabel nature of the problem.

2. **Pipeline Construction:**
   - Created a pipeline combining preprocessing steps and the classifier.

3. **Train-Test Split:**
   - Split the training data into training and validation sets (80-20 split).

4. **Model Training:**
   - Trained the model using the training set.

### Model Evaluation

1. **Predictions:**
   - Obtained predicted probabilities for the validation set.

2. **ROC AUC Calculation:**
   - Calculated the ROC AUC score for each target variable (`xyz_vaccine` and `seasonal_vaccine`) using `average='macro'`.
   - Computed the mean ROC AUC score to evaluate overall model performance.

### Predictions on Test Set

1. **Test Set Predictions:**
   - Made predictions on the test set and extracted probabilities for both target variables.

2. **Submission Preparation:**
   - Prepared the submission file with `respondent_id`, `xyz_vaccine`, and `seasonal_vaccine` columns.
   - Saved the submission file as `submission_nandiniruhela.csv`.

3. **File Download:**
   - Provided a mechanism to download the submission file in Google Colab.

## Results

The model's performance on the validation set is evaluated using the ROC AUC score, which indicates how well the model distinguishes between classes. The final submission file contains probabilities for each individual receiving the xyz and seasonal flu vaccines, ready for submission to the competition platform.

## Dependencies

- pandas
- numpy
- scikit-learn
- nltk

Ensure that these packages are installed before running the code.

```sh
pip install pandas numpy scikit-learn nltk
```

## Acknowledgements

Special thanks to the data providers and IITG Consulting and Analytics Club for the opportunity to work on this interesting and impactful problem.
