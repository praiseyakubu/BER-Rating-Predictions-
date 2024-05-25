## Readme for BER Rating Project

This project focuses on building a model to predict Building Energy Ratings (BER) based on various features. The project involves data cleaning, preprocessing, model building, and evaluation using machine learning techniques.

### Data Cleaning and Preprocessing
- The initial dataset is loaded and cleaned to handle null values and drop irrelevant features with excessive null values.
- Categorical columns are label encoded to prepare the data for modeling.
- One-hot encoding is applied to columns with fewer categorical values.
- Normalization of numerical features is done using StandardScaler.

### Model Building
- Random Forest Classifier is implemented with parameter tuning using Grid Search to find the best 'n_estimators' value.
- Feature importance is analyzed to select significant features for a refined model.
- AdaBoost Classifier is also implemented with parameter tuning to enhance the model's performance.
- Logistic Regression with Elastic Net regularization is utilized with parameter tuning for optimal results.

### Results and Evaluation
- The project evaluates models based on the recall metric to minimize false negatives.
- Feature importance is analyzed to understand the impact of different features on the BER prediction.
- The best parameters and scores for each model are reported to guide model selection.

