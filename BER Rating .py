
# Import all packages need to clean and prepare the data and build the model
import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame, Series, get_dummies
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
label_encoder = LabelEncoder()


BER_dataset = read_csv('BER_dataset.csv')


# Information about the DataFrame
summary = BER_dataset.describe()
num_rows, num_columns = BER_dataset.shape

# Null values in each column
null_values = BER_dataset.isnull().sum()

# Display the summary to check the number of rows, columns, and the null values
print("Data Summary:")
print("Number of Rows:", num_rows)
print("Number of Columns:", num_columns)
print("\nNull Values for Each Column:")
for column, null_count in null_values.items():
    print(f"{column}: {null_count}")

# Dropping every feature that has more than 60,000 null values
BER_Cleaned = BER_dataset.drop([
    'ApertureArea',
    'ZeroLossCollectorEff',
    'CollectorHeatLossCoEff',
    'AnnualSolarRadiation',
    'OvershadingFactor',
    'SolarStorageVolume',
    'VolumeOfPreHeatStore',
    'ElectricityConsumption',
    'gsdHSSupplHeatFraction',
    'gsdHSSupplSystemEff',
    'DistLossFactor',
    'CHPUnitHeatFraction',
    'CHPSystemType',
    'CHPElecEff',
    'CHPHeatEff',
    'CHPFuelType',
    'SupplHSFuelTypeID',
    'gsdSHRenewableResources',
    'gsdWHRenewableResources',
    'SolarHeatFraction',
    'DeliveredEnergySupplementaryWater',
    'CO2SupplementaryWater',
    'FirstEnerProdComment',
    'FirstEnerConsumedComment',
    'SecondEnerProdComment',
    'SecondEnerConsumedComment',
    'ThirdEnerProdComment',
    'ThirdEnerConsumedComment',
    'FirstBoilerFuelType',
    'FirstHeatGenPlantEff',
    'FirstPercentageHeat',
    'SecondBoilerFuelType',
    'SecondHeatGenPlantEff',
    'SecondPercentageHeat',
    'ThirdBoilerFuelType',
    'ThirdHeatGenPlantEff',
    'ThirdPercentageHeat',
    'SolarSpaceHeatingSystem',
    'TotalPrimaryEnergyFact',
    'TotalCO2Emissions',
    'FirstWallDescription',
    'SecondWallType_Description',
    'SecondWallDescription',
    'SecondWallArea',
    'SecondWallUValue',
    'SecondWallIsSemiExposed',
    'SecondWallAgeBandId',
    'SecondWallTypeId',
    'ThirdWallType_Description',
    'ThirdWallDescription',
    'ThirdWallArea',
    'ThirdWallUValue',
    'ThirdWallIsSemiExposed',
    'ThirdWallAgeBandId',
    'ThirdWallTypeId',
    'RER',
    'RenewEPnren',
    'RenewEPren',
    'CPC',
    'EPC',
    'MPCDERValue',  # This is dropped because it is almost similar as CO2Rating
    'BerRating',  # This is dropped because it is the same as Energy Rating column
    'DateOfAssessment'], axis=1)  # Date is not important in this case so it's dropped

# Replacing null values with mode for each column
for column in BER_Cleaned.columns:  # Using for loop to check each column
    # Check if the column has null values
    if BER_Cleaned[column].isnull().any():  # Check each null value in each column
        # Calculate the mode for the current column
        mode_column = BER_Cleaned[column].mode()[0]
        # Fill missing values in the current column with the mode
        BER_Cleaned[column].fillna(mode_column, inplace=True)  # Replace the null values with mode of each column

# Label Encoding for a column that has multiple categorical values
# Get dummies are not used for these features because the whole dataset will be too complex
BER_Cleaned['CountyName'] = label_encoder.fit_transform(BER_Cleaned['CountyName'])
BER_Cleaned['DwellingTypeDescr'] = label_encoder.fit_transform(BER_Cleaned['DwellingTypeDescr'])
BER_Cleaned['MainSpaceHeatingFuel'] = label_encoder.fit_transform(BER_Cleaned['MainSpaceHeatingFuel'])
BER_Cleaned['MainWaterHeatingFuel'] = label_encoder.fit_transform(BER_Cleaned['MainWaterHeatingFuel'])
BER_Cleaned['VentilationMethod'] = label_encoder.fit_transform(BER_Cleaned['VentilationMethod'])
BER_Cleaned['MainSpaceHeatingFuel'] = label_encoder.fit_transform(BER_Cleaned['MainSpaceHeatingFuel'])
BER_Cleaned['PrimaryCircuitLoss'] = label_encoder.fit_transform(BER_Cleaned['PrimaryCircuitLoss'])
BER_Cleaned['PurposeOfRating'] = label_encoder.fit_transform(BER_Cleaned['PurposeOfRating'])
BER_Cleaned['FirstWallType_Description'] = label_encoder.fit_transform(BER_Cleaned['FirstWallType_Description'])
BER_Cleaned['SA_Code'] = label_encoder.fit_transform(BER_Cleaned['SA_Code'])

# One hot encoding for a column that has less than 3 types of categorical values and for target label
# This label needs to be encoded this way because BER Rating of B3 and above is considered high energy efficiency while below that is considered low energy efficiency
BER_Cleaned['EnergyRating'] = BER_Cleaned['EnergyRating'].map({'A1': 1, 'A2': 1, 'A3': 1, 'B1': 1, 'B2': 1, 'B3': 1,
                                                               'C1': 0, 'C2': 0, 'C3': 0, 'D1': 0, 'D2': 0, 'E1': 0,
                                                               'E2': 0, 'F ': 0, 'G ': 0})
BER_Cleaned['MultiDwellingMPRN'] = BER_Cleaned['MultiDwellingMPRN'].map({'YES': 1, 'NO': 0})
BER_Cleaned['DraftLobby'] = BER_Cleaned['DraftLobby'].map({'YES': 1, 'NO': 0})
BER_Cleaned['PermeabilityTest'] = BER_Cleaned['PermeabilityTest'].map({'YES': 1, 'NO': 0})
BER_Cleaned['CHBoilerThermostatControlled'] = BER_Cleaned['CHBoilerThermostatControlled'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['OBBoilerThermostatControlled'] = BER_Cleaned['OBBoilerThermostatControlled'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['WarmAirHeatingSystem'] = BER_Cleaned['WarmAirHeatingSystem'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['UndergroundHeating'] = BER_Cleaned['UndergroundHeating'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['StorageLosses'] = BER_Cleaned['StorageLosses'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['ManuLossFactorAvail'] = BER_Cleaned['ManuLossFactorAvail'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['SolarHotWaterHeating'] = BER_Cleaned['SolarHotWaterHeating'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['ElecImmersionInSummer'] = BER_Cleaned['ElecImmersionInSummer'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['CylinderStat'] = BER_Cleaned['CylinderStat'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['CombinedCylinder'] = BER_Cleaned['CombinedCylinder'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['SWHPumpSolarPowered'] = BER_Cleaned['SWHPumpSolarPowered'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['ChargingBasisHeatConsumed'] = BER_Cleaned['ChargingBasisHeatConsumed'].map({'YES': 1, 'NO ': 0})
BER_Cleaned['ThermalMassCategory'] = BER_Cleaned['ThermalMassCategory'].map({'High                ': 4,
                                                                             'Medium-high         ': 3,
                                                                             'Medium              ': 2,
                                                                             'Medium-low          ': 1,
                                                                             'Low                 ': 0})
BER_Cleaned['FirstWallIsSemiExposed'] = BER_Cleaned['FirstWallIsSemiExposed'].map({'Yes': 1, 'No': 0})
BER_Cleaned['OBPumpInsideDwelling'] = BER_Cleaned['OBPumpInsideDwelling'].map({'YES': 1, 'NO ': 0})

# Encoding using Get_dummies
BER_Cleaned = get_dummies(BER_Cleaned, columns=['TypeofRating', 'StructureType', 'SuspendedWoodenFloor', 'CombiBoiler',
                                                'KeepHotFacility', 'InsulationType', 'PredominantRoofType',
                                                'FirstEnergyType_Description', 'SecondEnergyType_Description',
                                                'ThirdEnergyType_Description'], drop_first=True)

# Check if the null values of each column has been filled
# Information about the DataFrame
summary = BER_Cleaned.describe()
num_rows, num_columns = BER_Cleaned.shape

# Null values in each column
null_values = BER_Cleaned.isnull().sum()

# Display the summary to recheck if there is a null value contain in each column
print("Data Summary:")
print("Number of Rows:", num_rows)
print("Number of Columns:", num_columns)
print("\nNull Values for Each Column:")
for column, null_count in null_values.items():
    print(f"{column}: {null_count}")

# Dividing dataset into label and feature sets
X = BER_Cleaned.drop('EnergyRating', axis=1)  # Features
Y = BER_Cleaned['EnergyRating']  # Labels
print(type(X))
print(type(Y))
print(X.shape)  # Check how many column in X
print(Y.shape)   # Check how many column in Y

# Normalizing numerical features using standard scaler so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Implementing Random Forest Classifier
# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
Random_forest = Pipeline([
        ('balancing', SMOTE(random_state=101)),
        ('classification', RandomForestClassifier(criterion='entropy', max_features='sqrt', random_state=101))
    ])
grid_param_RF = {'classification__n_estimators': [15, 20, 30, 40]}

RF_GS = GridSearchCV(estimator=Random_forest, param_grid=grid_param_RF, scoring='recall', cv=5)  # Recall is used to minimise the false negative

RF_GS.fit(X_scaled, Y)

best_parameters_RF1 = RF_GS.best_params_
print(best_parameters_RF1)

best_result_RF1 = RF_GS.best_score_   # Mean cross-validated score of the best_estimator
print(best_result_RF1)

featimp_RF1 = pd.Series(RF_GS.best_estimator_.named_steps["classification"].feature_importances_, index=list(X)).sort_values(ascending=False)  # Getting feature importances list for the best model
print(featimp_RF1)

# Selecting features with higher significance and redefining feature set
X_impfeature = BER_Cleaned[['CO2Rating', 'HeatSystemControlCat', 'FirstWallUValue', 'TempFactorMultiplier', 'Year_of_Construction']]

feature_scaler = StandardScaler()
X_scaled_impfeature = feature_scaler.fit_transform(X_impfeature)

# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
Random_forest_impfeature = Pipeline([
        ('balancing', SMOTE(random_state=101)),
        ('classification', RandomForestClassifier(criterion='entropy', max_features='sqrt', random_state=101))
    ])
grid_param_RFbestimp = {'classification__n_estimators': [100, 150, 200, 250]}

RFimpfeature_GS = GridSearchCV(estimator=Random_forest_impfeature, param_grid=grid_param_RFbestimp, scoring='recall', cv=5)

RFimpfeature_GS.fit(X_scaled_impfeature, Y)

best_parameters_RF2 = RFimpfeature_GS.best_params_
print(best_parameters_RF2)

best_result_RF2 = RFimpfeature_GS.best_score_
print(best_result_RF2)

# Implementing AdaBoost
# Tuning the AdaBoost parameter 'n_estimators' and implementing cross-validation using Grid Search
Adaboost = Pipeline([
        ('balancing', SMOTE(random_state=101)),
        ('classification', AdaBoostClassifier(random_state=1))
    ])
grid_param_AB = {'classification__n_estimators': [10, 50, 80, 100, 150]}

Adaboost_GS = GridSearchCV(estimator=Adaboost, param_grid=grid_param_AB, scoring='recall', cv=5)

Adaboost_GS.fit(X_scaled, Y)

best_parameters_Adaboost1 = Adaboost_GS.best_params_
print(best_parameters_Adaboost1)

best_result_Adaboost1 = Adaboost_GS.best_score_  # Mean cross-validated score of the best_estimator
print(best_result_Adaboost1)

featimp_Adaboost1 = pd.Series(Adaboost_GS.best_estimator_.named_steps["classification"].feature_importances_, index=list(X)).sort_values(ascending=False)  # Getting feature importances list for the best model
print(featimp_Adaboost1)

# Implementing Logistic Regression
# Tuning eta0, max_iter, alpha, and l1_ratio parameters and implementing cross-validation using Grid Search
LR_EN = Pipeline([
        ('balancing', SMOTE(random_state=101)),   # Synthetic Minority Oversampling Technique
        ('classification', SGDClassifier(loss='log_loss', penalty='elasticnet', random_state=101))
    ])
HP_LR_EN = {'classification__eta0': [.001, .01, .1, 1, 10, 100], 'classification__max_iter': [100, 500, 1000], 'classification__alpha': [.001, .01, .1, 1, 10, 100], 'classification__l1_ratio': [0, 0.3, 0.5, 0.7, 1]}

GS_EN = GridSearchCV(estimator=LR_EN, param_grid=HP_LR_EN, scoring='recall', cv=5)

GS_EN.fit(X_scaled, Y)

best_parameters = GS_EN.best_params_
print("Best parameters: ", best_parameters)
best_result = GS_EN.best_score_  # Mean cross-validated score of the best_estimator
print("Best result: ", best_result)
best_model_EN = GS_EN.best_estimator_['classification']
print("Intercept β0: ", best_model_EN.intercept_[0])  # Assuming binary classification

coefficients_df = pd.DataFrame(zip(X.columns, best_model_EN.coef_[0]),columns=['Features', 'Coefficients']).sort_values(by=['Coefficients'], ascending=True)
print(coefficients_df)
