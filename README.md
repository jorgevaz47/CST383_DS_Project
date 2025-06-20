# Predicting Investor-Owned Electric Utilities (CST 383 Final Project) #

## Introduction ##

This project was created for the CST 383 Data Science course at CSU Monterey Bay. Our goal was to investigate whether an electric utility company’s ownership type—specifically if it's investor-owned—could be predicted using public data on electricity rates and service types. We approached this as a binary classification problem, where the target variable indicates whether a company is investor-owned (1) or not (0). The topic was chosen because utility ownership can affect electricity pricing and service, making it important for both consumers and policymakers.

## Dataset Description ##

The dataset is from data.gov and was compiled by the National Renewable Energy Laboratory (NREL) using data from ABB, the Velocity Suite, and the U.S. Energy Information Administration’s Form 861. It includes information about utility names, ownership type, residential/commercial rates, number of customers, and zip codes.

## Project Objective ##
We aim to build a machine learning model to predict whether an electric utility company is investor-owned or not, using relevant numerical and categorical features from the dataset.

## Selection of Data ##

We used two datasets provided by the National Renewable Energy Laboratory (NREL) and hosted on data.gov. The data files were:

- `iou_zipcodes_2020.csv`: Investor-owned utilities  
- `non_iou_zipcodes_2020.csv`: All other ownership types

These files were downloaded in **June 2025** and contain utility information by ZIP code, including:

- Utility name and ID  
- Ownership type (e.g., Investor Owned, Municipal, Cooperative)  
- Service type (Bundled, Delivery, Energy)  
- Residential (`res_rate`), Commercial (`comm_rate`), and Industrial (`ind_rate`) electricity rates
  
## Features Selection ## 

The dataset was compiled from two sources, covering both investor-owned utilities (IOUs) and non-investor-owned utilities. According to the publisher, the National Renewable Energy Laboratory (NREL), the files include average rates for each utility—not average rates per ZIP code. The combined dataset contains 80,204 rows and 9 features.
There are three true numeric features: comm_rate, ind_rate, and res_rate (see Table 1-1). While the ZIP code is stored as a numeric value, it represents a categorical location code. Similarly, eiaid is an arbitrary identifier and does not carry meaningful magnitude or order.
For this project, we drop both eiaid and utility_name, as they do not provide meaningful predictive information. Instead, we focus on comm_rate, ind_rate, res_rate, and service_type as predictors. These selected features are complete and contain no missing values. Additionally, we exclude ZIP codes due to their complexity and limited predictive value in Machine Learning model.

### Data Cleaning and Feature Engineering ##

We merged both datasets and dropped irrelevant columns (`zip`, `eiaid`, `utility_name`) since they were identifiers rather than predictive features. We removed rows with any zero values in the three rate columns, as those likely indicated missing or inapplicable data.

The `service_type` column was one-hot encoded into three binary columns:
- `service_type_Bundled`
- `service_type_Delivery`
- `service_type_Energy`

This allowed us to use these categorical variables in our models.

Final dataset size after preprocessing: **72,104 rows**

## Methods & Tools ##

The analysis was conducted in a Jupyter Notebook using Python. Libraries used include:

- `pandas`, `NumPy` for data handling  
- `matplotlib`, `seaborn` for visualization  
- `scikit-learn` for machine learning models and metrics

We trained and evaluated several supervised learning models:

1. **Decision Tree Classifier** – A simple, interpretable model used as a baseline  
2. **Linear Regression** – Used to explore relationships between service type and `comm_rate`  
3. **K-Nearest Neighbors (KNN)** – A distance-based classifier used as a benchmark  
4. **Random Forest Classifier** – An ensemble model to reduce overfitting and improve performance  
5. **Naive Bayes (GaussianNB)** – A fast probabilistic classifier

We also used:
- `GridSearchCV` for Decision Tree hyperparameter tuning  
- `StandardScaler` for feature scaling (for KNN and Naive Bayes)

## Results ##

### Decision Tree ###
- Accuracy: ~81%
- Strong on majority class (Investor Owned), weak on minority classes

### Linear Regression ###
- RMSE: ~0.016
- Weak but measurable correlation between service type and `comm_rate`

### Random Forest ###
- Accuracy: ~86%
- Best-performing model overall; balanced accuracy and generalization

### KNN ###
- Accuracy: ~73–75%
- Performed reasonably well but was slower on full dataset

### Naive Bayes ###
- Accuracy: ~75%
- Efficient but affected by class imbalance

Across all models, investor-owned utilities were predicted most reliably due to the large number of such entries in the dataset.

## Discussion ##

Our results show that it is possible to predict investor ownership status with reasonable accuracy using only rate and service type data. Key observations include:

- **Class Imbalance**: Less common ownership types like "Federal" and "State" were often misclassified
- **Model Insights**: Commercial rates and service types were among the most predictive features

### Future Directions ###

- Apply resampling techniques like SMOTE or undersampling  
- Include more features like region or utility size  
- Explore unsupervised techniques to cluster utility companies

These improvements could lead to better model performance and deeper insight into utility structures across the U.S.

## Summary ##

- Goal: Predict investor ownership using rate and service type data  
- Dataset: Merged and cleaned NREL ZIP-code-level utility data  
- Methods: Classification (Decision Tree, Random Forest, KNN, Naive Bayes) and regression  
- Best Results: Random Forest (accuracy ~86%)  
- Contribution: Demonstrated that ownership type is predictable from basic public utility data
