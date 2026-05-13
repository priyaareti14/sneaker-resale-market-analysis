
## Business Question: What factors drive sneaker resale prices above retail on the secondary market?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE



# =============================================================================
# STEP 1 - LOAD AND INSPECT DATA
# =============================================================================

# Dataset: StockX Data Contest 2019, available on Kaggle
#https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest

df = pd.read_csv('StockX-Data-Contest-2019-3.csv')

print(df.head())
print(df.shape)         # (99956, 8)
print(df.columns)
print(df.isnull().sum())  # no missing values
print(df.info())
print(df.describe())



# =============================================================================
# STEP 2 - DATA CLEANING
# =============================================================================

# Convert dates to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Release Date'] = pd.to_datetime(df['Release Date'])

# Remove $ and commas from price columns, convert to float
df['Sale Price'] = df['Sale Price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)
df['Retail Price'] = df['Retail Price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)

print(df.info())  # confirm data types are correct



# =============================================================================
# STEP 3 - FEATURE ENGINEERING
# =============================================================================

# Days between release and sale
df['Days_Since_Release'] = (df['Order Date'] - df['Release Date']).dt.days

# Markup in dollars and percent
df['Markup_Dollar'] = df['Sale Price'] - df['Retail Price']
df['Markup_Pct'] = (df['Markup_Dollar'] / df['Retail Price']) * 100

# Log transformation of sale price to handle skew
df['Log_Sale_Price'] = np.log(df['Sale Price'])

# Dummy variable for Brand (0 = Yeezy, 1 = Off-White)
df = pd.get_dummies(df, columns=['Brand'], drop_first=True)

# Extract year and month from dates
df['Sale_Year'] = df['Order Date'].dt.year
df['Sale_Month'] = df['Order Date'].dt.month
df['Release_Year'] = df['Release Date'].dt.year

# Target variable: 1 = sold above retail, 0 = sold at or below retail
df['Above_Retail'] = (df['Sale Price'] > df['Retail Price']).astype(int)

# Price category based on retail price
def price_category(price):
    if price < 150:
        return 'Budget'
    elif price < 220:
        return 'Mid-Range'
    else:
        return 'Premium'

df['Price_Category'] = df['Retail Price'].apply(price_category)

print(df.columns)
print(df.describe())


#Saving dataset
df.to_csv('StockX_Cleaned.csv', index=False)



# =============================================================================
# STEP 4 - EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

# --- Plot 1: Distribution of Sale Price ---
plt.figure()
plt.hist(df['Sale Price'], bins=50, color='steelblue', edgecolor='black')
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price ($)')
plt.ylabel('Count')
plt.show()

# --- Plot 2: Above Retail Distribution ---
above_counts = df['Above_Retail'].value_counts()
plt.figure()
plt.bar(['Above Retail', 'Below/At Retail'], above_counts.values, color=['mediumseagreen','tomato'])
plt.title('Above Retail vs Below/At Retail')
plt.ylabel('Count')
plt.show()

print('Above_Retail distribution:')
print(df['Above_Retail'].value_counts())
print(df['Above_Retail'].value_counts(normalize=True).round(4) * 100)

# --- Plot 3: Average Markup % by Brand ---
df['Brand'] = df['Brand_Off-White'].map({True: 'Off-White', False: 'Yeezy'})
brand_markup = df.groupby('Brand')['Markup_Pct'].mean()
plt.figure()
plt.bar(brand_markup.index, brand_markup.values, color=['royalblue', 'coral'])
plt.title('Average Markup % by Brand')
plt.xlabel('Brand')
plt.ylabel('Average Markup (%)')
plt.show()

# --- Plot 4: Markup % by Price Category ---
cat_markup = df.groupby('Price_Category')['Markup_Pct'].mean()
plt.figure()
plt.bar(cat_markup.index, cat_markup.values, color=['gold', 'lightcoral', 'skyblue'])
plt.title('Average Markup % by Price Category')
plt.xlabel('Price Category')
plt.ylabel('Average Markup (%)')
plt.show()

# --- Plot 5: Correlation Heatmap ---
num_cols = ['Sale Price', 'Retail Price', 'Shoe Size', 'Days_Since_Release',
            'Markup_Pct', 'Sale_Year', 'Sale_Month']
plt.figure(figsize=(10, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# =============================================================================
# STEP 5 - LOGISTIC REGRESSION (Predict: Above_Retail)
# =============================================================================

# Define X and y
X = df[['Retail Price', 'Shoe Size', 'Days_Since_Release',
        'Brand_Off-White', 'Sale_Year', 'Sale_Month', 'Release_Year']]
y = df['Above_Retail']

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle class imbalance (only on training data)
sm = SMOTE(random_state=42)
x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train)

print('Before SMOTE:', y_train.value_counts().to_dict())
print('After SMOTE:', y_train_sm.value_counts().to_dict())

# Fit logistic regression model
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(x_train_sm, y_train_sm)

# Predictions
y_pred_lr = lr.predict(x_test)
y_prob_lr = lr.predict_proba(x_test)[:, 1]

# Results
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print('ROC-AUC Score:', round(roc_auc_score(y_test, y_prob_lr), 4))

# classification report
#              precision    recall  f1-score   support
#           0       0.03      0.94      0.05       254
#           1       1.00      0.69      0.82     29733
#    accuracy                           0.69     29987
#   macro avg       0.51      0.82      0.43     29987
#weighted avg       0.99      0.69      0.81     29987

#confusion matrix:
    # [  240    14]
    # [ 9177 20556]

# ROC-AUC Score: 0.7841


# --- Confusion Matrix Heatmap: Logistic Regression ---
cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Below Retail', 'Above Retail'],
            yticklabels=['Below Retail', 'Above Retail'])
plt.title('Logistic Regression: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# --- ROC Curve Graph ---
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_prob_lr)
plt.title('Logistic Regression ROC Curve')
plt.savefig('roc_lr.png', dpi=150, bbox_inches='tight')



# =============================================================================
# STEP 6 - LINEAR REGRESSION (Predict: Sale Price)
# =============================================================================

X2 = df[['Retail Price', 'Shoe Size', 'Days_Since_Release',
          'Brand_Off-White', 'Sale_Year', 'Sale_Month', 'Release_Year']]
y2 = df['Sale Price']

x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)

# Fit linear regression
linreg = LinearRegression()
linreg.fit(x_train2, y_train2)

# Predictions
y_pred_lin = linreg.predict(x_test2)

# Results
print('R-squared:', round(r2_score(y_test2, y_pred_lin), 4))
print('RMSE:', round(np.sqrt(mean_squared_error(y_test2, y_pred_lin)), 4))
# R-squared: 0.4437
# RMSE: 191.8528


# Coefficients
coef_df2 = pd.DataFrame({'Feature': X2.columns, 'Coefficient': linreg.coef_})
print(coef_df2.sort_values('Coefficient', ascending=False))

#              Feature  Coefficient
#     Brand_Off-White   377.849143
#           Shoe Size     1.936015
#  Days_Since_Release     0.081269
#        Retail Price    -0.029605
#          Sale_Month   -20.355342
#        Release_Year   -94.850475
#           Sale_Year  -137.851391


# --- Actual vs Predicted Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test2, y_pred_lin, alpha=0.3, color='steelblue', edgecolors='none', s=10)
plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 
         color='red', linestyle='--', linewidth=1.5, label='Perfect Fit')
plt.title('Actual vs. Predicted Sale Price')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.legend()
plt.tight_layout()
plt.show()



# =============================================================================
# STEP 7 - DECISION TREE (Predict: Above_Retail)
# =============================================================================

# Fit decision tree on SMOTE-balanced training data
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(x_train_sm, y_train_sm)

# Predictions
y_pred_dt = dt.predict(x_test)

# Results
print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print('Decision Tree Accuracy:', round(dt.score(x_test, y_test), 4))

#              precision    recall  f1-score   support
#
#           0       0.08      0.94      0.14       254
#           1       1.00      0.91      0.95     29733
#
#    accuracy                           0.91     29987
#   macro avg       0.54      0.92      0.55     29987
#weighted avg       0.99      0.91      0.94     29987

#confusion matrix:
# [  238    16]
# [ 2793 26940]

# Decision Tree Accuracy: 0.9063


# Feature Importance
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': dt.feature_importances_})
print(importance_df.sort_values('Importance', ascending=False))

# Plot Feature Importance
plt.figure()
plt.barh(importance_df['Feature'], importance_df['Importance'], color='mediumseagreen')
plt.title('Decision Tree Feature Importance')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.show()

# Plot Decision Tree
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=list(X.columns), class_names=['Below Retail', 'Above Retail'],
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree (max_depth=5)')
plt.show()



# --- Confusion Matrix Heatmap: Decision Tree ---
cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Below Retail', 'Above Retail'],
            yticklabels=['Below Retail', 'Above Retail'])
plt.title('Decision Tree: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()




