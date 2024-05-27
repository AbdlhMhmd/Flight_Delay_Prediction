# Flight Delay Prediction

This project aims to predict flight delays using machine learning techniques. The dataset includes flight records for January 2019 and January 2020. The analysis involves data preprocessing, exploratory data analysis (EDA), feature engineering, and building machine learning models to predict whether a flight will be delayed.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AbdlhMhmd/flight-delay-prediction.git
   cd flight-delay-prediction
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset used in this project consists of flight records for January 2019 and January 2020. The data is stored in CSV files and loaded into pandas DataFrames for processing.

- `Jan_2019_ontime.csv`
- `Jan_2020_ontime.csv`

## Data Preprocessing

1. **Read Data**:
   ```python
   import pandas as pd

   df19 = pd.read_csv('Jan_2019_ontime.csv')
   df20 = pd.read_csv('Jan_2020_ontime.csv')
   ```

2. **Merge Data**:
   ```python
   dfa = pd.concat([df19, df20]).reset_index(drop=True)
   ```

3. **Drop Irrelevant Columns**:
   ```python
   dfa.drop(columns=['OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM',
                     'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_AIRPORT_ID', 'Unnamed: 21'], inplace=True)
   ```

4. **Handle Missing Values**:
   ```python
   dfa.dropna(subset=['ARR_TIME', 'ARR_DEL15', 'DEP_TIME', 'DEP_DEL15'], inplace=True)
   ```

## Exploratory Data Analysis

1. **Visualize Missing Data**:
   ```python
   import missingno as msno
   import matplotlib.pyplot as plt

   msno.bar(dfa)
   plt.show()
   ```

2. **Crosstab Analysis**:
   ```python
   import seaborn as sns

   dep_del15_arr_del15_crosstab = pd.crosstab(index=dfa['DEP_TIME_BLK'], columns=[dfa['ARR_DEL15']])
   sns.heatmap(dep_del15_arr_del15_crosstab, cmap='viridis', annot=True, fmt='d')
   plt.show()
   ```

3. **Day of Month Analysis**:
   ```python
   arrival_delays_over_day = dfa.groupby('DAY_OF_MONTH')['ARR_DEL15'].sum().sort_values(ascending=False)
   arrival_delays_over_day.plot(kind='bar')
   plt.show()
   ```

4. **Day of Week Analysis**:
   ```python
   day_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
   dfa['DAY_NAME'] = dfa['DAY_OF_WEEK'].map(day_map)
   arrival_delays_over_day = dfa.groupby('DAY_NAME')['ARR_DEL15'].sum().reindex(day_map.values()).sort_values(ascending=False)
   arrival_delays_over_day.plot(kind='bar')
   plt.show()
   ```

5. **Top Origins and Destinations**:
   ```python
   top_10_origin_arr_del15_counts = dfa.groupby('ORIGIN')['ARR_DEL15'].value_counts().sort_values(ascending=False).head(10)
   top_10_origin_arr_del15_counts.plot(kind='bar')
   plt.show()

   top_10_dest_counts = dfa.groupby('DEST')[['DEP_DEL15', 'ARR_DEL15']].sum().nlargest(10, 'DEP_DEL15')
   print(top_10_dest_counts)
   ```

## Feature Engineering

1. **Time Column Splitting**:
   ```python
   def split_time_columns(df, column_names):
       for column_name in column_names:
           df[column_name] = df[column_name].astype(str).fillna('0000')
           hours = df[column_name].apply(lambda x: int(x[:-2]) if x[:-2] else 0)
           minutes = df[column_name].apply(lambda x: int(x[-2:]) if x[-2:] else 0)
           df[f'{column_name}_Hours'] = hours
           df[f'{column_name}_Minutes'] = minutes
       return df

   dfm = split_time_columns(dfa.copy(), ['ARR_TIME', 'DEP_TIME'])
   ```

2. **Encoding Categorical Data**:
   ```python
   from sklearn.preprocessing import LabelEncoder

   label_encoder = LabelEncoder()
   dfm['DAY_NAME'] = label_encoder.fit_transform(dfm['DAY_NAME'])
   dfm = pd.get_dummies(dfm, columns=['ORIGIN', 'DEST', 'DEP_TIME_BLK'])
   ```

## Modeling

### Logistic Regression

1. **Train-Test Split**:
   ```python
   from sklearn.model_selection import train_test_split

   X = dfm.drop(columns=['ARR_DEL15'])
   y = dfm['ARR_DEL15']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   ```

2. **Train and Evaluate**:
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score, classification_report

   logistic_classifier = LogisticRegression(random_state=42)
   logistic_classifier.fit(X_train, y_train)
   y_pred = logistic_classifier.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   print("Classification Report:\n", classification_report(y_test, y_pred))
   ```

3. **ROC Curve**:
   ```python
   from sklearn.metrics import roc_curve, auc

   y_probs = logistic_classifier.predict_proba(X_test)[:, 1]
   fpr, tpr, thresholds = roc_curve(y_test, y_probs)
   roc_auc = auc(fpr, tpr)

   plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
   plt.plot([0, 1], [0, 1], linestyle='--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic (ROC) Curve')
   plt.legend(loc='lower right')
   plt.show()
   ```

### Random Forest Classifier

1. **Train and Evaluate**:
   ```python
   from sklearn.ensemble import RandomForestClassifier

   rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
   rf_classifier.fit(X_train, y_train)
   y_pred = rf_classifier.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   print("Classification Report:\n", classification_report(y_test, y_pred))
   ```

2. **Precision-Recall Curve**:
   ```python
   from sklearn.metrics import precision_recall_curve

   precision, recall, thresholds = precision_recall_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
   plt.plot(recall, precision, label='Precision-Recall curve')
   plt.xlabel('Recall')
   plt.ylabel('Precision')
   plt.title('Precision-Recall Curve')
   plt.legend(loc='lower left')
   plt.show()
   ```

## Conclusion

This project demonstrated the use of logistic regression and random forest classifiers to predict flight delays. Both models achieved good accuracy, with the random forest classifier slightly outperforming logistic regression. Further improvements could be made by tuning the models and exploring additional features.

## License

This project is licensed under the MIT License.
