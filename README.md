# ChurnGuard - Telecom Customer Churn Prediction

ChurnGuard is a Streamlit application for estimating telecom customer churn risk. It supports manual predictions for a single customer and batch predictions from a CSV file, with risk summaries and retention recommendations.

## Requirements

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`

## Run the application

From the project directory, install dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Retrain the model

The training script reads `telecom_churn_synthetic_csv_file_.csv`, evaluates Logistic Regression, Random Forest, and Decision Tree models, and saves the production Logistic Regression model and evaluation metrics.

```powershell
python train_model.py
```

The Streamlit app loads `logisticregression_model.pkl` and displays metrics from `logisticregression_metrics.json`. Model comparison results are saved in `model_comparison_metrics.csv`.

## Batch prediction CSV

Provide these columns in an uploaded CSV. `CustomerID` is optional.

```text
CustomerID,Gender,SeniorCitizen,Partner,Dependents,Tenure,PhoneService,InternetService,Contract,PaymentMethod,MonthlyCharges,TotalCharges
```

Feedback submissions are stored locally in `feedback_log.json`; this file is excluded from version control.
