# Customer Churn Prediction

This project predicts whether a telecom customer will churn based on:
- Monthly Charges
- Tenure (in months)
- Number of Support Calls
- Contract Type (Month-to-month, One year, Two year)

## How to Run

1. Install dependencies:
   pip install streamlit pandas scikit-learn

2. Run the app:
   streamlit run app.py

## Files Used

- app.py → Streamlit application
- model.pkl → Trained Random Forest model
- WA_Fn-UseC_-Telco-Customer-Churn.csv → Original dataset
- plots/ → Visualizations used for analysis

## Visualizations

- plots/churn_distribution.png  
  ➤ Shows the number of customers who churned vs. those who stayed. Useful to understand class imbalance.

- plots/churn_gender.png  
  ➤ Compares churn rates across male and female customers to detect any gender-based trends.

- plots/heatmap.png  
  ➤ Correlation heatmap between numerical features like tenure, monthly charges, and total charges.

- plots/pie.png  
  ➤ Pie chart showing distribution of payment methods (e.g., credit card, electronic check, etc.)

- plots/roc.png  
  ➤ ROC curve showing model performance. AUC score indicates how well the model distinguishes churn vs non-churn.

## Conclusion

This project demonstrates an end-to-end **churn prediction pipeline** using a Random Forest Classifier.  
The Streamlit app allows users to input customer details and get real-time predictions, while the visualizations provide insight into churn behavior.  
It can help telecom providers identify and retain customers who are likely to churn.

---