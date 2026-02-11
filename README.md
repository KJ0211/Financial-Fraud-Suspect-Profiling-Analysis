# Financial-Fraud-Suspect-Profiling-Analysis
This project uses Python and Pandas to analyze transaction records and identify high-risk individuals based on spending patterns and statistical outliers.

# Key Features:


Data Cleaning: Handled missing values in transaction categories and converted date strings to datetime objects.


Outlier Detection: Used the Interquartile Range (IQR) method to identify statistically significant transaction anomalies.


Feature Engineering: Created a spending_intensity metric to find suspects with abnormal spending-to-transaction ratios.


Risk Profiling: Shortlisted suspects who had more than 20 transactions and high-value spending behaviors.
