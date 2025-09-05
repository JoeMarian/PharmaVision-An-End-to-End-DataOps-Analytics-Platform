# PharmaVision-An-End-to-End-DataOps-Analytics-Platform

This project is a comprehensive solution for pharmaceutical data analysis, built as a dynamic Streamlit dashboard. It showcases an entire DataOps pipeline from raw data to actionable business intelligence and AI-driven insights.

Features
Data Ingestion & ETL: The pipeline begins with a robust ETL (Extract, Transform, Load) process that handles a sample pharma dataset. It automatically detects and cleans data, including removing duplicates and handling missing values, to ensure data quality.

Data Quality & Validation: A dedicated data quality panel provides key metrics such as missing values, row counts, and duplicate records. Simple validation rules are applied to identify and flag inconsistencies in the data.

Business Intelligence (BI): The interactive Streamlit dashboard offers various BI features for data exploration. Users can apply filters for drugs, regions, and dates to generate real-time visualizations like sales trends over time, top-performing drugs, and regional sales distribution.

AI/ML Insights: The project integrates several machine learning models to provide deeper analysis:

Sales Forecasting: Uses Prophet (with a Linear Regression fallback) to predict future sales trends.

Anomaly Detection: Employs Isolation Forest to automatically identify unusual spikes or dips in sales and adverse events.

Drug Clustering: Groups drugs based on sales and adverse events using K-Means clustering to identify potential risk categories.

Automated Reporting: A unique feature of the dashboard is its ability to generate a client-ready PDF report with a single click, summarizing all key KPIs and insights.

Orchestration: A Prefect workflow is included to automate the ETL and QC processes, ensuring the data pipeline is reliable and repeatable.

Technologies
Data Analysis & Manipulation: Python, Pandas, NumPy

Visualization: Streamlit, Plotly

Machine Learning: Scikit-learn, Prophet, TensorFlow (for optional LSTM)

Automation: Prefect

Reporting: ReportLab

How to Run
Clone the repository:

git clone [https://github.com/JoeMarian/PharmaVision-An-End-to-End-DataOps-Analytics-Platform.git](https://github.com/JoeMarian/PharmaVision-An-End-to-End-DataOps-Analytics-Platform.git)
cd PharmaVision-An-End-to-End-DataOps-Analytics-Platform

Install the required packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

You can now view the dashboard in your browser at http://localhost:8501.
