# PharmaVision: An End-to-End DataOps & Analytics Platform

PharmaVision is a comprehensive solution for **pharmaceutical data analysis**, built as a dynamic **Streamlit dashboard**.  
It showcases an entire **DataOps pipeline** — from raw data to actionable **business intelligence (BI)** and **AI-driven insights**.

---

## Features

### 📥 Data Ingestion & ETL
- Robust **ETL (Extract, Transform, Load)** process handles pharma datasets.  
- Automatically detects and cleans data (duplicates removal, missing value handling).  
- Ensures high **data quality** and consistency.

### Data Quality & Validation
- Dedicated **Data Quality Panel** with key metrics:
  - Missing values
  - Row counts
  - Duplicate records  
- Validation rules to flag inconsistencies.

### Business Intelligence (BI)
- Interactive **Streamlit dashboard** for exploration.  
- Filters by **drugs, regions, and dates**.  
- Real-time visualizations:
  - Sales trends over time  
  - Top-performing drugs  
  - Regional sales distribution  

### AI/ML Insights
- Integrated **machine learning models** for deeper analysis:
  - **Sales Forecasting** → Prophet (with Linear Regression fallback).  
  - **Anomaly Detection** → Isolation Forest for unusual spikes/dips.  
  - **Drug Clustering** → K-Means based on sales & adverse events (risk segmentation).  

### Automated Reporting
- Generate a **client-ready PDF report** with one click.  
- Includes all **KPIs & insights** from the dashboard.  

### Orchestration
- **Prefect workflow** automates ETL + QC.  
- Ensures the pipeline is **reliable and repeatable**.

---

## Technologies

- **Data Analysis & Manipulation**: Python, Pandas, NumPy  
- **Visualization**: Streamlit, Plotly  
- **Machine Learning**: Scikit-learn, Prophet, TensorFlow (optional LSTM)  
- **Automation**: Prefect  
- **Reporting**: ReportLab  

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/JoeMarian/PharmaVision-An-End-to-End-DataOps-Analytics-Platform.git
cd PharmaVision-An-End-to-End-DataOps-Analytics-Platform 
```

Install the dependencies:
```bash
pip install -r requirements.txt
```
Run the Streamlit app:
```bash
streamlit run app.py
```

Open in your browser at 👉 http://localhost:8501
