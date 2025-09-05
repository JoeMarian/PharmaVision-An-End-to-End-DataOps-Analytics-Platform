# prefect_flow.py
"""
Simple Prefect flow to run ETL -> QC -> Export cleaned CSV.
Install Prefect: pip install prefect
Run: prefect or python prefect_flow.py
"""
from prefect import flow, task
import pandas as pd
from etl import load_data, clean_data, qc_report

@task
def run_etl(path="data/PharmaDrugSales.csv", out="data/cleaned_pharma.csv"):
    df = load_data(path)
    df2 = clean_data(df)
    df2.to_csv(out, index=False)
    return out

@task
def qc_task(path):
    df = pd.read_csv(path, low_memory=False)
    return qc_report(df)

@flow
def etl_flow():
    cleaned = run_etl()
    qc = qc_task(cleaned)
    print("QC:", qc)
    return {"cleaned": cleaned, "qc": qc}

if __name__ == "__main__":
    etl_flow()
