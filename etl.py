# etl.py
import pandas as pd
import numpy as np

DATE_COL = None

def _detect_date_column(df):
    for c in ["Date","Time","Timestamp","Month"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    return df.columns[0]

def load_data(path="data/PharmaDrugSales.csv"):
    df = pd.read_csv(path, low_memory=False)
    date_col = _detect_date_column(df)
    global DATE_COL
    DATE_COL = date_col
    try:
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m", errors="coerce")

    except Exception:
        pass
    df.attrs["__date_col__"] = date_col
    return df

def clean_data(df):
    # drop fully empty rows
    df = df.dropna(how="all").copy()
    # strip whitespace in string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    # detect numeric drug columns (wide format)
    meta_cols = set([df.attrs.get("__date_col__", "Date"), "Year", "Month", "Region", "Day", "Hour", "Time"])
    candidate_cols = [c for c in df.columns if c not in meta_cols]
    numeric_candidates = []
    for c in candidate_cols:
        try:
            if pd.to_numeric(df[c].dropna().iloc[:5], errors="coerce").notnull().any():
                numeric_candidates.append(c)
        except Exception:
            continue
    if len(numeric_candidates) > 0:
        # treat as wide format: melt numeric drug columns to long
        id_vars = [df.attrs.get("__date_col__", "Date")]
        if "Region" in df.columns:
            id_vars.append("Region")
        long = df.melt(id_vars=id_vars, value_vars=numeric_candidates, var_name="Drug", value_name="Sales")
        long["Sales"] = pd.to_numeric(long["Sales"], errors="coerce").fillna(0)
        # AdverseEvents: if present attach, otherwise 0
        if "AdverseEvents" in df.columns:
            long["AdverseEvents"] = pd.to_numeric(df["AdverseEvents"], errors="coerce").fillna(0)
        else:
            long["AdverseEvents"] = 0
        long = long.rename(columns={df.attrs.get("__date_col__", "Date"):"Date"})
        return long
    else:
        # tidy format: ensure numeric columns exist
        if "Sales" in df.columns:
            df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
        if "AdverseEvents" in df.columns:
            df["AdverseEvents"] = pd.to_numeric(df["AdverseEvents"], errors="coerce").fillna(0)
        return df

def qc_report(df):
    report = {}
    report["rows"] = len(df)
    report["columns"] = len(df.columns)
    report["missing_values_total"] = int(df.isnull().sum().sum())
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["missing_percent_per_column"] = (df.isnull().mean() * 100).round(2).to_dict()
    if "Sales" in df.columns:
        report["invalid_sales_count"] = int((df["Sales"] < 0).sum())
    else:
        report["invalid_sales_count"] = 0
    return report

def compute_launch_month(df):
    """
    Compute first month a drug appears with non-zero sales. Returns DataFrame with columns: Drug, launch_date
    """
    if "Drug" not in df.columns:
        return pd.DataFrame(columns=["Drug","launch_date"])
    tmp = df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
    tmp = tmp.dropna(subset=["Date"])
    first = tmp[tmp["Sales"] > 0].groupby("Drug")["Date"].min().reset_index().rename(columns={"Date":"launch_date"})
    return first
