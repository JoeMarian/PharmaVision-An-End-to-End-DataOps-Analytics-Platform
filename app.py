import streamlit as st
import plotly.express as px
import pandas as pd
from etl import load_data, clean_data, qc_report
from ml_models import forecast_sales, detect_anomalies, cluster_drugs, simple_linear_forecast
from report_generator import generate_pdf

DATE_COL = "Month"

# --- CONFIG ---
st.set_page_config(page_title="AI Pharma DataOps", layout="wide", initial_sidebar_state="expanded")

# --- THEME ---
st.markdown(
    """
    <style>
    body {background-color: #f5f7fb;}
    .css-18e3th9 {background-color: #eaf6f6;}
    .css-1d391kg {background-color: #ffffff;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- TITLE ---
st.title("💊 AI-Powered Pharma DataOps & Analytics Dashboard")
st.markdown("End-to-end ETL + QC + BI + AI — built with Streamlit")

# --- LOAD DATA ---
df = load_data()
df = clean_data(df)

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔎 Global Filters")
drug_list = ["All"] + sorted(df["Drug"].dropna().unique().tolist()) if "Drug" in df.columns else ["All"]
region_list = ["All"] + sorted(df["Region"].dropna().unique().tolist()) if "Region" in df.columns else ["All"]

selected_drug = st.sidebar.selectbox("Drug", drug_list)
selected_region = st.sidebar.selectbox("Region", region_list)

date_min = pd.to_datetime(df[DATE_COL].min()) if DATE_COL in df.columns else None
date_max = pd.to_datetime(df[DATE_COL].max()) if DATE_COL in df.columns else None
start_date = st.sidebar.date_input("Start date", date_min)
end_date = st.sidebar.date_input("End date", date_max)

# --- APPLY FILTERS ---
dff = df.copy()
if "Drug" in dff.columns and selected_drug != "All":
    dff = dff[dff["Drug"] == selected_drug]
if "Region" in dff.columns and selected_region != "All":
    dff = dff[dff["Region"] == selected_region]
if DATE_COL in dff.columns:
    dff[DATE_COL] = pd.to_datetime(dff[DATE_COL])
    dff = dff[(dff[DATE_COL] >= pd.to_datetime(start_date)) & (dff[DATE_COL] <= pd.to_datetime(end_date))]

# --- KPI CALCULATIONS ---
total_sales = int(dff["Sales"].sum()) if "Sales" in dff.columns else 0
top_drug = dff.groupby("Drug")["Sales"].sum().idxmax() if "Drug" in dff.columns else "N/A"
qc = qc_report(dff)

# Growth % (Month-over-Month)
growth = 0
if DATE_COL in dff.columns:
    monthly = dff.groupby(dff[DATE_COL].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly[DATE_COL] = monthly[DATE_COL].dt.to_timestamp()
    if len(monthly) > 1:
        growth = ((monthly.iloc[-1, 1] - monthly.iloc[-2, 1]) / monthly.iloc[-2, 1]) * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(label="💰 Total Sales", value=f"{total_sales:,}")
k2.metric(label="⭐ Top Drug", value=f"{top_drug}")
k3.metric(label="📊 Rows", value=f"{qc['rows']}")
k4.metric(label="⚠️ Missing Values", value=f"{qc['missing_values_total']}", delta=f"{qc['duplicate_rows']} duplicates")
k5.metric(label="📈 Growth (MoM)", value=f"{growth:.2f} %")

# --- TABS ---
tabs = st.tabs([
    "📈 Sales Insights",
    "🔍 Data Quality",
    "🤖 AI Models",
    "🌍 Geo Dashboard",
    "🔬 Drill-Down Explorer",
    "📑 Reports"
])

# --- TAB 1: SALES INSIGHTS ---
with tabs[0]:
    st.header("📈 Sales Insights")

    if DATE_COL in dff.columns:
        fig = px.line(dff, x=DATE_COL, y="Sales", color="Drug" if "Drug" in dff.columns else None, title="Sales Over Time")
        st.plotly_chart(fig, use_container_width=True)

    if "Drug" in dff.columns:
        st.subheader("Top Drugs by Sales")
        top = dff.groupby("Drug")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(10)
        fig2 = px.bar(top, x="Drug", y="Sales", title="Top 10 Drugs", text_auto=True, color="Sales")
        st.plotly_chart(fig2, use_container_width=True)

    if "Region" in dff.columns:
        st.subheader("Sales by Region")
        reg = dff.groupby("Region")["Sales"].sum().reset_index()
        fig3 = px.pie(reg, names="Region", values="Sales", title="Sales Distribution by Region")
        st.plotly_chart(fig3, use_container_width=True)

    if "DrugClass" in dff.columns:
        st.subheader("Sales by Drug Class")
        cls = dff.groupby("DrugClass")["Sales"].sum().reset_index()
        fig4 = px.bar(cls, x="DrugClass", y="Sales", title="Sales by Drug Class", text_auto=True)
        st.plotly_chart(fig4, use_container_width=True)

# --- TAB 2: DATA QUALITY ---
with tabs[1]:
    st.header("🔍 Data Quality & QC")
    st.json(qc)

    mv = (dff.isnull().mean() * 100).round(2).reset_index()
    mv.columns = ["column", "missing_percent"]
    fig_mv = px.bar(mv, x="column", y="missing_percent", title="Missing % per Column")
    st.plotly_chart(fig_mv, use_container_width=True)

# --- TAB 3: AI MODELS ---
with tabs[2]:
    st.header("🤖 AI Models & Risk Analysis")

    st.subheader("Forecasting")
    try:
        forecast = forecast_sales(dff)
        figf = px.line(forecast, x="ds", y="yhat", title="Forecasted Sales (Prophet)")
        st.plotly_chart(figf, use_container_width=True)

        forecast_lr = simple_linear_forecast(dff)
        figf2 = px.line(forecast_lr, x="ds", y="yhat", title="Forecasted Sales (Linear Regression)")
        st.plotly_chart(figf2, use_container_width=True)
    except Exception as e:
        st.error("Forecasting error: " + str(e))

    st.subheader("Anomaly Detection (IsolationForest)")
    try:
        anom = detect_anomalies(dff)
        anom["is_anomaly"] = anom["is_anomaly"].astype(bool)
        fig_anom = px.scatter(anom, x=DATE_COL, y="Sales", color="is_anomaly", title="Anomalies in Sales")
        st.plotly_chart(fig_anom, use_container_width=True)
    except Exception as e:
        st.error("Anomaly detection error: " + str(e))

    st.subheader("Clustering (Drugs)")
    clusters = cluster_drugs(dff)
    if not clusters.empty:
        fig_cl = px.scatter(clusters, x="Sales", y="AdverseEvents", color="cluster", text="Drug", size="Sales", title="Drug Clusters")
        st.plotly_chart(fig_cl, use_container_width=True)

    st.subheader("Correlation Heatmap (Sales vs Adverse Events)")
    if "Sales" in dff.columns and "AdverseEvents" in dff.columns:
        corr = dff[["Sales", "AdverseEvents"]].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Sales vs Adverse Events (Risk Outliers)")
        fig_scatter = px.scatter(dff, x="Sales", y="AdverseEvents", color="Drug", trendline="ols", title="Sales vs Adverse Events with Regression Line")
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 4: GEO DASHBOARD ---
with tabs[3]:
    st.header("🌍 Geo Dashboard")
    if "Region" in dff.columns and "Sales" in dff.columns:
        geo = dff.groupby("Region")["Sales"].sum().reset_index()
        fig_map = px.choropleth(geo, locations="Region", locationmode="country names", color="Sales", title="Sales by Region (Geo View)")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No Region data available for geo analysis.")

# --- TAB 5: DRILL-DOWN ---
with tabs[4]:
    st.header("🔬 Drill-Down Explorer")
    if "Drug" in df.columns:
        drill_drug = st.selectbox("Choose a drug for detailed exploration", sorted(df["Drug"].unique()))
        drug_df = df[df["Drug"] == drill_drug].copy()

        if not drug_df.empty:
            st.subheader(f"📊 KPIs for {drill_drug}")
            d1, d2 = st.columns(2)
            d1.metric("Total Sales", f"{drug_df['Sales'].sum():,}")
            d2.metric("Adverse Events", f"{drug_df['AdverseEvents'].sum():,}" if "AdverseEvents" in drug_df.columns else "N/A")

            if DATE_COL in drug_df.columns:
                figd = px.line(drug_df, x=DATE_COL, y="Sales", title=f"Sales Over Time for {drill_drug}")
                st.plotly_chart(figd, use_container_width=True)

            # Forecast just for this drug
            try:
                forecast_d = forecast_sales(drug_df)
                figdf = px.line(forecast_d, x="ds", y="yhat", title=f"Forecasted Sales for {drill_drug}")
                st.plotly_chart(figdf, use_container_width=True)
            except:
                st.info("Not enough data for forecasting")

            # Anomalies
            try:
                anom_d = detect_anomalies(drug_df)
                anom_d["is_anomaly"] = anom_d["is_anomaly"].astype(bool)
                figda = px.scatter(anom_d, x=DATE_COL, y="Sales", color="is_anomaly", title=f"Anomalies in Sales for {drill_drug}")
                st.plotly_chart(figda, use_container_width=True)
            except:
                st.info("Not enough data for anomaly detection")

            # QC
            st.subheader("QC Report")
            st.json(qc_report(drug_df))

    else:
        st.info("Drug column not available in dataset.")

# --- TAB 6: REPORTS ---
with tabs[5]:
    st.header("📑 Reports")
    st.write("Generate a PDF summary report with KPIs.")
    if st.button("Generate PDF Report"):
        kpis = {
            "Total Sales": total_sales,
            "Top Drug": top_drug,
            "Rows": qc["rows"],
            "Missing Values": qc["missing_values_total"],
            "Growth (MoM)": f"{growth:.2f}%",
        }
        generate_pdf("pharma_report.pdf", "Pharma Analytics Report", kpis=kpis, text_lines=["Generated from Streamlit dashboard"])
        with open("pharma_report.pdf", "rb") as f:
            st.download_button("⬇️ Download Report", f, file_name="pharma_report.pdf", mime="application/pdf")
        st.success("Report generated.")
