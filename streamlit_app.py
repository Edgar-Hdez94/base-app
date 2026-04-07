import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from io import BytesIO

 

st.set_page_config(page_title="Feeder Analysis App", layout="wide")

 

st.title("Feeder Reliability Analysis")
st.write("Sube un archivo Excel o CSV con datos de outages para analizar circuitos por subestación + feeder.")

 

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

 

def load_file(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    elif file.name.endswith(".csv"):
        return pd.read_csv(file)
    return None

 

def clean_columns(df):
    df.columns = [str(col).strip() for col in df.columns]
    return df

 

def export_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Resumen")
    return output.getvalue()

 

if uploaded_file is not None:
    try:
        df = load_file(uploaded_file)
        df = clean_columns(df)

 

        st.subheader("Preview of uploaded data")
        st.dataframe(df.head())

 

        required_cols = ["Substation", "Feeder", "Outage #", "SAIDI", "Customers Out", "Duration"]

 

        missing = [col for col in required_cols if col not in df.columns]

 

        if missing:
            st.error(f"Faltan columnas requeridas: {missing}")
            st.stop()

 

        resumen = df.groupby(["Substation", "Feeder"]).agg({
            "Outage #": "count",
            "SAIDI": "sum",
            "Customers Out": "sum",
            "Duration": "mean"
        }).reset_index()

 

        resumen = resumen.rename(columns={
            "Outage #": "fallas",
            "Customers Out": "clientes_afectados",
            "Duration": "duracion_promedio"
        })

 

        resumen["circuito"] = resumen["Substation"].astype(str) + "-" + resumen["Feeder"].astype(str)

 

        X = resumen[["SAIDI", "fallas", "clientes_afectados"]].fillna(0)

 

        n_clusters = st.sidebar.slider("Número de clusters", 2, 6, 3)

 

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        resumen["grupo"] = kmeans.fit_predict(X)

 

        st.subheader("Resumen por circuito")
        st.dataframe(resumen)

 

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total circuitos", len(resumen))
        col2.metric("Total fallas", int(resumen["fallas"].sum()))
        col3.metric("SAIDI total", round(resumen["SAIDI"].sum(), 2))
        col4.metric("Clientes afectados", int(resumen["clientes_afectados"].sum()))

 

        st.subheader("Top circuitos por SAIDI")
        top_saidi = resumen.sort_values(by="SAIDI", ascending=False).head(10)

 

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(top_saidi["circuito"], top_saidi["SAIDI"])
        ax1.set_title("Top 10 circuitos con mayor SAIDI")
        ax1.set_xlabel("Circuito")
        ax1.set_ylabel("SAIDI")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

 

        st.subheader("Clustering de circuitos")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        scatter = ax2.scatter(resumen["SAIDI"], resumen["fallas"], c=resumen["grupo"])
        ax2.set_xlabel("SAIDI")
        ax2.set_ylabel("Fallas")
        ax2.set_title("Clustering por circuito")

 

        for i in range(len(resumen)):
            ax2.text(
                resumen["SAIDI"].iloc[i],
                resumen["fallas"].iloc[i],
                resumen["circuito"].iloc[i],
                fontsize=8
            )

 

        st.pyplot(fig2)

 

        st.subheader("Filtros")
        substation_filter = st.multiselect(
            "Selecciona subestación",
            options=sorted(resumen["Substation"].astype(str).unique())
        )

 

        filtered = resumen.copy()
        if substation_filter:
            filtered = filtered[filtered["Substation"].astype(str).isin(substation_filter)]

 

        st.subheader("Tabla filtrada")
        st.dataframe(filtered.sort_values(by="SAIDI", ascending=False))

 

        excel_data = export_excel(filtered)
        st.download_button(
            label="Descargar resumen en Excel",
            data=excel_data,
            file_name="resumen_circuitos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

 

    except Exception as e:
        st.error(f"Error al procesar archivo: {e}")

 

else:
    st.info("Sube un archivo para comenzar.")
)
