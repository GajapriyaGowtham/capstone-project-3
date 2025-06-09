import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mariadb

# --- Database Connection Function ---
@st.cache_resource
def connect_to_mariadb():
    try:
        conn = mariadb.connect(
            host="localhost",
            user="root",
            password="",
            database="project",
            port=3306
        )
        return conn
    except mariadb.Error as err:
        st.error(f"Database connection error: {err}")
        return None

# Load data from SQL
@st.cache_data
def load_data():
    conn = connect_to_mariadb()
    query = "SELECT * FROM cropdata"
    df_sql = pd.read_sql(query, conn)
    conn.close()
    return df_sql

df = load_data()


# Sidebar Filters
st.sidebar.header("📌 Filter Data")
area = st.sidebar.selectbox("Select Area (Country/Region)", sorted(df['Area'].unique()))
item = st.sidebar.selectbox("Select Crop Item", sorted(df[df['Area'] == area]['Item'].unique()))

df_filtered = df[(df['Area'] == area) & (df['Item'] == item)].sort_values(by='Year')

# Main title
st.title("🌾 Crop Prediction Dashboard")
st.markdown("Visualizing predicted crop data over years")

st.subheader(f"📊 Data for {item} in {area}")
st.dataframe(df_filtered)

# Line Charts
col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(df_filtered, x='Year', y='Production', title='Production over Years', markers=True)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.line(df_filtered, x='Year', y='Yield', title='Yield over Years', markers=True)
    st.plotly_chart(fig2, use_container_width=True)

st.plotly_chart(
    px.line(df_filtered, x='Year', y='Productivity', title='Productivity over Years', markers=True),
    use_container_width=True
)

# Predicted vs Actual Comparison
if 'Predicted_Production' in df.columns:
    st.subheader("🎯 Predicted vs Actual Production")
    comparison_df = df_filtered[['Year', 'Production', 'Predicted_Production']].melt(id_vars='Year')
    fig3 = px.line(comparison_df, x='Year', y='value', color='variable', markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# Model Training
X = df[['Area', 'Item', 'Year', 'Area_Harvested', 'Yield']]
y = df['Production']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'Item'])
], remainder='passthrough')

pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', LinearRegression())
])

pipeline.fit(X, y)

# Prediction Input
st.header("🎯 Predict Crop Production")

area_input = st.selectbox("Select Area (for prediction)", sorted(df['Area'].unique()), index=0)
item_input = st.selectbox("Select Crop Item (for prediction)", sorted(df[df['Area'] == area_input]['Item'].unique()), index=0)
year_input = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
area_harvested_input = st.number_input("Area Harvested (hectares)", min_value=0, value=36000)
yield_input = st.number_input("Yield (kg per hectare)", min_value=0, value=1800)

if st.button("Predict Production"):
    custom_input = pd.DataFrame([{
        'Area': area_input,
        'Item': item_input,
        'Year': year_input,
        'Area_Harvested': area_harvested_input,
        'Yield': yield_input
    }])
    prediction = pipeline.predict(custom_input)
    st.success(f"📢 Predicted Production: {prediction[0]:,.2f} tons")