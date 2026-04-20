import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Config ---
st.set_page_config(page_title="Video Games Sales Dashboard", page_icon="🎮", layout="wide")

# --- Custom CSS for Dark Theme & Neon Accents ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00ffff !important; /* Neon Cyan */
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #1f242d;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #bc13fe; /* Neon Purple on hover */
    }
    
    /* Plotly Charts Background */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f242d;
        border-radius: 5px;
        color: #ffffff;
        padding: 8px 16px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #bc13fe; /* Neon Purple */
        color: white;
    }
    
    /* Custom divider */
    hr {
        margin: 1.5em 0;
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("vgsales.csv")
        # Preprocessing similar to notebook
        # Impute Year with median (convert to int for nicer display)
        median_year = df["Year"].median()
        df["Year"] = df["Year"].fillna(median_year).astype(int)
        
        # Impute Publisher with "Unknown"
        df["Publisher"] = df["Publisher"].fillna("Unknown")
        
        # Drop duplicates if any
        df.drop_duplicates(inplace=True)
        
        return df
    except FileNotFoundError:
        st.error("Dataset 'vgsales.csv' not found. Please ensure it is in the same directory.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- Sidebar Filters ---
st.sidebar.title("🎮 Filters")
st.sidebar.markdown("Refine your view of the gaming world.")

# Filter by Year
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
years = st.sidebar.slider("Select Year Range", min_year, max_year, (1980, 2016))

# Filter by Genre
all_genres = ["All"] + sorted(df["Genre"].unique().tolist())
genre = st.sidebar.selectbox("Select Genre", all_genres)

# Filter by Platform
all_platforms = ["All"] + sorted(df["Platform"].unique().tolist())
platform = st.sidebar.selectbox("Select Platform", all_platforms)

# Applying Filters
df_filtered = df[(df["Year"] >= years[0]) & (df["Year"] <= years[1])]
if genre != "All":
    df_filtered = df_filtered[df_filtered["Genre"] == genre]
if platform != "All":
    df_filtered = df_filtered[df_filtered["Platform"] == platform]

# --- Main Dashboard ---
st.title("🚀 Global Video Game Sales Dashboard")
st.markdown("Explore trends, discover hits, and predict future sales in the gaming industry.")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
total_sales = df_filtered["Global_Sales"].sum()
top_game = df_filtered.loc[df_filtered["Global_Sales"].idxmax(), "Name"] if not df_filtered.empty else "N/A"
active_publishers = df_filtered["Publisher"].nunique()
total_games_count = df_filtered.shape[0]

with col1:
    st.metric("Total Global Sales", f"${total_sales:,.2f}M")
with col2:
    st.metric("Total Games", f"{total_games_count:,}")
with col3:
    st.metric("Top Game", top_game)
with col4:
    st.metric("Active Publishers", f"{active_publishers:,}")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "🔍 Deep Dive", "🔮 Prediction Lab", "📄 Raw Data"])

# --- Tab 1: Overview ---
with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Global Sales Trend Over Time")
        sales_by_year = df_filtered.groupby("Year")["Global_Sales"].sum().reset_index()
        fig_trend = px.line(sales_by_year, x="Year", y="Global_Sales", 
                            title="Total Sales History (Millions USD)",
                            markers=True, template="plotly_dark")
        fig_trend.update_traces(line_color="#00ffff", line_width=3)
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_chart2:
        st.subheader("Regional Sales Distribution")
        # Melting for regional comparison
        regional_sales = df_filtered[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum().reset_index()
        regional_sales.columns = ["Region", "Sales"]
        fig_pie = px.pie(regional_sales, names="Region", values="Sales", 
                         title="Market Share by Region",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         template="plotly_dark", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("Top 10 Games by Global Sales")
    top_10 = df_filtered.nlargest(10, "Global_Sales").sort_values("Global_Sales", ascending=True)
    fig_bar = px.bar(top_10, x="Global_Sales", y="Name", orientation='h',
                     title="Best Sellers in Selected Range",
                     text="Global_Sales", template="plotly_dark",
                     color="Global_Sales", color_continuous_scale="Viridis")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Tab 2: Deep Dive ---
with tab2:
    col_dd1, col_dd2 = st.columns(2)
    
    with col_dd1:
        st.subheader("Sales by Genre")
        genre_sales = df_filtered.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False).reset_index()
        fig_genre = px.bar(genre_sales, x="Genre", y="Global_Sales",
                           title="Most Popular Genres",
                           color="Global_Sales", color_continuous_scale="Purples",
                           template="plotly_dark")
        st.plotly_chart(fig_genre, use_container_width=True)
        
    with col_dd2:
        st.subheader("Platform Dominance")
        platform_sales = df_filtered.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).head(15).reset_index()
        fig_treemap = px.treemap(platform_sales, path=["Platform"], values="Global_Sales",
                                 title="Top Platforms Analysis",
                                 color="Global_Sales", color_continuous_scale="Blues",
                                 template="plotly_dark")
        st.plotly_chart(fig_treemap, use_container_width=True)
        
    st.subheader("Publisher Performance")
    pub_sales = df_filtered.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
    fig_pub = px.funnel(pub_sales, x='Global_Sales', y='Publisher', 
                        title="Top 10 Publishers by Revenue",
                        template="plotly_dark")
    st.plotly_chart(fig_pub, use_container_width=True)

# --- Tab 3: Predictive Modeling ---
with tab3:
    st.markdown("### 🤖 Predict Global Sales")
    st.info("Train a machine learning model to estimate the potential global sales of a game based on its Platform, Genre, Publisher, and Release Year.")
    
    # Model Training Section
    if st.button("Training Random Forest Model"):
        with st.spinner("Training model... This might take a moment."):
            # Prepare data for modeling
            # We use the whole dataset (df) not the filtered one for training to have more data
            model_df = df.copy()
            
            # Simple Feature Transformation for "Publisher" to reduce cardinality (keep top 50, rest 'Other')
            top_pubs = model_df["Publisher"].value_counts().nlargest(50).index
            model_df["Publisher_Cat"] = model_df["Publisher"].apply(lambda x: x if x in top_pubs else "Other")
            
            features = ["Platform", "Genre", "Publisher_Cat", "Year"]
            target = "Global_Sales"
            
            # Preprocessing Pipeline
            categorical_features = ["Platform", "Genre", "Publisher_Cat"]
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", categorical_transformer, categorical_features),
                    ("num", "passthrough", ["Year"])
                ]
            )
            
            # Model Pipeline
            model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            X = model_df[features]
            y = np.log1p(model_df[target]) # Log transform target as per notebook analysis
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Evaluation
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Save model to session state
            st.session_state["model"] = model
            st.session_state["model_r2"] = r2
            st.session_state["model_rmse"] = rmse
            st.session_state["model_trained"] = True
            st.success("Model trained successfully!")

    # Prediction Interface
    if "model_trained" in st.session_state and st.session_state["model_trained"]:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Model R² Score", f"{st.session_state['model_r2']:.4f}")
        with col_m2:
            st.metric("Model RMSE (Log)", f"{st.session_state['model_rmse']:.4f}")
            
        st.markdown("---")
        st.subheader("Make a Prediction")
        
        col_in1, col_in2 = st.columns(2)
        col_in3, col_in4 = st.columns(2)
        
        # Input widgets
        with col_in1:
            pred_platform = st.selectbox("Platform", sorted(df["Platform"].unique()))
        with col_in2:
            pred_genre = st.selectbox("Genre", sorted(df["Genre"].unique()))
        with col_in3:
            # Logic to handle user input for publisher, mapping it effectively
            # Using the top publishers list logic or allowing typed input could be complex
            # For simplicity, we use the existing list but note that "Other" logic applies internally
            pred_publisher = st.selectbox("Publisher", sorted(df["Publisher"].unique())) 
        with col_in4:
            pred_year = st.slider("Release Year", 1980, 2030, 2010)
            
        if st.button("Predict Sales"):
            # Prepare input dataframe
            # Note: We need to apply the same "Publisher_Cat" logic
            top_pubs_list = df["Publisher"].value_counts().nlargest(50).index.tolist()
            pub_cat_val = pred_publisher if pred_publisher in top_pubs_list else "Other"
            
            input_data = pd.DataFrame({
                "Platform": [pred_platform],
                "Genre": [pred_genre],
                "Publisher_Cat": [pub_cat_val],
                "Year": [pred_year]
            })
            
            # Predict
            log_prediction = st.session_state["model"].predict(input_data)[0]
            prediction = np.expm1(log_prediction) # Reverse log transform
            
            st.balloons()
            st.markdown(f"### 💰 Predicted Global Sales: **${prediction:,.2f} Million**")
            
    else:
        st.warning("⚠️ Please train the model first to make predictions.")

# --- Tab 4: Raw Data ---
with tab4:
    st.subheader("Detailed Data View")
    st.dataframe(df_filtered, use_container_width=True)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Filtered CSV",
        csv,
        "filtered_vgsales.csv",
        "text/csv",
        key='download-csv'
    )

st.markdown("---")

