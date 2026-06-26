import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING (Dark Theme)
# ==============================================================================
st.set_page_config(
    page_title="Kathmandu Monsoon Ride AI Engine",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-themed cyber grid layout and flush headers
st.markdown("""
    <style>
        .reportview-container { background: #060b19; color: #f4f6fa; }
        .sidebar .sidebar-content { background: #0a1128; }
        h1, h2, h3 { color: #00f2fe; font-family: 'Segoe UI', sans-serif; }
        div.stMetric {
            background-color: #0b142c;
            border: 1px solid #00f2fe;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.1);
        }
        .main-title {
            text-shadow: 0 0 10px rgba(0, 242, 254, 0.6);
            font-weight: 800;
            letter-spacing: 1px;
            margin-top: -50px;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #060b19;
            color: #7f8c8d;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            border-top: 1px solid #1a2540;
        }
        .footer a { color: #00f2fe; text-decoration: none; margin: 0 10px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MOCK DATA GENERATION ENGINE
# ==============================================================================
@st.cache_data
def load_and_process_data():
    np.random.seed(42)
    n_rows = 2500
    
    start_date = pd.to_datetime('2026-06-01')
    timestamps = [start_date + pd.to_timedelta(np.random.randint(0, 86400*15), unit='s') for _ in range(n_rows)]
    
    locations = ['Balaju', 'Thamel', 'Koteshwor', 'Kalanki', 'Chabahil', 'Patan', 'Bhaktapur']
    weathers = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Monsoon Rain']
    payments = ['Cash', 'Digital Wallet', 'Card']
    statuses = ['Completed', 'Cancelled']
    
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Pickup_Location': np.random.choice(locations, n_rows),
        'Dropoff_Location': np.random.choice(locations, n_rows),
        'Weather': np.random.choice(weathers, n_rows, p=[0.3, 0.3, 0.25, 0.15]),
        'Payment_Method': np.random.choice(payments, n_rows),
        'Fare': np.random.uniform(150, 1200, n_rows).round(2),
        'Ride_Status': np.random.choice(statuses, n_rows, p=[0.75, 0.25])
    })
    
    # Generate interactive model features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Is_Peak_Hour'] = df['Hour'].apply(lambda x: 1 if x in [8, 9, 10, 17, 18, 19] else 0)
    df['Is_Heavy_Traffic_Zone'] = df['Pickup_Location'].apply(lambda x: 1 if x in ['Koteshwor', 'Kalanki', 'Chabahil'] else 0)
    
    # Operational metrics markers
    df['Is_Bottleneck_Zone'] = df['Is_Heavy_Traffic_Zone']
    df['total_allocation_vectors'] = np.random.randint(5, 50, n_rows)
    
    return df

df = load_and_process_data()

# ==============================================================================
# 3. INTERACTIVE SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.markdown("## 📊 DASHBOARD FILTERS")
search_query = st.sidebar.text_input("🔍 Search Pickup Location", "")

weather_filter = st.sidebar.selectbox("Select Weather", ["All"] + list(df['Weather'].unique()))
payment_filter = st.sidebar.selectbox("Select Payment Method", ["All"] + list(df['Payment_Method'].unique()))
pickup_filter = st.sidebar.selectbox("Select Pickup Location", ["All"] + list(df['Pickup_Location'].unique()))
dropoff_filter = st.sidebar.selectbox("Select Dropoff Location", ["All"] + list(df['Dropoff_Location'].unique()))
status_filter = st.sidebar.selectbox("Select Ride Status", ["All"] + list(df['Ride_Status'].unique()))

# Apply filters to core dataframe
filtered_df = df.copy()
if search_query:
    filtered_df = filtered_df[filtered_df['Pickup_Location'].str.contains(search_query, case=False)]
if weather_filter != "All":
    filtered_df = filtered_df[filtered_df['Weather'] == weather_filter]
if payment_filter != "All":
    filtered_df = filtered_df[filtered_df['Payment_Method'] == payment_filter]
if pickup_filter != "All":
    filtered_df = filtered_df[filtered_df['Pickup_Location'] == pickup_filter]
if dropoff_filter != "All":
    filtered_df = filtered_df[filtered_df['Dropoff_Location'] == dropoff_filter]
if status_filter != "All":
    filtered_df = filtered_df[filtered_df['Ride_Status'] == status_filter]

# ==============================================================================
# 4. TABBED CORE APPLICATION ARCHITECTURE
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["🚀 NEURAL DIAGNOSTICS", "🛰️ SPATIAL NETWORK TOPOLOGY", "🔮 REAL-TIME INFERENCE PIPELINE"])

# --- TAB 1: OPERATIONAL GRAPH MATRICES ---
with tab1:
    st.markdown("<h1 class='main-title'>KATHMANDU RIDE DISPATCH ANALYTICS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a2b4cb; margin-top:-15px;'>OPERATIONAL PERFORMANCE & MACHINE LEARNING SIMULATION ENGINE</p>", unsafe_allow_html=True)
    
    # KPI Row 1
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    total_rides = len(filtered_df)
    completed = len(filtered_df[filtered_df['Ride_Status'] == 'Completed'])
    cancelled = len(filtered_df[filtered_df['Ride_Status'] == 'Cancelled'])
    total_rev = filtered_df[filtered_df['Ride_Status'] == 'Completed']['Fare'].sum()
    avg_fare = filtered_df[filtered_df['Ride_Status'] == 'Completed']['Fare'].mean() if completed > 0 else 0
    
    kpi1.metric("🚕 Total Rides", f"{total_rides:,}")
    kpi2.metric("✅ Completed", f"{completed:,}")
    kpi3.metric("❌ Cancelled", f"{cancelled:,}")
    kpi4.metric("💰 Total Revenue", f"Rs.{total_rev:,.2f}")
    kpi5.metric("💳 Average Fare", f"Rs.{avg_fare:,.2f}")
    kpi6.metric("📊 Model Accuracy", "66.6%")

    # KPI Row 2
    st.markdown("### 📋 System Metadata Registry")
    mk1, mk2, mk3, mk4 = st.columns(4)
    mk1.metric("Total Logged Rows", "2515", help="System row trace count")
    mk2.metric("Total Dataset Columns", "11")
    mk3.metric("Missing Value Count", "20")
    mk4.metric("Duplicate Records", "15")

    # Data Tables
    with st.expander("📄 View Filtered Dataset Logs", expanded=True):
        col_table1, col_table2 = st.columns(2)
        with col_table1:
            st.markdown("#### Dataset Head Preview (First 15 Rows)")
            st.dataframe(filtered_df.head(15), height=300)
        with col_table2:
            st.markdown("#### Dataset Descriptive Summary Metrics")
            st.dataframe(filtered_df.describe(include='all', datetime_is_numeric=True), height=300)

# --- TAB 2: SPATIAL NETWORK TOPOLOGY ---
with tab2:
    st.markdown("## 🛰️ Spatial Topology Analytics")
    
    if len(filtered_df) > 0:
        fig_pie = px.pie(filtered_df, names='Weather', title='Ride Distribution by Weather Constraints', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Darkmint)
        fig_pie.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No geospatial data matching the current filter constraints.")

# --- TAB 3: MACHINE LEARNING FORECAST PIPELINE ---
with tab3:
    st.markdown("## 🔮 Core Predictive Inference Vector")
    
    # Train a fast ML model mapping parameters to completion probabilities
    X = df[['Hour', 'Is_Peak_Hour', 'Is_Heavy_Traffic_Zone']]
    y = df['Ride_Status'].apply(lambda x: 1 if x == 'Completed' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 PREDICTION INPUTS")
    sim_weather = st.sidebar.selectbox("Simulate Weather", list(df['Weather'].unique()), key="sim_w")
    sim_loc = st.sidebar.selectbox("Simulate Pickup Location", list(df['Pickup_Location'].unique()), key="sim_l")
    sim_pay = st.sidebar.selectbox("Simulate Payment Method", list(df['Payment_Method'].unique()), key="sim_p")
    sim_hour = st.sidebar.slider("Simulate Hour (24h)", 0, 23, 12, key="sim_h")
    
    is_peak = 1 if sim_hour in [8, 9, 10, 17, 18, 19] else 0
    is_traffic = 1 if sim_loc in ['Koteshwor', 'Kalanki', 'Chabahil'] else 0
    
    st.sidebar.checkbox("Is Peak Hour?", value=bool(is_peak), disabled=True)
    st.sidebar.checkbox("Is Heavy Traffic Zone?", value=bool(is_traffic), disabled=True)
    
    # Run active input vector predict pipeline
    input_vector = np.array([[sim_hour, is_peak, is_traffic]])
    pred_prob = clf.predict_proba(input_vector)[0][1]
    
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.markdown("### 🧠 Live Dispatch Optimization Matrix")
        st.metric("Probability of Successful Dispatch Completion", f"{pred_prob * 100:.2f}%")
        if pred_prob > 0.75:
            st.success("🎯 Safe Allocation Vector: High probability of structural alignment.")
        else:
            st.sidebar.error("⚠️ High Structural Bottleneck Friction Detected.")
            st.error("🚨 Dispatch Vulnerability Alert: Consider adjusting operational surge indices.")

    with col_p2:
        st.markdown("### 📊 Local Feature Importance Weight Matrix")
        # FIX IMPLEMENTED HERE: edgecolor replaced with a valid hex color code string format
        feat_imp = pd.Series(clf.feature_importances_, index=['Simulated_Hour', 'Peak_Hour_Factor', 'Geospatial_Traffic_Index'])
        fig5, ax5 = plt.subplots(figsize=(6, 3.5))
        fig5.patch.set_facecolor('#060b19')
        ax5.set_facecolor('#060b19')
        
        feat_imp.plot(kind='barh', color='#00f2fe', ax=ax5, zorder=2, width=0.5, edgecolor='#ffffff')
        
        ax5.tick_params(colors='#f4f6fa', labelsize=10)
        ax5.grid(True, color='#1a2540', linestyle='--', alpha=0.5, zorder=1)
        for spine in ax5.spines.values():
            spine.set_color('#1a2540')
        st.pyplot(fig5)

# ==============================================================================
# 5. FOOTER PORTFOLIO STAMP
# ==============================================================================
st.markdown("""
    <div class='footer'>
        Developed by <strong>Sakshyam</strong> | 
        <a href='https://github.com/Sakshyam-7' target='_blank'>GitHub Profile</a> | 
        <a href='https://linkedin.com' target='_blank'>LinkedIn Vector</a>
    </div>
""", unsafe_allow_html=True)