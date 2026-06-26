import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =====================================================================
# 1. PAGE CONFIGURATION & LAYOUT ADJUSTMENTS
# =====================================================================
st.set_page_config(
    page_title="Kathmandu Ride Analytics Dashboard",
    page_icon="🚖",
    layout="wide"
)

# Pull layout elements upward and completely force default blank top headers out
st.markdown("""
    <style>
    /* Nuke default Streamlit top spacing and header gaps */
    [data-testid="stHeader"] {
        display: none !important;
    }
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: -50px !important;
    }
    section[data-testid="stSidebar"] div.st-emotion-cache-6qobix,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        padding-top: 0rem !important;
        margin-top: -40px !important;
    }
    
    /* Theme definitions */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #091026 0%, #040714 100%) !important;
        color: #e2e8f0 !important;
    }
    .sticky-header-block {
        background-color: #050915;
        padding: 12px 20px;
        border-bottom: 2px solid #00f2fe;
        border-radius: 0 0 8px 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.15);
    }
    .neon-title {
        font-family: 'Courier New', Courier, monospace;
        font-size: 28px !important;
        font-weight: 900;
        color: #00f2fe;
        text-shadow: 0 0 12px rgba(0, 242, 254, 0.6);
        letter-spacing: 2px;
        margin: 0;
    }
    .neon-subtitle {
        color: #9b5de5;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 4px 0 0 0;
    }
    .metric-box {
        background: linear-gradient(135deg, rgba(10, 18, 42, 0.8) 0%, rgba(5, 9, 24, 0.9) 100%);
        border-left: 4px solid #00f2fe;
        border-radius: 6px;
        padding: 8px 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 22px !important;
        font-weight: bold !important;
        color: #00f2fe !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #050811 !important;
        border-right: 1px solid rgba(0, 242, 254, 0.1);
    }
    .insight-card {
        background: rgba(155, 93, 229, 0.05);
        border: 1px solid rgba(155, 93, 229, 0.2);
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
    }
    
    /* Social Footer Links Layout */
    .footer-container {
        text-align: center;
        font-family: monospace;
        color: #a0aec0;
        font-size: 12px;
        padding: 20px 0;
        margin-top: 20px;
    }
    .footer-link {
        color: #00f2fe !important;
        text-decoration: none;
        font-weight: bold;
        margin: 0 10px;
        transition: color 0.3s ease;
    }
    .footer-link:hover {
        color: #9b5de5 !important;
        text-shadow: 0 0 8px rgba(155, 93, 229, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. INITIALIZE SESSION STATE
# =====================================================================
if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []

# =====================================================================
# 3. MOCK DATA GENERATOR
# =====================================================================
@st.cache_data
def generate_master_telemetry():
    np.random.seed(42)
    n = 2500
    timestamps = pd.date_range(start='2026-06-01', periods=n, freq='20min')
    
    weather_choices = ['Clear', 'Cloudy', 'Heavy Monsoon Rain']
    payment_choices = ['Cash', 'Esewa', 'Khalti']
    hubs = ['Thamel', 'Koteshwor', 'Kalanki', 'Chabahil', 'Balaju', 'Patandhoka', 'New Road', 'Maharajgunj', 'Baneshwor', 'Jawalakhel']
    
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Ride_Status': np.random.choice(['Completed', 'Cancelled'], size=n, p=[0.72, 0.28]),
        'Weather': np.random.choice(weather_choices, size=n, p=[0.40, 0.35, 0.25]),
        'Is_Peak_Hour': np.random.choice([0, 1], size=n, p=[0.65, 0.35]),
        'Is_Bottleneck_Zone': np.random.choice([0, 1], size=n, p=[0.70, 0.30]),
        'Payment_Method': np.random.choice(payment_choices, size=n, p=[0.40, 0.40, 0.20]),
        'Pickup_Location': np.random.choice(hubs, size=n, p=[0.20, 0.12, 0.15, 0.08, 0.07, 0.10, 0.13, 0.05, 0.06, 0.04]),
        'Dropoff_Location': np.random.choice(hubs, size=n, p=[0.15, 0.10, 0.18, 0.07, 0.08, 0.12, 0.11, 0.06, 0.08, 0.05]),
        'total_amount': np.random.uniform(120, 850, size=n)
    })
    
    df.loc[np.random.choice(df.index, size=12, replace=False), 'Weather'] = np.nan
    df.loc[np.random.choice(df.index, size=8, replace=False), 'total_amount'] = np.nan
    
    duplicates = df.sample(n=15, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    df['Is_Completed'] = df['Ride_Status'].apply(lambda x: 1 if x == 'Completed' else 0)
    df['Hour'] = df['Timestamp'].dt.hour
    return df

base_df = generate_master_telemetry()

# Data Clean-up
clean_df = base_df.copy()
clean_df['Weather'] = clean_df['Weather'].fillna(clean_df['Weather'].mode()[0])
clean_df['total_amount'] = clean_df['total_amount'].fillna(clean_df['total_amount'].median())
clean_df_no_dupes = clean_df.drop_duplicates()

# =====================================================================
# 4. MODEL TRAINING PIPELINE
# =====================================================================
@st.cache_resource
def execute_model_training(data):
    features = data[['Payment_Method', 'Weather', 'Is_Peak_Hour', 'Is_Bottleneck_Zone', 'Hour', 'Pickup_Location']]
    X = pd.get_dummies(features, drop_first=False)
    y = data['Is_Completed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    return model, X.columns, X_test, y_test, y_pred, y_probs

model, model_columns, X_test_matrix, y_test_vector, y_pred_vector, y_probs_vector = execute_model_training(clean_df_no_dupes)
model_accuracy = (y_pred_vector == y_test_vector).mean() * 100

# =====================================================================
# 5. SIDEBAR FILTERS (CLEAN TERMINOLOGY)
# =====================================================================
st.sidebar.markdown("<h2 style='color:#00f2fe;font-family:monospace;font-size:15px;margin-bottom:2px;margin-top:0px;'>🎛️ DASHBOARD FILTERS</h2>", unsafe_allow_html=True)

search_pickup = st.sidebar.text_input("🔍 Search Pickup Location", "").strip()

all_weathers = ['All'] + list(clean_df_no_dupes['Weather'].unique())
filter_weather = st.sidebar.selectbox("Select Weather", all_weathers)

all_payments = ['All'] + list(clean_df_no_dupes['Payment_Method'].unique())
filter_payment = st.sidebar.selectbox("Select Payment Method", all_payments)

all_pickups = ['All'] + sorted(list(clean_df_no_dupes['Pickup_Location'].unique()))
filter_pickup = st.sidebar.selectbox("Select Pickup Location", all_pickups)

all_drops = ['All'] + sorted(list(clean_df_no_dupes['Dropoff_Location'].unique()))
filter_drop = st.sidebar.selectbox("Select Dropoff Location", all_drops)

filter_peak = st.sidebar.selectbox("Select Time Period", ['All', 'Peak Hours Only', 'Standard Hours Only'])
filter_status = st.sidebar.selectbox("Select Ride Status", ['All', 'Completed', 'Cancelled'])

# Filter execution logic
f_df = clean_df_no_dupes.copy()
if search_pickup:
    f_df = f_df[f_df['Pickup_Location'].str.contains(search_pickup, case=False)]
if filter_weather != 'All':
    f_df = f_df[f_df['Weather'] == filter_weather]
if filter_payment != 'All':
    f_df = f_df[f_df['Payment_Method'] == filter_payment]
if filter_pickup != 'All':
    f_df = f_df[f_df['Pickup_Location'] == filter_pickup]
if filter_drop != 'All':
    f_df = f_df[f_df['Dropoff_Location'] == filter_drop]
if filter_peak == 'Peak Hours Only':
    f_df = f_df[f_df['Is_Peak_Hour'] == 1]
elif filter_peak == 'Standard Hours Only':
    f_df = f_df[f_df['Is_Peak_Hour'] == 0]
if filter_status != 'All':
    f_df = f_df[f_df['Ride_Status'] == filter_status]

# Prediction Parameters setup inside sidebar
st.sidebar.markdown("<hr style='border-color:rgba(0,242,254,0.1); margin:6px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='color:#9b5de5;font-family:monospace;font-size:15px;margin-bottom:2px;'>🔮 PREDICTION INPUTS</h2>", unsafe_allow_html=True)

sim_weather = st.sidebar.selectbox("Simulate Weather", sorted(clean_df_no_dupes['Weather'].unique()))
sim_location = st.sidebar.selectbox("Simulate Pickup Location", sorted(clean_df_no_dupes['Pickup_Location'].unique()))
sim_payment = st.sidebar.selectbox("Simulate Payment Method", sorted(clean_df_no_dupes['Payment_Method'].unique()))
sim_hour = st.sidebar.slider("Simulate Hour (24h)", 0, 23, 17)
sim_peak = st.sidebar.checkbox("Is Peak Hour?", value=True)
sim_bottleneck = st.sidebar.checkbox("Is Heavy Traffic Zone?")

run_prediction = st.sidebar.button("🔮 Run Neural Prediction", use_container_width=True)

# =====================================================================
# 6. HEADER LAYOUT (SNAPPED TO TOP)
# =====================================================================
st.markdown("""
    <div class="sticky-header-block">
        <h1 class="neon-title">KATHMANDU RIDE DISPATCH ANALYTICS</h1>
        <p class="neon-subtitle">Operational Performance & Machine Learning Simulation Engine</p>
    </div>
""", unsafe_allow_html=True)

# =====================================================================
# 7. BUSINESS PERFORMANCE KPI CARDS
# =====================================================================
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
total_rides = len(f_df)
completed_rides = len(f_df[f_df['Ride_Status'] == 'Completed'])
cancelled_rides = len(f_df[f_df['Ride_Status'] == 'Cancelled'])
total_revenue = f_df['total_amount'].sum()
avg_fare_val = f_df['total_amount'].mean() if total_rides > 0 else 0

with kpi1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("🚖 Total Rides", f"{total_rides:,}")
    st.markdown('</div>', unsafe_allow_html=True)
with kpi2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("✅ Completed", f"{completed_rides:,}")
    st.markdown('</div>', unsafe_allow_html=True)
with kpi3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("❌ Cancelled", f"{cancelled_rides:,}")
    st.markdown('</div>', unsafe_allow_html=True)
with kpi4:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("💰 Total Revenue", f"Rs.{total_revenue:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with kpi5:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("🎟️ Average Fare", f"Rs.{avg_fare_val:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with kpi6:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("📈 Model Accuracy", f"{model_accuracy:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

# Data Quality KPI Subsections
dq1, dq2, dq3, dq4 = st.columns(4)
with dq1:
    st.markdown(f"<div class='metric-box' style='border-left-color:#9b5de5;'><b>Total Logged Rows</b><br><span style='font-size:20px;color:#9b5de5;font-weight:bold;'>{len(base_df)}</span></div>", unsafe_allow_html=True)
with dq2:
    st.markdown(f"<div class='metric-box' style='border-left-color:#9b5de5;'><b>Total Dataset Columns</b><br><span style='font-size:20px;color:#9b5de5;font-weight:bold;'>{len(base_df.columns)}</span></div>", unsafe_allow_html=True)
with dq3:
    st.markdown(f"<div class='metric-box' style='border-left-color:#f15bb5;'><b>Missing Value Count</b><br><span style='font-size:20px;color:#f15bb5;font-weight:bold;'>{base_df.isna().sum().sum()}</span></div>", unsafe_allow_html=True)
with dq4:
    st.markdown(f"<div class='metric-box' style='border-left-color:#f15bb5;'><b>Duplicate Records</b><br><span style='font-size:20px;color:#f15bb5;font-weight:bold;'>{base_df.duplicated().sum()}</span></div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

with st.expander("📋 View Filtered Dataset Logs"):
    exp_col1, exp_col2 = st.columns([1, 1])
    with exp_col1:
        st.markdown("##### Dataset Head Preview (First 15 Rows)")
        st.dataframe(f_df.head(15), use_container_width=True)
    with exp_col2:
        st.markdown("##### Dataset Descriptive Summary Metrics")
        st.dataframe(f_df.describe(), use_container_width=True)

# =====================================================================
# 8. ANALYTICS TABS SYSTEM
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 RIDE ANALYTICS", 
    "🛰️ LOCATION ANALYSIS", 
    "🔮 SIMULATION & PREDICTIONS",
    "📊 DATA QUALITY ENGINE"
])

# TAB 1: RIDE BREAKDOWN ANALYTICS
with tab1:
    t1_c1, t1_c2, t1_c3 = st.columns(3)
    with t1_c1:
        st.markdown("##### Ride Fulfillment Breakdown")
        if len(f_df) > 0:
            fig_pie = px.pie(f_df, names='Ride_Status', color='Ride_Status',
                             color_discrete_map={'Completed': '#00f2fe', 'Cancelled': '#f15bb5'},
                             hole=0.4, template="plotly_dark")
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("No matching metrics selected.")
    with t1_c2:
        st.markdown("##### Total Rides by Payment Method")
        if len(f_df) > 0:
            pay_counts = f_df['Payment_Method'].value_counts().reset_index()
            fig_pay = px.bar(pay_counts, x='Payment_Method', y='count', color='Payment_Method',
                             color_discrete_sequence=['#00f2fe', '#9b5de5', '#f15bb5'], template="plotly_dark")
            fig_pay.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240, showlegend=False)
            st.plotly_chart(fig_pay, use_container_width=True)
    with t1_c3:
        st.markdown("##### Completion Rate by Payment Method")
        if len(f_df) > 0:
            pay_comp = f_df.groupby('Payment_Method')['Is_Completed'].mean().reset_index()
            pay_comp['Completion Rate %'] = pay_comp['Is_Completed'] * 100
            fig_pay_comp = px.bar(pay_comp, x='Payment_Method', y='Completion Rate %', 
                                  color='Payment_Method', template="plotly_dark", range_y=[0, 100])
            fig_pay_comp.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240, showlegend=False)
            st.plotly_chart(fig_pay_comp, use_container_width=True)

    st.markdown("---")
    t1_r2_c1, t1_r2_c2 = st.columns([2, 1])
    with t1_r2_c1:
        st.markdown("##### Hourly Revenue Trends")
        if len(f_df) > 0:
            hourly_data = f_df.groupby('Hour').agg({'total_amount': 'sum'}).reset_index()
            hourly_data.columns = ['Hour', 'Total Revenue']
            fig_trend = px.line(hourly_data, x='Hour', y='Total Revenue', markers=True, template="plotly_dark")
            fig_trend.update_traces(line_color='#00f2fe', marker=dict(size=6))
            fig_trend.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240)
            st.plotly_chart(fig_trend, use_container_width=True)
    with t1_r2_c2:
        st.markdown("##### Total Revenue by Weather Type")
        if len(f_df) > 0:
            w_rev = f_df.groupby('Weather')['total_amount'].sum().reset_index()
            fig_w_rev = px.bar(w_rev, x='Weather', y='total_amount', color='Weather',
                               color_discrete_sequence=['#00f2fe', '#9b5de5', '#f15bb5'], template="plotly_dark")
            fig_w_rev.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240, showlegend=False)
            st.plotly_chart(fig_w_rev, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Cancellation Matrix Drivers (% Cancelled)")
    t1_r3_c1, t1_r3_c2, t1_r3_c3, t1_r3_c4 = st.columns(4)
    with t1_r3_c1:
        st.markdown("###### Cancellations by Weather")
        if len(f_df) > 0:
            st.dataframe(f_df.groupby('Weather')['Ride_Status'].value_counts(normalize=True).unstack().style.format("{:.1%}"), use_container_width=True)
    with t1_r3_c2:
        st.markdown("###### Cancellations by Payment")
        if len(f_df) > 0:
            st.dataframe(f_df.groupby('Payment_Method')['Ride_Status'].value_counts(normalize=True).unstack().style.format("{:.1%}"), use_container_width=True)
    with t1_r3_c3:
        st.markdown("###### Cancellations by Peak Hours")
        if len(f_df) > 0:
            st.dataframe(f_df.groupby('Is_Peak_Hour')['Ride_Status'].value_counts(normalize=True).unstack().style.format("{:.1%}"), use_container_width=True)
    with t1_r3_c4:
        st.markdown("###### Cancellations by Area Segment")
        if len(f_df) > 0:
            st.dataframe(f_df.groupby('Pickup_Location')['Ride_Status'].value_counts(normalize=True).unstack().tail(3).style.format("{:.1%}"), use_container_width=True)

# TAB 2: GEOGRAPHIC LOCATION TOPOLOGY
with tab2:
    t2_c1, t2_c2 = st.columns(2)
    with t2_c1:
        st.markdown("##### Top 10 Pickup Locations (Volume)")
        if len(f_df) > 0:
            top_p = f_df['Pickup_Location'].value_counts().head(10).reset_index()
            fig_p = px.bar(top_p, x='count', y='Pickup_Location', orientation='h', template="plotly_dark")
            fig_p.update_traces(marker_color='#00f2fe')
            fig_p.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260)
            fig_p.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_p, use_container_width=True)
    with t2_c2:
        st.markdown("##### Top 10 Dropoff Locations (Volume)")
        if len(f_df) > 0:
            top_d = f_df['Dropoff_Location'].value_counts().head(10).reset_index()
            fig_d = px.bar(top_d, x='count', y='Dropoff_Location', orientation='h', template="plotly_dark")
            fig_d.update_traces(marker_color='#9b5de5')
            fig_d.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260)
            fig_d.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")
    t2_r2_c1, t2_r2_c2 = st.columns([2, 1])
    with t2_r2_c1:
        st.markdown("##### Route Network Revenue Density Matrix Map")
        if len(f_df) > 0:
            loc_heat = f_df.groupby(['Pickup_Location', 'Dropoff_Location'])['total_amount'].sum().unstack().fillna(0)
            fig_heat = px.imshow(loc_heat, color_continuous_scale='Magma', template="plotly_dark")
            fig_heat.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_heat, use_container_width=True)
    with t2_r2_c2:
        st.markdown("##### Total Generated Revenue by Pickup Location")
        if len(f_df) > 0:
            loc_yield = f_df.groupby('Pickup_Location')['total_amount'].sum().sort_values(ascending=False).reset_index()
            loc_yield.columns = ['Pickup Location Hub', 'Total Revenue Yield']
            st.dataframe(loc_yield.style.format({'Total Revenue Yield': 'Rs.{:,.2f}'}), use_container_width=True, height=320)

# TAB 3: MACHINE LEARNING SIMULATION LABORATORY
with tab3:
    st.markdown("##### ⚙️ Simulation Control Deck")
    st.write("Adjust environmental factors inside the **Prediction Inputs** panel on the left sidebar, then hit the run trigger.")
    
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    input_df['Hour'] = sim_hour
    input_df['Is_Peak_Hour'] = 1 if sim_peak else 0
    input_df['Is_Bottleneck_Zone'] = 1 if sim_bottleneck else 0
    
    for k, v in [('Weather', sim_weather), ('Payment_Method', sim_payment), ('Pickup_Location', sim_location)]:
        if f'{k}_{v}' in model_columns:
            input_df[f'{k}_{v}'] = 1
            
    prob_score = model.predict_proba(input_df)[0][1] * 100
    
    if run_prediction:
        st.session_state['prediction_history'].append({
            'Weather Condition': sim_weather,
            'Time Element': f"{sim_hour}:00",
            'Selected Location': sim_location,
            'Success Vector Score': f"{prob_score:.1f}%"
        })
        st.toast("Prediction logged directly into processing records!", icon="✅")

    t3_c1, t3_c2 = st.columns([1, 1])
    with t3_c1:
        st.markdown("##### System Prediction Probability Gauge Meter")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fulfillment Vector Probability %", 'font': {'size': 14, 'color': '#00f2fe'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#e2e8f0"},
                'bar': {'color': "#00f2fe"},
                'bgcolor': "rgba(10, 18, 42, 0.8)",
                'borderwidth': 2,
                'bordercolor': "rgba(0, 242, 254, 0.2)",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(241, 91, 181, 0.2)'},
                    {'range': [50, 75], 'color': 'rgba(155, 93, 229, 0.2)'},
                    {'range': [75, 100], 'color': 'rgba(0, 242, 254, 0.2)'}
                ]
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", margin=dict(t=30, b=10, l=20, r=20), height=240)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with t3_c2:
        st.markdown("##### Dynamic Real-time Diagnostics Reporting")
        risk_level = "HIGH RISK ENVIRONMENT" if prob_score < 50 else ("VOLATILE RUN MATRIX" if prob_score < 75 else "STABLE DISPATCH ZONE")
        color_alert = "#f15bb5" if prob_score < 50 else ("#9b5de5" if prob_score < 75 else "#00f2fe")
        
        st.markdown(f"""
            <div style="background:rgba(5,9,24,0.6); padding:15px; border-radius:6px; border:1px solid {color_alert}; height:240px;">
                <b style="color:{color_alert}; font-size:15px;">🔍 AI DISPATCH VERDICT:</b><br>
                <span style="font-size:20px; font-weight:bold; color:#e2e8f0;">Prediction: { 'RIDE SUCCESS VECTOR' if prob_score >= 50 else 'NETWORK DISPATCH FAILURE' }</span><br><br>
                <b>Confidence/Probability Value:</b> {prob_score:.2f}% Baseline Metric Accuracy Score<br>
                <b>Risk Matrix Level:</b> <span style="color:{color_alert}; font-weight:bold;">{risk_level}</span><br>
                <b>System Operational Recommendation:</b> { 'PROCEED WITH NORMAL ALLOCATION PIPELINES' if prob_score >= 75 else 'INJECT BONUS INCENTIVES OR SURGE ADJUSTMENT ROUTERS' }
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    t3_r2_c1, t3_r2_c2, t3_r2_c3 = st.columns([1, 1, 1])
    with t3_r2_c1:
        st.markdown("##### Top 10 Model Feature Importance Parameters")
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=model_columns).sort_values(ascending=True).tail(10).reset_index()
        feat_imp.columns = ['Feature Metric Name', 'Information Weight']
        fig_imp = px.bar(feat_imp, x='Information Weight', y='Feature Metric Name', orientation='h', template="plotly_dark")
        fig_imp.update_traces(marker_color='#00f2fe')
        fig_imp.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig_imp, use_container_width=True)
    with t3_r2_c2:
        st.markdown("##### Model Confusion Matrix Validation View")
        cm = confusion_matrix(y_test_vector, y_pred_vector)
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted Labels", y="True Labels"),
                           x=['Failure', 'Success'], y=['Failure', 'Success'],
                           color_continuous_scale='Purples', template="plotly_dark")
        fig_cm.update_yaxes(autorange="reversed")
        fig_cm.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig_cm, use_container_width=True)
    with t3_r2_c3:
        st.markdown("##### Receiver Operating Characteristic (ROC Curve)")
        fpr, tpr, _ = roc_curve(y_test_vector, y_probs_vector)
        roc_auc = auc(fpr, tpr)
        fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Area Under Curve (AUC = {roc_auc:.4f})',
                          labels=dict(x='False Positive Rate', y='True Positive Rate'), template="plotly_dark")
        fig_roc.add_shape(type='line', line=dict(dash='dash', color='#f15bb5'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Simulated Model Testing Run History Logs")
    st.dataframe(pd.DataFrame(st.session_state['prediction_history']).tail(10), use_container_width=True)

# TAB 4: SYSTEM AND DATA QUALITY METRICS
with tab4:
    t4_c1, t4_c2 = st.columns(2)
    with t4_c1:
        st.markdown("##### Advanced Null Matrix Mapping Reports")
        dq_report = pd.DataFrame({
            'Data Column Attribute': base_df.columns,
            'Missing Fields': base_df.isna().sum().values,
            'Unique Structural Cardinality': base_df.nunique().values,
            'Variable Array Data Encoding Type': [str(x) for x in base_df.dtypes.values]
        })
        st.dataframe(dq_report, use_container_width=True, height=260)
    with t4_c2:
        st.markdown("##### Complete Statistical Variance Matrix Map")
        numeric_cols = f_df.select_dtypes(include=np.number).columns
        stat_summary = pd.DataFrame({
            'Mean Metric Average': f_df[numeric_cols].mean(),
            'Median Vector Anchor': f_df[numeric_cols].median(),
            'Standard Error Deviation': f_df[numeric_cols].std(),
            'Variance Range Array': f_df[numeric_cols].var()
        })
        # FIXED: Resolved Matplotlib grid display configuration mismatch
        st.dataframe(stat_summary.style.format("{:.4f}"), use_container_width=True, height=260)
        
    st.markdown("---")
    st.markdown("##### Numerical Features Correlation Matrix")
    num_df = f_df.select_dtypes(include=np.number).drop(columns=['Hour', 'Is_Peak_Hour', 'Is_Bottleneck_Zone', 'Is_Completed'], errors='ignore')
    if len(num_df.columns) > 1:
        corr_matrix = num_df.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', vmin=-1, vmax=1, template="plotly_dark")
        fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.write("Insufficient dimensional properties to plot variance trends.")

# =====================================================================
# 9. COMPUTE LOG AUTOMATED INSIGHTS GENERATION ENGINE
# =====================================================================
st.markdown("---")
st.markdown("### 📊 DYNAMIC OPERATIONAL INSIGHTS HUB")
bi1, bi2, bi3, bi4 = st.columns(4)

with bi1:
    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
    st.markdown("##### 🏆 Best Pickup Location")
    if len(f_df) > 0:
        best_pick = f_df.groupby('Pickup_Location')['Is_Completed'].mean().idxmax()
        best_rate = f_df.groupby('Pickup_Location')['Is_Completed'].mean().max() * 100
        st.markdown(f"Top Hub Node: **{best_pick}**<br>Completion Rate: `{best_rate:.1f}%`", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with bi2:
    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
    st.markdown("##### ⚠️ Worst Pickup Location")
    if len(f_df) > 0:
        worst_pick = f_df.groupby('Pickup_Location')['Is_Completed'].mean().idxmin()
        worst_rate = f_df.groupby('Pickup_Location')['Is_Completed'].mean().min() * 100
        st.markdown(f"Friction Node: **{worst_pick}**<br>Completion Rate: `{worst_rate:.1f}%`", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with bi3:
    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
    st.markdown("##### ⏰ Time and Vol Peak Demands")
    if len(f_df) > 0:
        peak_hour_calc = f_df.groupby('Hour').size().idxmax()
        low_hour_calc = f_df.groupby('Hour').size().idxmin()
        st.markdown(f"Highest Load Hour: **{peak_hour_calc}:00**<br>Lowest Load Hour: **{low_hour_calc}:00**", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with bi4:
    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
    st.markdown("##### 🧠 Automated Action Strategy")
    if len(f_df) > 0:
        worst_loc = f_df.groupby('Pickup_Location')['Is_Completed'].mean().idxmin()
        st.markdown(f"**Recommendation Plan:** Dispatch dynamic vehicle surges to **{worst_loc}** immediately to counter system drops.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# 10. EXPORT CHANNEL SYSTEMS
# =====================================================================
st.markdown("---")
csv_data = f_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="💾 EXPORT ACTIVE FILTERED TELEMETRY ARCHIVE LOG (CSV)",
    data=csv_data,
    file_name="kathmandu_filtered_telemetry.csv",
    mime="text/csv",
    use_container_width=True
)

# =====================================================================
# 11. BRANDED PORTS & ENGAGEMENT FOOTER
# =====================================================================
st.markdown("""
    <div class="footer-container">
        <hr style="border-color:rgba(0,242,254,0.15); margin-bottom:15px;">
        Kathmandu Ride Engine Infrastructure Portal • Built with Streamlit & Plotly Canvas <br><br>
        👨‍💻 Developed by: 
        <a class="footer-link" href="https://github.com/Sakshyam-7" target="_blank">GitHub @Sakshyam-7</a> | 
        <a class="footer-link" href="https://www.linkedin.com/in/sakshyam-bhandari" target="_blank">LinkedIn @sakshyam-bhandari</a>
    </div>
""", unsafe_allow_html=True)