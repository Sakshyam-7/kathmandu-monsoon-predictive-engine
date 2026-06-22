import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =====================================================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# =====================================================================
st.set_page_config(
    page_title="Advanced Kathmandu Ride Analytics",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Advanced Kathmandu Monsoon Ride Analytics & Prediction Engine")

# =====================================================================
# 2. CACHED DATA LOADING ENGINE
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/kathmandu_monsoon_rides.csv")
    df['Is_Completed'] = df['Ride_Status'].apply(lambda x: 1 if x == 'Completed' else 0)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    return df

df = load_data()

# Calculate global baseline metrics for relative data scaling indicators
global_completion_baseline = df['Is_Completed'].mean() * 100

# =====================================================================
# 3. ROBUST MACHINE LEARNING TRAINING ENGINE (NO DUMMY TRAPS)
# =====================================================================
@st.cache_resource
def train_prediction_engine(data):
    ml_features = data[['Payment_Method', 'Weather', 'Is_Peak_Hour', 'Is_Bottleneck_Zone', 'Hour', 'Pickup_Location']]
    X = pd.get_dummies(ml_features)
    y = data['Is_Completed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test) * 100
    return model, X.columns, accuracy

model, model_columns, model_accuracy = train_prediction_engine(df)

# =====================================================================
# 4. SIDEBAR INPUT INTERFACES & REAL-TIME DISPATCH SIMULATION
# =====================================================================
st.sidebar.header("🔍 Dashboard Filters")
weather_filter = st.sidebar.selectbox(
    "Select Weather View:", 
    ['All', 'Clear', 'Cloudy', 'Heavy Monsoon Rain']
)

# Apply reactive filtering conditions across the UI metrics
filtered_df = df if weather_filter == 'All' else df[df['Weather'] == weather_filter]

st.sidebar.markdown("---")
st.sidebar.header("🤖 Live Ride Dispatch Simulator")
st.sidebar.write("Simulate an upcoming ride booking request to calculate its fulfillment probability:")

sim_weather = st.sidebar.selectbox("Simulated Weather Conditions:", sorted(df['Weather'].unique()))
sim_location = st.sidebar.selectbox("Simulated Pickup Location Corridor:", sorted(df['Pickup_Location'].unique()))
sim_payment = st.sidebar.selectbox("Simulated Customer Payment:", sorted(df['Payment_Method'].unique()))
sim_hour = st.sidebar.slider("Simulated Hour of Day (24h):", 0, 23, 12)
sim_peak = st.sidebar.checkbox("Is Peak Rush Hour?", value=False)
sim_bottleneck = st.sidebar.checkbox("Is Traffic Bottleneck Zone?", value=False)

# Empty container anchor near the top of the main dashboard space
prediction_container = st.container()

if st.sidebar.button("🔮 Run AI Prediction"):
    with st.spinner("🤖 AI Engine is calculating probabilities..."):
        input_data = pd.DataFrame(0, index=[0], columns=model_columns)
        
        input_data['Hour'] = sim_hour
        input_data['Is_Peak_Hour'] = 1 if sim_peak else 0
        input_data['Is_Bottleneck_Zone'] = 1 if sim_bottleneck else 0
        
        if f'Weather_{sim_weather}' in model_columns:
            input_data[f'Weather_{sim_weather}'] = 1
        if f'Payment_Method_{sim_payment}' in model_columns:
            input_data[f'Payment_Method_{sim_payment}'] = 1
        if f'Pickup_Location_{sim_location}' in model_columns:
            input_data[f'Pickup_Location_{sim_location}'] = 1
            
        probabilities = model.predict_proba(input_data)[0]
        completion_prob = probabilities[1] * 100
        
        # 1. Output result directly inside the sidebar menu
        st.sidebar.markdown("### 📊 Simulation Result")
        if completion_prob >= 70:
            st.sidebar.success(f"🟢 **High Success Rate: {completion_prob:.1f}%**")
            alert_color = "success"
        elif completion_prob >= 50:
            st.sidebar.warning(f"🟡 **Moderate Risk: {completion_prob:.1f}%**")
            alert_color = "warning"
        else:
            st.sidebar.error(f"🔴 **Critical Risk: {completion_prob:.1f}%**")
            alert_color = "error"
            
        # 2. Mirror output to the top of the main display grid so it catches your eye
        with prediction_container:
            st.markdown(f"### 🔮 Live Simulation Output")
            info_text = f"A ride request at **{sim_location}** ({sim_hour}:00, via {sim_payment}) facing **{sim_weather}** conditions holds a **{completion_prob:.1f}%** completion probability score."
            if alert_color == "success":
                st.success(info_text)
            elif alert_color == "warning":
                st.warning(info_text)
            else:
                st.error(info_text)
            
            # 💡 NEW FEATURE ADDITION: "What-If" Prescriptive Scenario Optimizer
            if completion_prob < 70 and sim_payment == 'Cash':
                optimized_data = input_data.copy()
                if 'Payment_Method_Cash' in optimized_data.columns:
                    optimized_data['Payment_Method_Cash'] = 0
                
                # Pick an alternative reliable payment method present in your database mapping
                alternative_payment = 'Esewa' if 'Payment_Method_Esewa' in optimized_data.columns else sorted(df['Payment_Method'].unique())[-1]
                if f'Payment_Method_{alternative_payment}' in optimized_data.columns:
                    optimized_data[f'Payment_Method_{alternative_payment}'] = 1
                
                # Extract optimization metric probabilities
                opt_prob = model.predict_proba(optimized_data)[0][1] * 100
                st.info(f"🛠️ **AI Optimization Recommendation:** Nudging this specific booking segment to settle fares via **{alternative_payment}** boosts fulfillment likelihood to **{opt_prob:.1f}%** (an optimization margin of +{opt_prob - completion_prob:.1f}%).")
            
            st.markdown("---")

# =====================================================================
# 5. CORE EXECUTIVE KPI DASHBOARD LAYER
# =====================================================================
col1, col2, col3 = st.columns(3)

current_completion_rate = filtered_df['Is_Completed'].mean() * 100
delta_completion = current_completion_rate - global_completion_baseline

col1.metric("Total Bookings Analyzed", f"{len(filtered_df):,}")
col2.metric(
    label="Current View Completion Rate", 
    value=f"{current_completion_rate:.2f}%",
    delta=f"{delta_completion:.2f}% vs City Baseline",
    delta_color="normal"
)
col3.metric("AI Engine Operational Accuracy", f"{model_accuracy:.1f}%", help="Evaluated via Random Forest Classifier Split Testing Matrices")

st.markdown("---")

# =====================================================================
# 6. ROW 1: COMPARATIVE BEHAVIOR VISUALIZATIONS
# =====================================================================
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Fulfillment Gridlock Matrix")
    gridlock = filtered_df.groupby(['Is_Peak_Hour', 'Is_Bottleneck_Zone'])['Is_Completed'].mean().unstack() * 100
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(gridlock, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax1, vmin=0, vmax=100)
    ax1.set_ylabel("Is Peak Hour (0=No, 1=Yes)")
    ax1.set_xlabel("Is Bottleneck Zone (0=No, 1=Yes)")
    st.pyplot(fig1)

with chart_col2:
    st.subheader(f"Weather vs. Completion Rate ({weather_filter} View)")
    
    # RESPONSIVE FIX: Group by 'filtered_df' instead of structural database constants
    weather_comp = filtered_df.groupby('Weather')['Is_Completed'].mean() * 100
    
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    
    # COLOR PROTECTION: Retain static color themes whether plotting isolated profiles or baseline records
    color_mapping = {
        'Clear': '#2ca02c',
        'Cloudy': '#bcbd22',
        'Heavy Monsoon Rain': '#d62728'
    }
    active_colors = [color_mapping.get(weather, '#1f77b4') for weather in weather_comp.index]
    
    bars = weather_comp.plot(kind='bar', color=active_colors, ax=ax2, zorder=2)
    
    # DATA LABEL ADDITION: Automatically map precise values directly over the target bars
    ax2.bar_label(ax2.containers[0], fmt='%.2f%%', padding=3, fontsize=10, weight='bold')
    
    ax2.set_ylabel("Completion Rate (%)")
    ax2.set_xlabel("Weather Condition")
    ax2.set_ylim(0, 115)  # Extended headroom prevents upper layout truncation issues
    ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    plt.xticks(rotation=15)
    st.pyplot(fig2)

st.markdown("---")

# =====================================================================
# 7. ROW 2: TEMPORAL DEMAND DYNAMICS TIMELINE
# =====================================================================
st.subheader("⏰ Hourly Booking Demand Profile")
all_hours = pd.Series(0, index=range(0, 24))
actual_demand = filtered_df.groupby('Hour').size()
hourly_demand = actual_demand.reindex(all_hours.index, fill_value=0)

fig3, ax3 = plt.subplots(figsize=(14, 3.5))
hourly_demand.plot(kind='line', marker='o', color='#1f77b4', linewidth=2.5, markersize=6, ax=ax3, zorder=3)
ax3.set_xlabel("Hour of the Day (24h)")
ax3.set_ylabel("Number of Bookings")
ax3.set_xticks(range(0, 24))
ax3.set_xlim(-0.5, 23.5)
ax3.grid(True, linestyle='--', alpha=0.6, zorder=1)
st.pyplot(fig3)

# =====================================================================
# 8. ROW 3: SPATIAL CORRIDOR ROUTE ROUTING MATRIX ANALYSIS
# =====================================================================
st.markdown("---")
st.subheader("📍 Localized Geographic Route Bottleneck Analysis")

# CSS adjustments to minimize spacing bugs between alert containers
st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div {
        padding-bottom: 4px !important;
    }
    </style>
""", unsafe_allow_html=True)

geo_col1, geo_col2 = st.columns([2, 1])

with geo_col1:
    st.write("#### Spatial Routing Fulfillment Map Matrix")
    route_matrix = filtered_df.groupby(['Pickup_Location', 'Dropoff_Location'])['Is_Completed'].mean().unstack() * 100
    
    fig4, ax4 = plt.subplots(figsize=(10, 6.5))
    sns.heatmap(
        route_matrix, 
        annot=True, 
        fmt=".1f", 
        cmap="YlOrRd_r", 
        ax=ax4,
        cbar_kws={'label': 'Route Success Rate (%)'}
    )
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    ax4.set_xlabel("Dropoff Location", labelpad=12)
    ax4.set_ylabel("Pickup Location", labelpad=12)
    plt.tight_layout()
    st.pyplot(fig4)

with geo_col2:
    st.write("#### 🚨 High Friction Fulfillment Zones")
    st.write("Identifying specific localization clusters demonstrating volatile cancellation metrics in current filter index window.")
    
    worst_pickups = filtered_df.groupby('Pickup_Location')['Is_Completed'].agg(
        Completion_Rate=lambda x: x.mean() * 100,
        Total_Bookings='count'
    ).sort_values(by='Completion_Rate').head(3)
    
    for idx, row in worst_pickups.iterrows():
        st.error(f"📍 **Locality Corridor: {idx}** \n\nFulfillment Performance: **{row['Completion_Rate']:.1f}%** | Sample Size: {row['Total_Bookings']:,} requests")
        
    st.info("💡 **Operational Recommendation:** When severe localized monsoon notifications cross reference these specific pickup bubbles, deploy dynamic micro-surges to protect fleet availability stability thresholds.")

# =====================================================================
# 9. AI EXPLAINABILITY & MODEL PERFORMANCE METRICS
# =====================================================================
st.markdown("---")
st.subheader("🧠 ML Engine Transparency & Behavioral Drivers")

explain_col1, explain_col2 = st.columns(2)

with explain_col1:
    st.write("#### Feature Importance Profile (What Drives the AI's Predictions)")
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=model_columns).sort_values(ascending=True).tail(10)
    
    fig5, ax5 = plt.subplots(figsize=(7, 4.5))
    feat_importances.plot(kind='barh', color='#8c564b', ax=ax5, zorder=2)
    ax5.set_xlabel("Relative Information Gain Weight")
    ax5.set_ylabel("Engine Encoded Feature Attribute")
    ax5.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
    plt.tight_layout()
    st.pyplot(fig5)

with explain_col2:
    st.write("#### Comprehensive Model Evaluation Profile")
    with st.expander("🔬 View Detailed Evaluation Summary Report (Precision/Recall Matrix)"):
        from sklearn.metrics import classification_report
        
        ml_features_eval = df[['Payment_Method', 'Weather', 'Is_Peak_Hour', 'Is_Bottleneck_Zone', 'Hour', 'Pickup_Location']]
        X_eval = pd.get_dummies(ml_features_eval)
        y_eval = df['Is_Completed']
        _, X_test_eval, _, y_test_eval = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)
        
        y_pred_eval = model.predict(X_test_eval)
        report_dict = classification_report(y_test_eval, y_pred_eval, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        report_df.rename(index={'0': 'Dropped/Canceled Journey', '1': 'Successful Conversion'}, inplace=True)
        
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        st.caption("**Note on Metrics:** High Precision minimizes false positive dispatch alerts. High Recall ensures actual gridlock dropouts are captured accurately by the operational risk algorithms before they occur.")
