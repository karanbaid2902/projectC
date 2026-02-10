"""
BitesUAE Analytics Dashboard - Comprehensive Version
A full-featured Streamlit dashboard for food delivery analytics
Answers ALL business questions from the project requirements
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="BitesUAE Analytics Dashboard",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #FF6B35;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 0.8rem 0;
        font-size: 1.1rem;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 0.8rem 0;
    }
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #dc3545;
        margin: 0.8rem 0;
    }
    .question-header {
        background: linear-gradient(135deg, #FF6B35 0%, #ff8c42 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0 0.5rem 0;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
        background-color: #e9ecef;
        border-radius: 8px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B35 0%, #ff8c42 100%);
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    .section-divider {
        border-top: 3px solid #FF6B35;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Get the directory where this script is located (works on Streamlit Cloud)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    """Load all data from CSV files"""
    data_dir = os.path.join(SCRIPT_DIR, "data")
    
    if not os.path.exists(data_dir):
        st.error("Data folder not found. Please run data_generator.py first.")
        st.stop()
    
    customers = pd.read_csv(f'{data_dir}/customers.csv', parse_dates=['registration_date'])
    restaurants = pd.read_csv(f'{data_dir}/restaurants.csv')
    riders = pd.read_csv(f'{data_dir}/riders.csv', parse_dates=['join_date'])
    orders = pd.read_csv(f'{data_dir}/orders.csv', parse_dates=['order_time'])
    delivery_events = pd.read_csv(f'{data_dir}/delivery_events.csv', 
                                   parse_dates=['restaurant_confirmed_time', 'food_ready_time', 
                                               'rider_picked_up_time', 'delivered_time', 
                                               'expected_delivery_time'])
    complaints = pd.read_csv(f'{data_dir}/complaints.csv', parse_dates=['complaint_time'])
    
    # Extract city from zone
    orders['city'] = orders['zone'].apply(lambda x: x.split('(')[1].replace(')', '').strip() if '(' in str(x) else 'Unknown')
    restaurants['city'] = restaurants['zone'].apply(lambda x: x.split('(')[1].replace(')', '').strip() if '(' in str(x) else 'Unknown')
    
    return customers, restaurants, riders, orders, delivery_events, complaints


def calculate_comprehensive_kpis(orders, delivery_events, complaints, customers):
    """Calculate all key performance indicators"""
    delivered_orders = orders[orders['status'] == 'Delivered']
    
    # GMV
    total_gmv = delivered_orders['order_value'].sum()
    
    # Order counts
    total_orders = len(orders)
    delivered_count = len(delivered_orders)
    cancelled_count = len(orders[orders['status'] == 'Cancelled'])
    
    # Repeat Rate & Frequency
    customer_order_counts = orders.groupby('customer_id').size()
    repeat_customers = (customer_order_counts > 1).sum()
    total_unique_customers = len(customer_order_counts)
    repeat_rate = (repeat_customers / total_unique_customers * 100) if total_unique_customers > 0 else 0
    avg_order_frequency = customer_order_counts.mean()
    
    # Average Order Values
    avg_order_value = delivered_orders['order_value'].mean() if len(delivered_orders) > 0 else 0
    
    # On-time Delivery
    valid_deliveries = delivery_events[delivery_events['is_on_time'].notna()]
    on_time_count = valid_deliveries['is_on_time'].sum()
    on_time_rate = (on_time_count / len(valid_deliveries) * 100) if len(valid_deliveries) > 0 else 0
    late_deliveries = len(valid_deliveries) - on_time_count
    
    # Delivery Times
    delivery_events_copy = delivery_events.copy()
    delivery_events_copy['delivery_duration'] = (
        (delivery_events_copy['delivered_time'] - delivery_events_copy['rider_picked_up_time']).dt.total_seconds() / 60
    )
    delivery_events_copy['prep_time'] = (
        (delivery_events_copy['food_ready_time'] - delivery_events_copy['restaurant_confirmed_time']).dt.total_seconds() / 60
    )
    delivery_events_copy['pickup_wait'] = (
        (delivery_events_copy['rider_picked_up_time'] - delivery_events_copy['food_ready_time']).dt.total_seconds() / 60
    )
    delivery_events_copy['total_time'] = (
        (delivery_events_copy['delivered_time'] - delivery_events_copy['restaurant_confirmed_time']).dt.total_seconds() / 60
    )
    
    avg_delivery_time = delivery_events_copy['delivery_duration'].mean()
    avg_prep_time = delivery_events_copy['prep_time'].mean()
    avg_pickup_wait = delivery_events_copy['pickup_wait'].mean()
    avg_total_time = delivery_events_copy['total_time'].mean()
    
    # Cancellation Rate
    cancellation_rate = (cancelled_count / total_orders * 100) if total_orders > 0 else 0
    
    # Discount Metrics
    total_discount = orders['discount_amount'].sum()
    discount_burn_rate = (total_discount / total_gmv * 100) if total_gmv > 0 else 0
    
    # Complaint Rate
    complaint_rate = (len(complaints) / delivered_count * 100) if delivered_count > 0 else 0
    
    return {
        'total_gmv': total_gmv,
        'total_orders': total_orders,
        'delivered_count': delivered_count,
        'cancelled_count': cancelled_count,
        'repeat_rate': repeat_rate,
        'avg_order_frequency': avg_order_frequency,
        'total_unique_customers': total_unique_customers,
        'repeat_customers': repeat_customers,
        'avg_order_value': avg_order_value,
        'on_time_rate': on_time_rate,
        'on_time_count': on_time_count,
        'late_deliveries': late_deliveries,
        'avg_delivery_time': avg_delivery_time if pd.notna(avg_delivery_time) else 0,
        'avg_prep_time': avg_prep_time if pd.notna(avg_prep_time) else 0,
        'avg_pickup_wait': avg_pickup_wait if pd.notna(avg_pickup_wait) else 0,
        'avg_total_time': avg_total_time if pd.notna(avg_total_time) else 0,
        'cancellation_rate': cancellation_rate,
        'total_discount': total_discount if pd.notna(total_discount) else 0,
        'discount_burn_rate': discount_burn_rate if pd.notna(discount_burn_rate) else 0,
        'complaint_rate': complaint_rate,
        'total_complaints': len(complaints)
    }


def generate_auto_insights(kpis, orders, delivery_events, restaurants, complaints):
    """Generate comprehensive auto-generated insights"""
    insights = []
    
    # Top zone by GMV
    zone_gmv = orders[orders['status'] == 'Delivered'].groupby('zone')['order_value'].sum()
    if not zone_gmv.empty:
        top_zone = zone_gmv.idxmax()
        top_zone_pct = (zone_gmv.max() / zone_gmv.sum() * 100)
        insights.append({
            'type': 'success',
            'text': f"📊 **GMV is AED {kpis['total_gmv']:,.0f}** with **{top_zone_pct:.1f}%** from **{top_zone}**. Repeat customer rate is **{kpis['repeat_rate']:.1f}%**."
        })
    
    # Best performing city
    city_gmv = orders[orders['status'] == 'Delivered'].groupby('city')['order_value'].sum()
    if not city_gmv.empty:
        top_city = city_gmv.idxmax()
        insights.append({
            'type': 'info',
            'text': f"🏙️ **{top_city}** leads in GMV with **AED {city_gmv.max():,.0f}** ({city_gmv.max()/city_gmv.sum()*100:.1f}% of total)."
        })
    
    # On-time performance insight
    if kpis['on_time_rate'] < 75:
        insights.append({
            'type': 'danger',
            'text': f"🚨 **Critical: On-time delivery rate is only {kpis['on_time_rate']:.1f}%** - significantly below target of 85%. Immediate action required."
        })
    elif kpis['on_time_rate'] < 85:
        insights.append({
            'type': 'warning',
            'text': f"⚠️ **On-time delivery rate is {kpis['on_time_rate']:.1f}%** - below 85% target. {int(kpis['late_deliveries'])} late deliveries detected."
        })
    else:
        insights.append({
            'type': 'success',
            'text': f"✅ **On-time delivery rate is {kpis['on_time_rate']:.1f}%** - exceeding 85% target!"
        })
    
    # Delay breakdown
    if kpis['avg_prep_time'] > 25:
        insights.append({
            'type': 'warning',
            'text': f"🍳 **Restaurant prep time averaging {kpis['avg_prep_time']:.1f} mins** - consider working with slow restaurants."
        })
    
    if kpis['avg_pickup_wait'] > 15:
        insights.append({
            'type': 'warning',
            'text': f"🏍️ **Rider pickup wait averaging {kpis['avg_pickup_wait']:.1f} mins** - indicates rider availability issues."
        })
    
    # Discount burn rate
    if kpis['discount_burn_rate'] > 15:
        insights.append({
            'type': 'danger',
            'text': f"💸 **Discount burn rate is {kpis['discount_burn_rate']:.1f}%** - high margin erosion! Total discounts: AED {kpis['total_discount']:,.0f}"
        })
    elif kpis['discount_burn_rate'] > 10:
        insights.append({
            'type': 'warning',
            'text': f"💰 **Discount burn rate is {kpis['discount_burn_rate']:.1f}%** - monitor promo effectiveness."
        })
    
    # Cancellation insight
    if kpis['cancellation_rate'] > 10:
        top_cancel_reasons = orders[orders['status'] == 'Cancelled']['cancellation_reason'].value_counts().head(3)
        reasons_text = ", ".join([f"{r} ({c})" for r, c in top_cancel_reasons.items()])
        insights.append({
            'type': 'danger',
            'text': f"🚫 **Cancellation rate is {kpis['cancellation_rate']:.1f}%** - Top reasons: {reasons_text}"
        })
    
    # Complaint correlation
    if kpis['total_complaints'] > 0:
        estimated_from_late = kpis['late_deliveries'] / 5  # 1 complaint per 5 late orders
        insights.append({
            'type': 'info',
            'text': f"📞 **{int(kpis['total_complaints'])} total complaints** - Estimated {int(estimated_from_late)} are due to late deliveries (assuming 1 complaint per 5 late orders)."
        })
    
    return insights


def apply_filters(orders, delivery_events, complaints, restaurants, riders, 
                  date_range, selected_city, selected_zones, selected_cuisines, 
                  selected_tiers, selected_segments, customers):
    """Apply all filters to the data"""
    filtered_orders = orders.copy()
    
    # Date filter
    if date_range and len(date_range) == 2:
        filtered_orders = filtered_orders[
            (filtered_orders['order_time'].dt.date >= date_range[0]) &
            (filtered_orders['order_time'].dt.date <= date_range[1])
        ]
    
    # City filter
    if selected_city != 'All':
        filtered_orders = filtered_orders[filtered_orders['city'] == selected_city]
    
    # Zone filter
    if selected_zones and 'All' not in selected_zones:
        filtered_orders = filtered_orders[filtered_orders['zone'].isin(selected_zones)]
    
    # Cuisine filter
    if selected_cuisines and 'All' not in selected_cuisines:
        rest_ids = restaurants[restaurants['cuisine_type'].isin(selected_cuisines)]['restaurant_id'].tolist()
        filtered_orders = filtered_orders[filtered_orders['restaurant_id'].isin(rest_ids)]
    
    # Tier filter
    if selected_tiers and 'All' not in selected_tiers:
        rest_ids = restaurants[restaurants['tier'].isin(selected_tiers)]['restaurant_id'].tolist()
        filtered_orders = filtered_orders[filtered_orders['restaurant_id'].isin(rest_ids)]
    
    # Customer segment filter
    if selected_segments and 'All' not in selected_segments:
        cust_ids = customers[customers['customer_segment'].isin(selected_segments)]['customer_id'].tolist()
        filtered_orders = filtered_orders[filtered_orders['customer_id'].isin(cust_ids)]
    
    # Filter related tables
    filtered_delivery = delivery_events[delivery_events['order_id'].isin(filtered_orders['order_id'])]
    filtered_complaints = complaints[complaints['order_id'].isin(filtered_orders['order_id'])]
    
    return filtered_orders, filtered_delivery, filtered_complaints


def render_sidebar_filters(orders, restaurants, customers):
    """Render comprehensive sidebar filters"""
    st.sidebar.markdown("## 🎛️ Filters")
    st.sidebar.markdown("---")
    
    # Date Range Filter
    st.sidebar.markdown("### 📅 Date Range")
    min_date = orders['order_time'].min().date()
    max_date = orders['order_time'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_filter"
    )
    
    st.sidebar.markdown("---")
    
    # City Filter
    st.sidebar.markdown("### 🏙️ City")
    cities = ['All'] + sorted(orders['city'].unique().tolist())
    selected_city = st.sidebar.selectbox("Select City", cities, key="city_filter")
    
    # Zone Filter (dependent on city)
    st.sidebar.markdown("### 📍 Zones")
    if selected_city == 'All':
        available_zones = orders['zone'].unique().tolist()
    else:
        available_zones = orders[orders['city'] == selected_city]['zone'].unique().tolist()
    zone_options = ['All'] + sorted(available_zones)
    selected_zones = st.sidebar.multiselect("Select Zones", zone_options, default=['All'], key="zone_filter")
    
    st.sidebar.markdown("---")
    
    # Cuisine Filter
    st.sidebar.markdown("### 🍽️ Cuisine Type")
    cuisines = ['All'] + sorted(restaurants['cuisine_type'].unique().tolist())
    selected_cuisines = st.sidebar.multiselect("Select Cuisines", cuisines, default=['All'], key="cuisine_filter")
    
    # Restaurant Tier Filter
    st.sidebar.markdown("### ⭐ Restaurant Tier")
    tiers = ['All'] + list(restaurants['tier'].unique())
    selected_tiers = st.sidebar.multiselect("Select Tiers", tiers, default=['All'], key="tier_filter")
    
    st.sidebar.markdown("---")
    
    # Customer Segment Filter
    st.sidebar.markdown("### 👥 Customer Segment")
    segments = ['All'] + sorted(customers['customer_segment'].unique().tolist())
    selected_segments = st.sidebar.multiselect("Select Segments", segments, default=['All'], key="segment_filter")
    
    st.sidebar.markdown("---")
    
    # Quick Time Presets
    st.sidebar.markdown("### ⏱️ Quick Presets")
    preset = st.sidebar.radio("Time Period", ["Custom", "Last 7 Days", "Last 30 Days", "Last 90 Days"], key="time_preset")
    
    return date_range, selected_city, selected_zones, selected_cuisines, selected_tiers, selected_segments, preset


def render_executive_view(orders, delivery_events, restaurants, customers, complaints, riders, kpis):
    """Render Executive Dashboard - Answers strategic questions"""
    
    st.markdown('<div class="main-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Strategic Business Performance Overview</div>', unsafe_allow_html=True)
    
    # ==================== KPI CARDS ====================
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total GMV", f"AED {kpis['total_gmv']:,.0f}")
    with col2:
        st.metric("Total Orders", f"{kpis['total_orders']:,}")
    with col3:
        st.metric("Repeat Rate", f"{kpis['repeat_rate']:.1f}%", 
                  delta=f"{kpis['repeat_rate']-50:.1f}% vs 50% target")
    with col4:
        st.metric("Avg Order Value", f"AED {kpis['avg_order_value']:.0f}")
    with col5:
        st.metric("On-Time Rate", f"{kpis['on_time_rate']:.1f}%",
                  delta=f"{kpis['on_time_rate']-85:.1f}% vs 85% target")
    with col6:
        st.metric("Discount Burn", f"{kpis['discount_burn_rate']:.1f}%")
    
    st.markdown("---")
    
    # ==================== AUTO-GENERATED INSIGHTS ====================
    st.markdown("### 💡 Auto-Generated Insights")
    insights = generate_auto_insights(kpis, orders, delivery_events, restaurants, complaints)
    
    for insight in insights:
        if insight['type'] == 'success':
            st.markdown(f'<div class="insight-box">{insight["text"]}</div>', unsafe_allow_html=True)
        elif insight['type'] == 'warning':
            st.markdown(f'<div class="warning-box">{insight["text"]}</div>', unsafe_allow_html=True)
        elif insight['type'] == 'danger':
            st.markdown(f'<div class="danger-box">{insight["text"]}</div>', unsafe_allow_html=True)
        else:
            st.info(insight['text'])
    
    st.markdown("---")
    
    # ==================== QUESTION 1: GMV TREND & TOP ZONES ====================
    st.markdown('<div class="question-header">📈 Q1: What is our GMV trend over the last 90 days, and which zones contribute the most?</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Daily GMV Trend
        daily_gmv = orders[orders['status'] == 'Delivered'].groupby(
            orders['order_time'].dt.date
        )['order_value'].sum().reset_index()
        daily_gmv.columns = ['Date', 'GMV']
        daily_gmv['7-Day MA'] = daily_gmv['GMV'].rolling(7).mean()
        
        fig_gmv = go.Figure()
        fig_gmv.add_trace(go.Scatter(x=daily_gmv['Date'], y=daily_gmv['GMV'], 
                                     mode='lines', name='Daily GMV', 
                                     line=dict(color='#FF6B35', width=1), opacity=0.5))
        fig_gmv.add_trace(go.Scatter(x=daily_gmv['Date'], y=daily_gmv['7-Day MA'], 
                                     mode='lines', name='7-Day Moving Avg', 
                                     line=dict(color='#FF6B35', width=3)))
        fig_gmv.update_layout(height=350, title='Daily GMV Trend with 7-Day Moving Average',
                              xaxis_title='Date', yaxis_title='GMV (AED)')
        st.plotly_chart(fig_gmv, use_container_width=True)
    
    with col2:
        # Top Zones by GMV
        zone_gmv = orders[orders['status'] == 'Delivered'].groupby('zone')['order_value'].sum().reset_index()
        zone_gmv.columns = ['Zone', 'GMV']
        zone_gmv = zone_gmv.sort_values('GMV', ascending=True).tail(10)
        zone_gmv['Percentage'] = (zone_gmv['GMV'] / zone_gmv['GMV'].sum() * 100).round(1)
        
        fig_zone = px.bar(zone_gmv, x='GMV', y='Zone', orientation='h',
                          text=zone_gmv['Percentage'].apply(lambda x: f'{x}%'),
                          color='GMV', color_continuous_scale='Oranges')
        fig_zone.update_layout(height=350, title='Top 10 Zones by GMV', showlegend=False)
        fig_zone.update_traces(textposition='outside')
        st.plotly_chart(fig_zone, use_container_width=True)
    
    # Zone GMV Table
    zone_summary = orders[orders['status'] == 'Delivered'].groupby('zone').agg({
        'order_id': 'count',
        'order_value': ['sum', 'mean']
    }).reset_index()
    zone_summary.columns = ['Zone', 'Orders', 'Total GMV', 'Avg Order Value']
    zone_summary['GMV %'] = (zone_summary['Total GMV'] / zone_summary['Total GMV'].sum() * 100).round(1)
    zone_summary = zone_summary.sort_values('Total GMV', ascending=False)
    st.dataframe(zone_summary.head(10).style.format({
        'Total GMV': 'AED {:,.0f}',
        'Avg Order Value': 'AED {:,.2f}',
        'GMV %': '{:.1f}%'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ==================== QUESTION 2: DISCOUNT BURN & PROMO EFFECTIVENESS ====================
    st.markdown('<div class="question-header">💳 Q2: What is our discount burn rate, and which promo codes are driving orders vs. just reducing margins?</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.metric("Total Discounts", f"AED {kpis['total_discount']:,.0f}")
        st.metric("Burn Rate", f"{kpis['discount_burn_rate']:.2f}%")
        avg_disc = orders['discount_amount'].mean()
        st.metric("Avg Discount/Order", f"AED {avg_disc:.2f}" if pd.notna(avg_disc) else "N/A")
    
    with col2:
        # Promo code performance
        promo_data = orders[orders['promo_code'].notna()].groupby('promo_code').agg({
            'order_id': 'count',
            'discount_amount': 'sum',
            'order_value': 'sum'
        }).reset_index()
        promo_data.columns = ['Promo Code', 'Orders', 'Discount Given', 'GMV Generated']
        promo_data['Discount Rate %'] = (promo_data['Discount Given'] / promo_data['GMV Generated'] * 100).round(2)
        promo_data['Order Value/Order'] = (promo_data['GMV Generated'] / promo_data['Orders']).round(2)
        promo_data = promo_data.sort_values('Orders', ascending=False)
        
        # Identify efficient vs wasteful promos
        avg_discount_rate = promo_data['Discount Rate %'].mean()
        promo_data['Efficiency'] = promo_data['Discount Rate %'].apply(
            lambda x: '✅ Efficient' if x < avg_discount_rate else '⚠️ High Cost'
        )
        
        st.dataframe(promo_data.style.format({
            'Discount Given': 'AED {:,.0f}',
            'GMV Generated': 'AED {:,.0f}',
            'Discount Rate %': '{:.1f}%',
            'Order Value/Order': 'AED {:,.0f}'
        }), use_container_width=True, hide_index=True)
    
    with col3:
        # Promo effectiveness scatter
        fig_promo = px.scatter(promo_data, x='Orders', y='Discount Rate %', 
                               size='GMV Generated', color='Promo Code',
                               hover_data=['Discount Given', 'GMV Generated'],
                               title='Promo Effectiveness: Orders vs Discount Rate')
        fig_promo.add_hline(y=avg_discount_rate, line_dash="dash", 
                           annotation_text=f"Avg: {avg_discount_rate:.1f}%")
        fig_promo.update_layout(height=350)
        st.plotly_chart(fig_promo, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== QUESTION 3: REPEAT CUSTOMERS ====================
    st.markdown('<div class="question-header">🔄 Q3: What percentage of customers are repeat buyers, and what is their average order frequency?</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Customer order frequency analysis
    customer_orders = orders.groupby('customer_id').agg({
        'order_id': 'count',
        'order_value': 'sum'
    }).reset_index()
    customer_orders.columns = ['customer_id', 'order_count', 'lifetime_value']
    customer_orders = customer_orders.merge(customers[['customer_id', 'customer_segment']], on='customer_id')
    
    with col1:
        st.metric("Total Unique Customers", f"{kpis['total_unique_customers']:,}")
        st.metric("Repeat Customers", f"{kpis['repeat_customers']:,}")
        st.metric("Repeat Rate", f"{kpis['repeat_rate']:.1f}%")
        st.metric("Avg Order Frequency", f"{kpis['avg_order_frequency']:.2f} orders/customer")
    
    with col2:
        # Order frequency distribution
        freq_dist = customer_orders['order_count'].value_counts().sort_index().head(15).reset_index()
        freq_dist.columns = ['Orders', 'Customers']
        
        fig_freq = px.bar(freq_dist, x='Orders', y='Customers', 
                          title='Customer Order Frequency Distribution',
                          color='Customers', color_continuous_scale='Blues')
        fig_freq.update_layout(height=350)
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col3:
        # Segment breakdown
        segment_stats = customer_orders.groupby('customer_segment').agg({
            'order_count': 'mean',
            'lifetime_value': 'mean',
            'customer_id': 'count'
        }).reset_index()
        segment_stats.columns = ['Segment', 'Avg Orders', 'Avg LTV', 'Customers']
        
        fig_segment = px.bar(segment_stats, x='Segment', y='Avg Orders',
                             color='Avg LTV', title='Avg Orders by Customer Segment',
                             text='Customers', color_continuous_scale='Oranges')
        fig_segment.update_layout(height=350)
        st.plotly_chart(fig_segment, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== QUESTION 4: RESTAURANT TIER ECONOMICS ====================
    st.markdown('<div class="question-header">🏪 Q4: Which restaurant tier (QSR, Casual, Premium, Fine Dining) has the best unit economics?</div>', unsafe_allow_html=True)
    
    # Merge orders with restaurant data
    tier_orders = orders.merge(restaurants[['restaurant_id', 'tier', 'cuisine_type']], on='restaurant_id')
    tier_delivered = tier_orders[tier_orders['status'] == 'Delivered']
    
    tier_summary = tier_delivered.groupby('tier').agg({
        'order_id': 'count',
        'order_value': ['sum', 'mean'],
        'discount_amount': 'sum'
    }).reset_index()
    tier_summary.columns = ['Tier', 'Orders', 'Total GMV', 'Avg Order Value', 'Total Discounts']
    tier_summary['Revenue per Order'] = tier_summary['Avg Order Value']
    tier_summary['Discount per Order'] = tier_summary['Total Discounts'] / tier_summary['Orders']
    tier_summary['Net Revenue per Order'] = tier_summary['Revenue per Order'] - tier_summary['Discount per Order']
    tier_summary['Margin %'] = ((tier_summary['Net Revenue per Order'] / tier_summary['Revenue per Order']) * 100).round(1)
    tier_summary = tier_summary.sort_values('Net Revenue per Order', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(tier_summary[['Tier', 'Orders', 'Avg Order Value', 'Discount per Order', 
                                   'Net Revenue per Order', 'Margin %']].style.format({
            'Avg Order Value': 'AED {:,.2f}',
            'Discount per Order': 'AED {:,.2f}',
            'Net Revenue per Order': 'AED {:,.2f}',
            'Margin %': '{:.1f}%'
        }), use_container_width=True, hide_index=True)
        
        best_tier = tier_summary.iloc[0]['Tier']
        st.success(f"🏆 **{best_tier}** has the best unit economics with AED {tier_summary.iloc[0]['Net Revenue per Order']:.2f} net revenue per order")
    
    with col2:
        fig_tier = px.bar(tier_summary, x='Tier', y=['Revenue per Order', 'Discount per Order'],
                          title='Revenue vs Discount by Restaurant Tier',
                          barmode='group', color_discrete_sequence=['#28a745', '#dc3545'])
        fig_tier.update_layout(height=350)
        st.plotly_chart(fig_tier, use_container_width=True)


def render_manager_view(orders, delivery_events, restaurants, customers, complaints, riders, kpis):
    """Render Operations Manager Dashboard - Answers operational questions"""
    
    st.markdown('<div class="main-header">⚙️ Operations Manager Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Operational Performance & Bottleneck Analysis</div>', unsafe_allow_html=True)
    
    # ==================== OPERATIONAL METRICS ====================
    st.markdown("### 📊 Operational Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("On-Time Rate", f"{kpis['on_time_rate']:.1f}%")
    with col2:
        st.metric("Late Deliveries", f"{int(kpis['late_deliveries']):,}")
    with col3:
        st.metric("Avg Prep Time", f"{kpis['avg_prep_time']:.1f} min")
    with col4:
        st.metric("Avg Pickup Wait", f"{kpis['avg_pickup_wait']:.1f} min")
    with col5:
        st.metric("Avg Delivery", f"{kpis['avg_delivery_time']:.1f} min")
    with col6:
        st.metric("Cancellation Rate", f"{kpis['cancellation_rate']:.1f}%")
    
    st.markdown("---")
    
    # ==================== QUESTION 5: ON-TIME DELIVERY BY ZONE ====================
    st.markdown('<div class="question-header">⏱️ Q5: What is our on-time delivery rate by zone, and which zones have the worst performance?</div>', unsafe_allow_html=True)
    
    # Calculate on-time rate by zone
    zone_delivery = delivery_events.merge(orders[['order_id', 'zone']], on='order_id')
    zone_performance = zone_delivery.groupby('zone').agg({
        'is_on_time': lambda x: (x == True).sum() / x.notna().sum() * 100 if x.notna().sum() > 0 else 0,
        'order_id': 'count'
    }).reset_index()
    zone_performance.columns = ['Zone', 'On-Time Rate %', 'Total Orders']
    zone_performance = zone_performance.sort_values('On-Time Rate %')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Worst performing zones
        worst_zones = zone_performance.head(10)
        fig_worst = px.bar(worst_zones, x='On-Time Rate %', y='Zone', orientation='h',
                           title='⚠️ Bottom 10 Zones by On-Time Rate',
                           color='On-Time Rate %', color_continuous_scale='RdYlGn',
                           text=worst_zones['On-Time Rate %'].apply(lambda x: f'{x:.1f}%'))
        fig_worst.add_vline(x=85, line_dash="dash", annotation_text="85% Target")
        fig_worst.update_layout(height=400)
        fig_worst.update_traces(textposition='outside')
        st.plotly_chart(fig_worst, use_container_width=True)
    
    with col2:
        # Best performing zones
        best_zones = zone_performance.tail(10).sort_values('On-Time Rate %', ascending=True)
        fig_best = px.bar(best_zones, x='On-Time Rate %', y='Zone', orientation='h',
                          title='✅ Top 10 Zones by On-Time Rate',
                          color='On-Time Rate %', color_continuous_scale='RdYlGn',
                          text=best_zones['On-Time Rate %'].apply(lambda x: f'{x:.1f}%'))
        fig_best.add_vline(x=85, line_dash="dash", annotation_text="85% Target")
        fig_best.update_layout(height=400)
        fig_best.update_traces(textposition='outside')
        st.plotly_chart(fig_best, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== TOP 10 PROBLEM AREAS TABLE ====================
    st.markdown('<div class="question-header">🚨 Top 10 Problem Areas</div>', unsafe_allow_html=True)
    st.caption("Columns: Zone, Late Deliveries, Avg Delay (mins), Top Complaint")
    
    problem_areas = []
    for zone in orders['zone'].unique():
        zone_orders = orders[orders['zone'] == zone]
        zone_del = delivery_events[delivery_events['order_id'].isin(zone_orders['order_id'])]
        zone_comp = complaints[complaints['zone'] == zone]
        
        # Late deliveries count
        late_count = (zone_del['is_on_time'] == False).sum()
        
        # Average delay for late orders
        zone_del_copy = zone_del.copy()
        zone_del_copy['delay'] = (zone_del_copy['delivered_time'] - zone_del_copy['expected_delivery_time']).dt.total_seconds() / 60
        late_orders = zone_del_copy[zone_del_copy['delay'] > 0]
        avg_delay = late_orders['delay'].mean() if len(late_orders) > 0 else 0
        
        # Top complaint
        if len(zone_comp) > 0:
            top_complaint = zone_comp['complaint_category'].mode().values[0]
        else:
            top_complaint = 'No complaints'
        
        problem_areas.append({
            'Zone': zone,
            'Late Deliveries': int(late_count),
            'Avg Delay (mins)': round(avg_delay, 1) if pd.notna(avg_delay) else 0,
            'Top Complaint': top_complaint,
            'Total Orders': len(zone_orders),
            'Late %': round(late_count / len(zone_del) * 100, 1) if len(zone_del) > 0 else 0
        })
    
    problem_df = pd.DataFrame(problem_areas)
    problem_df = problem_df.sort_values('Late Deliveries', ascending=False).head(10)
    
    st.dataframe(problem_df[['Zone', 'Late Deliveries', 'Avg Delay (mins)', 'Top Complaint', 'Total Orders', 'Late %']].style.format({
        'Late %': '{:.1f}%',
        'Avg Delay (mins)': '{:.1f}'
    }).background_gradient(subset=['Late Deliveries'], cmap='Reds'), 
    use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ==================== QUESTION 6: CANCELLATION ANALYSIS ====================
    st.markdown('<div class="question-header">🚫 Q6: What are the top 3 cancellation reasons, and which zones have the highest cancellation rates?</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    cancelled = orders[orders['status'] == 'Cancelled']
    
    with col1:
        # Top 3 cancellation reasons
        cancel_reasons = cancelled['cancellation_reason'].value_counts().reset_index()
        cancel_reasons.columns = ['Reason', 'Count']
        cancel_reasons['Percentage'] = (cancel_reasons['Count'] / cancel_reasons['Count'].sum() * 100).round(1)
        
        st.markdown("#### Top 3 Cancellation Reasons")
        for i, row in cancel_reasons.head(3).iterrows():
            st.markdown(f"**{i+1}. {row['Reason']}** - {row['Count']} orders ({row['Percentage']}%)")
        
        fig_cancel = px.pie(cancel_reasons, values='Count', names='Reason',
                            title='All Cancellation Reasons', hole=0.4,
                            color_discrete_sequence=px.colors.sequential.Reds)
        fig_cancel.update_layout(height=300)
        st.plotly_chart(fig_cancel, use_container_width=True)
    
    with col2:
        # Cancellation rate by zone
        zone_cancel = orders.groupby('zone').apply(
            lambda x: (x['status'] == 'Cancelled').sum() / len(x) * 100
        ).reset_index(name='Cancellation Rate %')
        zone_cancel = zone_cancel.sort_values('Cancellation Rate %', ascending=False).head(10)
        
        fig_zone_cancel = px.bar(zone_cancel, x='Cancellation Rate %', y='zone', orientation='h',
                                 title='Top 10 Zones by Cancellation Rate',
                                 color='Cancellation Rate %', color_continuous_scale='Reds',
                                 text=zone_cancel['Cancellation Rate %'].apply(lambda x: f'{x:.1f}%'))
        fig_zone_cancel.update_layout(height=400)
        fig_zone_cancel.update_traces(textposition='outside')
        st.plotly_chart(fig_zone_cancel, use_container_width=True)
    
    with col3:
        # Cancellation trend
        cancel_daily = cancelled.groupby(cancelled['order_time'].dt.date).size().reset_index(name='Cancellations')
        cancel_daily.columns = ['Date', 'Cancellations']
        
        fig_trend = px.line(cancel_daily, x='Date', y='Cancellations',
                            title='Daily Cancellation Trend',
                            color_discrete_sequence=['#dc3545'])
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== QUESTION 7: RIDER PERFORMANCE ====================
    st.markdown('<div class="question-header">🏍️ Q7: Which riders have the longest average delivery times, and should we provide retraining?</div>', unsafe_allow_html=True)
    
    # Rider performance analysis
    rider_delivery = delivery_events.merge(orders[['order_id', 'rider_id']], on='order_id')
    rider_delivery['delivery_duration'] = (
        (rider_delivery['delivered_time'] - rider_delivery['rider_picked_up_time']).dt.total_seconds() / 60
    )
    
    rider_perf = rider_delivery.groupby('rider_id').agg({
        'delivery_duration': 'mean',
        'is_on_time': lambda x: (x == True).sum() / x.notna().sum() * 100 if x.notna().sum() > 0 else 0,
        'order_id': 'count'
    }).reset_index()
    rider_perf.columns = ['rider_id', 'Avg Delivery Time', 'On-Time Rate %', 'Total Deliveries']
    rider_perf = rider_perf.merge(riders[['rider_id', 'rider_name', 'vehicle_type', 'rating', 'primary_zone']], on='rider_id')
    
    # Filter riders with minimum deliveries for fair comparison
    min_deliveries = 10
    qualified_riders = rider_perf[rider_perf['Total Deliveries'] >= min_deliveries]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Slowest riders - need retraining
        slowest = qualified_riders.nlargest(15, 'Avg Delivery Time')
        
        st.markdown("#### 🐢 Riders Needing Retraining (Slowest)")
        st.dataframe(slowest[['rider_name', 'vehicle_type', 'primary_zone', 'Total Deliveries', 
                              'Avg Delivery Time', 'On-Time Rate %', 'rating']].style.format({
            'Avg Delivery Time': '{:.1f} min',
            'On-Time Rate %': '{:.1f}%',
            'rating': '{:.1f}'
        }).background_gradient(subset=['Avg Delivery Time'], cmap='Reds'),
        use_container_width=True, hide_index=True)
        
        st.warning(f"⚠️ **{len(slowest)} riders** identified for potential retraining based on delivery times above average.")
    
    with col2:
        # Best performing riders
        fastest = qualified_riders.nsmallest(15, 'Avg Delivery Time')
        
        st.markdown("#### 🚀 Top Performing Riders")
        st.dataframe(fastest[['rider_name', 'vehicle_type', 'primary_zone', 'Total Deliveries', 
                              'Avg Delivery Time', 'On-Time Rate %', 'rating']].style.format({
            'Avg Delivery Time': '{:.1f} min',
            'On-Time Rate %': '{:.1f}%',
            'rating': '{:.1f}'
        }).background_gradient(subset=['On-Time Rate %'], cmap='Greens'),
        use_container_width=True, hide_index=True)
    
    # Rider performance distribution
    st.markdown("#### Rider Performance Distribution")
    fig_dist = px.histogram(qualified_riders, x='Avg Delivery Time', nbins=30,
                            title='Distribution of Average Delivery Times Across Riders',
                            color_discrete_sequence=['#FF6B35'])
    avg_time = qualified_riders['Avg Delivery Time'].mean()
    fig_dist.add_vline(x=avg_time, line_dash="dash", annotation_text=f"Avg: {avg_time:.1f} min")
    fig_dist.update_layout(height=300)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== PEAK HOURS ANALYSIS ====================
    st.markdown('<div class="question-header">🕐 Peak Hours Analysis</div>', unsafe_allow_html=True)
    
    orders_copy = orders.copy()
    orders_copy['hour'] = orders_copy['order_time'].dt.hour
    orders_copy['day_of_week'] = orders_copy['order_time'].dt.day_name()
    
    hourly_orders = orders_copy.groupby('hour').size().reset_index(name='Orders')
    
    # Merge with on-time rates
    delivery_copy = delivery_events.copy()
    delivery_copy = delivery_copy.merge(orders[['order_id', 'order_time']], on='order_id')
    delivery_copy['hour'] = delivery_copy['order_time'].dt.hour
    hourly_ontime = delivery_copy.groupby('hour')['is_on_time'].apply(
        lambda x: (x == True).sum() / x.notna().sum() * 100 if x.notna().sum() > 0 else 0
    ).reset_index(name='On-Time Rate %')
    
    hourly_combined = hourly_orders.merge(hourly_ontime, on='hour')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hourly.add_trace(
            go.Bar(x=hourly_combined['hour'], y=hourly_combined['Orders'], name='Orders',
                   marker_color='#FF6B35', opacity=0.7),
            secondary_y=False
        )
        fig_hourly.add_trace(
            go.Scatter(x=hourly_combined['hour'], y=hourly_combined['On-Time Rate %'], 
                       name='On-Time Rate %', mode='lines+markers', marker_color='#28a745',
                       line=dict(width=3)),
            secondary_y=True
        )
        fig_hourly.add_hline(y=85, line_dash="dash", line_color="red", 
                             annotation_text="85% Target", secondary_y=True)
        fig_hourly.update_layout(height=400, title='Orders & On-Time Rate by Hour')
        fig_hourly.update_xaxes(title_text='Hour of Day', tickmode='linear')
        fig_hourly.update_yaxes(title_text='Number of Orders', secondary_y=False)
        fig_hourly.update_yaxes(title_text='On-Time Rate %', secondary_y=True)
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Identify problem hours
        problem_hours = hourly_combined[hourly_combined['On-Time Rate %'] < 85]['hour'].tolist()
        if problem_hours:
            st.warning(f"⚠️ Peak hours with low on-time rates: {', '.join([f'{h}:00' for h in problem_hours])}")
    
    with col2:
        # Heatmap by day and hour
        heatmap_data = orders_copy.groupby(['day_of_week', 'hour']).size().reset_index(name='Orders')
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='Orders').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])
        
        fig_heat = px.imshow(heatmap_pivot, 
                             labels=dict(x='Hour', y='Day', color='Orders'),
                             title='Order Volume Heatmap by Day and Hour',
                             color_continuous_scale='YlOrRd')
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== DELAY BREAKDOWN ====================
    st.markdown('<div class="question-header">⏳ Delay Breakdown: Restaurant Prep vs Rider Pickup vs Delivery</div>', unsafe_allow_html=True)
    
    del_analysis = delivery_events.copy()
    del_analysis['prep_time'] = (del_analysis['food_ready_time'] - del_analysis['restaurant_confirmed_time']).dt.total_seconds() / 60
    del_analysis['pickup_wait'] = (del_analysis['rider_picked_up_time'] - del_analysis['food_ready_time']).dt.total_seconds() / 60
    del_analysis['delivery_time'] = (del_analysis['delivered_time'] - del_analysis['rider_picked_up_time']).dt.total_seconds() / 60
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_prep = px.histogram(del_analysis, x='prep_time', nbins=40,
                                title='Restaurant Prep Time Distribution',
                                color_discrete_sequence=['#FF6B35'])
        fig_prep.add_vline(x=del_analysis['prep_time'].mean(), line_dash="dash",
                           annotation_text=f"Avg: {del_analysis['prep_time'].mean():.1f} min")
        fig_prep.update_layout(height=300, xaxis_title='Prep Time (mins)')
        st.plotly_chart(fig_prep, use_container_width=True)
    
    with col2:
        fig_pickup = px.histogram(del_analysis, x='pickup_wait', nbins=40,
                                  title='Rider Pickup Wait Distribution',
                                  color_discrete_sequence=['#2E86AB'])
        fig_pickup.add_vline(x=del_analysis['pickup_wait'].mean(), line_dash="dash",
                             annotation_text=f"Avg: {del_analysis['pickup_wait'].mean():.1f} min")
        fig_pickup.update_layout(height=300, xaxis_title='Pickup Wait (mins)')
        st.plotly_chart(fig_pickup, use_container_width=True)
    
    with col3:
        fig_del = px.histogram(del_analysis, x='delivery_time', nbins=40,
                               title='Rider Delivery Time Distribution',
                               color_discrete_sequence=['#28a745'])
        fig_del.add_vline(x=del_analysis['delivery_time'].mean(), line_dash="dash",
                          annotation_text=f"Avg: {del_analysis['delivery_time'].mean():.1f} min")
        fig_del.update_layout(height=300, xaxis_title='Delivery Time (mins)')
        st.plotly_chart(fig_del, use_container_width=True)
    
    # Average time breakdown pie chart
    avg_breakdown = pd.DataFrame({
        'Stage': ['Prep Time', 'Pickup Wait', 'Delivery'],
        'Minutes': [del_analysis['prep_time'].mean(), 
                    del_analysis['pickup_wait'].mean(), 
                    del_analysis['delivery_time'].mean()]
    })
    
    fig_pie = px.pie(avg_breakdown, values='Minutes', names='Stage',
                     title='Average Time Breakdown by Stage',
                     color_discrete_sequence=['#FF6B35', '#2E86AB', '#28a745'])
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)


def render_drilldown_view(orders, delivery_events, restaurants, customers, complaints, riders, kpis):
    """Render Zone Drill-Down Analysis"""
    
    st.markdown('<div class="main-header">🔍 Zone Drill-Down Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep dive into zone-specific performance</div>', unsafe_allow_html=True)
    
    # Zone Selection
    selected_zone = st.selectbox("🎯 Select Zone for Deep Analysis", 
                                 sorted(orders['zone'].unique().tolist()),
                                 key="drill_zone")
    
    # Filter data for selected zone
    zone_orders = orders[orders['zone'] == selected_zone]
    zone_delivery = delivery_events[delivery_events['order_id'].isin(zone_orders['order_id'])]
    zone_complaints = complaints[complaints['zone'] == selected_zone]
    zone_restaurants = restaurants[restaurants['zone'] == selected_zone]
    
    st.markdown("---")
    
    # Zone Summary
    st.markdown(f"### 📍 {selected_zone} - Performance Summary")
    
    zone_gmv = zone_orders[zone_orders['status'] == 'Delivered']['order_value'].sum()
    zone_on_time = (zone_delivery['is_on_time'] == True).sum()
    zone_total_del = zone_delivery['is_on_time'].notna().sum()
    zone_on_time_rate = (zone_on_time / zone_total_del * 100) if zone_total_del > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Zone GMV", f"AED {zone_gmv:,.0f}")
    with col2:
        st.metric("Total Orders", f"{len(zone_orders):,}")
    with col3:
        st.metric("On-Time Rate", f"{zone_on_time_rate:.1f}%")
    with col4:
        st.metric("Restaurants", f"{len(zone_restaurants)}")
    with col5:
        st.metric("Complaints", f"{len(zone_complaints)}")
    
    st.markdown("---")
    
    # Restaurant Performance in Zone
    st.markdown("### 🍽️ Restaurant Performance (Prep Times)")
    
    rest_orders = zone_orders.merge(restaurants[['restaurant_id', 'restaurant_name', 'tier', 'cuisine_type']], on='restaurant_id')
    rest_delivery = zone_delivery.merge(zone_orders[['order_id', 'restaurant_id']], on='order_id')
    rest_delivery['prep_time'] = (rest_delivery['food_ready_time'] - rest_delivery['restaurant_confirmed_time']).dt.total_seconds() / 60
    
    rest_perf = rest_delivery.groupby('restaurant_id').agg({
        'prep_time': 'mean',
        'is_on_time': lambda x: (x == True).sum() / x.notna().sum() * 100 if x.notna().sum() > 0 else 0,
        'order_id': 'count'
    }).reset_index()
    rest_perf.columns = ['restaurant_id', 'Avg Prep Time', 'On-Time Rate %', 'Orders']
    rest_perf = rest_perf.merge(restaurants[['restaurant_id', 'restaurant_name', 'tier', 'cuisine_type', 'rating']], on='restaurant_id')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🐢 Slowest Restaurants (High Prep Time)")
        slow_rest = rest_perf.nlargest(10, 'Avg Prep Time')
        st.dataframe(slow_rest[['restaurant_name', 'tier', 'Orders', 'Avg Prep Time', 'On-Time Rate %', 'rating']].style.format({
            'Avg Prep Time': '{:.1f} min',
            'On-Time Rate %': '{:.1f}%',
            'rating': '{:.1f}'
        }).background_gradient(subset=['Avg Prep Time'], cmap='Reds'),
        use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 🚀 Fastest Restaurants")
        fast_rest = rest_perf.nsmallest(10, 'Avg Prep Time')
        st.dataframe(fast_rest[['restaurant_name', 'tier', 'Orders', 'Avg Prep Time', 'On-Time Rate %', 'rating']].style.format({
            'Avg Prep Time': '{:.1f} min',
            'On-Time Rate %': '{:.1f}%',
            'rating': '{:.1f}'
        }).background_gradient(subset=['On-Time Rate %'], cmap='Greens'),
        use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Rider Performance in Zone
    st.markdown("### 🏍️ Rider Performance (Delivery Times)")
    
    zone_orders_riders = zone_orders.merge(riders[['rider_id', 'rider_name', 'vehicle_type', 'rating']], on='rider_id')
    rider_delivery = zone_delivery.merge(zone_orders[['order_id', 'rider_id']], on='order_id')
    rider_delivery['delivery_duration'] = (rider_delivery['delivered_time'] - rider_delivery['rider_picked_up_time']).dt.total_seconds() / 60
    
    rider_perf = rider_delivery.groupby('rider_id').agg({
        'delivery_duration': 'mean',
        'is_on_time': lambda x: (x == True).sum() / x.notna().sum() * 100 if x.notna().sum() > 0 else 0,
        'order_id': 'count'
    }).reset_index()
    rider_perf.columns = ['rider_id', 'Avg Delivery Time', 'On-Time Rate %', 'Deliveries']
    rider_perf = rider_perf.merge(riders[['rider_id', 'rider_name', 'vehicle_type', 'rating']], on='rider_id')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🐢 Slowest Riders in Zone")
        slow_riders = rider_perf[rider_perf['Deliveries'] >= 3].nlargest(10, 'Avg Delivery Time')
        st.dataframe(slow_riders[['rider_name', 'vehicle_type', 'Deliveries', 'Avg Delivery Time', 'On-Time Rate %', 'rating']].style.format({
            'Avg Delivery Time': '{:.1f} min',
            'On-Time Rate %': '{:.1f}%',
            'rating': '{:.1f}'
        }).background_gradient(subset=['Avg Delivery Time'], cmap='Reds'),
        use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 🚀 Fastest Riders in Zone")
        fast_riders = rider_perf[rider_perf['Deliveries'] >= 3].nsmallest(10, 'Avg Delivery Time')
        st.dataframe(fast_riders[['rider_name', 'vehicle_type', 'Deliveries', 'Avg Delivery Time', 'On-Time Rate %', 'rating']].style.format({
            'Avg Delivery Time': '{:.1f} min',
            'On-Time Rate %': '{:.1f}%',
            'rating': '{:.1f}'
        }).background_gradient(subset=['On-Time Rate %'], cmap='Greens'),
        use_container_width=True, hide_index=True)


def render_whatif_view(orders, delivery_events, restaurants, customers, complaints, riders, kpis):
    """Render What-If Analysis with projected complaint reduction"""
    
    st.markdown('<div class="main-header">🔮 What-If Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Scenario Planning & Impact Projections</div>', unsafe_allow_html=True)
    
    # Current state summary
    st.markdown("### 📊 Current State")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Late Deliveries", f"{int(kpis['late_deliveries']):,}")
    with col2:
        st.metric("Total Complaints", f"{int(kpis['total_complaints']):,}")
    with col3:
        st.metric("On-Time Rate", f"{kpis['on_time_rate']:.1f}%")
    with col4:
        st.metric("Avg Total Delivery Time", f"{kpis['avg_total_time']:.1f} mins")
    
    st.info("💡 **Assumption**: 1 customer complaint is generated for every 5 late deliveries")
    
    st.markdown("---")
    
    # Scenario 1: Reduce Prep Time
    st.markdown("### 🍳 Scenario 1: Reduce Restaurant Prep Time")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        prep_reduction = st.slider("Reduce avg prep time by (minutes)", 0, 20, 5, key="prep_slider")
        
        # Calculate impact
        current_avg_prep = kpis['avg_prep_time']
        new_avg_prep = max(5, current_avg_prep - prep_reduction)
        
        # Estimate late delivery reduction (assume linear relationship)
        late_reduction_pct = min(50, prep_reduction * 3)  # 3% reduction per minute
        new_late_deliveries = kpis['late_deliveries'] * (1 - late_reduction_pct / 100)
        late_reduction_count = kpis['late_deliveries'] - new_late_deliveries
        
        # Complaint reduction (1 per 5 late orders)
        complaint_reduction = late_reduction_count / 5
        new_complaints = max(0, kpis['total_complaints'] - complaint_reduction)
        
        new_on_time_rate = min(100, kpis['on_time_rate'] + (late_reduction_pct * kpis['on_time_rate'] / 100))
    
    with col2:
        st.markdown("#### 📈 Projected Impact")
        
        impact_data = pd.DataFrame({
            'Metric': ['Avg Prep Time', 'Late Deliveries', 'On-Time Rate', 'Complaints'],
            'Current': [f"{current_avg_prep:.1f} min", f"{int(kpis['late_deliveries']):,}", 
                       f"{kpis['on_time_rate']:.1f}%", f"{int(kpis['total_complaints']):,}"],
            'Projected': [f"{new_avg_prep:.1f} min", f"{int(new_late_deliveries):,}", 
                         f"{new_on_time_rate:.1f}%", f"{int(new_complaints):,}"],
            'Change': [f"-{prep_reduction} min", f"-{int(late_reduction_count):,}", 
                      f"+{new_on_time_rate - kpis['on_time_rate']:.1f}%", f"-{int(complaint_reduction):,}"]
        })
        st.dataframe(impact_data, use_container_width=True, hide_index=True)
        
        st.success(f"✅ Reducing prep time by {prep_reduction} mins could reduce complaints by **{int(complaint_reduction)}** ({complaint_reduction/max(1,kpis['total_complaints'])*100:.1f}%)")
    
    st.markdown("---")
    
    # Scenario 2: Add Riders
    st.markdown("### 🏍️ Scenario 2: Expand Rider Fleet")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        current_active_riders = len(riders[riders['is_active'] == True])
        additional_riders = st.slider("Add riders to fleet", 0, 300, 100, key="rider_slider")
        
        # Calculate impact
        capacity_increase = additional_riders / max(1, current_active_riders) * 100
        pickup_wait_reduction = min(kpis['avg_pickup_wait'] * 0.5, capacity_increase * 0.2)  # Diminishing returns
        new_pickup_wait = max(5, kpis['avg_pickup_wait'] - pickup_wait_reduction)
        
        late_reduction_pct2 = min(30, capacity_increase * 0.3)
        new_late_deliveries2 = kpis['late_deliveries'] * (1 - late_reduction_pct2 / 100)
        late_reduction_count2 = kpis['late_deliveries'] - new_late_deliveries2
        complaint_reduction2 = late_reduction_count2 / 5
        
        monthly_cost = additional_riders * 5000  # AED 5000 per rider per month
    
    with col2:
        st.markdown("#### 📈 Projected Impact")
        
        impact_data2 = pd.DataFrame({
            'Metric': ['Active Riders', 'Avg Pickup Wait', 'Late Deliveries', 'Complaints'],
            'Current': [f"{current_active_riders:,}", f"{kpis['avg_pickup_wait']:.1f} min", 
                       f"{int(kpis['late_deliveries']):,}", f"{int(kpis['total_complaints']):,}"],
            'Projected': [f"{current_active_riders + additional_riders:,}", f"{new_pickup_wait:.1f} min", 
                         f"{int(new_late_deliveries2):,}", f"{int(max(0, kpis['total_complaints'] - complaint_reduction2)):,}"],
            'Change': [f"+{additional_riders}", f"-{pickup_wait_reduction:.1f} min", 
                      f"-{int(late_reduction_count2):,}", f"-{int(complaint_reduction2):,}"]
        })
        st.dataframe(impact_data2, use_container_width=True, hide_index=True)
        
        st.info(f"💰 **Monthly Cost**: AED {monthly_cost:,} | **Complaints Reduced**: {int(complaint_reduction2)} | **Cost per Complaint Avoided**: AED {monthly_cost/max(1, complaint_reduction2):.0f}")
    
    st.markdown("---")
    
    # Scenario 3: Promo Optimization
    st.markdown("### 💳 Scenario 3: Optimize Promo Strategy")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        promo_reduction = st.slider("Reduce discount rates by (%)", 0, 50, 20, key="promo_slider")
        
        current_discount = kpis['total_discount']
        reduced_discount = current_discount * (1 - promo_reduction / 100)
        savings = current_discount - reduced_discount
        
        # Estimate order loss
        order_loss_pct = promo_reduction * 0.15  # 0.15% order loss per 1% discount reduction
        order_loss = kpis['total_orders'] * order_loss_pct / 100
        gmv_loss = order_loss * kpis['avg_order_value']
    
    with col2:
        st.markdown("#### 📈 Projected Impact")
        
        net_benefit = savings - gmv_loss
        
        impact_data3 = pd.DataFrame({
            'Metric': ['Total Discounts', 'Discount Burn Rate', 'Est. Order Loss', 'Net Benefit'],
            'Current': [f"AED {current_discount:,.0f}", f"{kpis['discount_burn_rate']:.1f}%", '-', '-'],
            'Projected': [f"AED {reduced_discount:,.0f}", f"{kpis['discount_burn_rate'] * (1-promo_reduction/100):.1f}%", 
                         f"{int(order_loss):,} orders", f"AED {net_benefit:,.0f}"],
            'Change': [f"-AED {savings:,.0f}", f"-{promo_reduction}%", 
                      f"-{order_loss_pct:.1f}%", "Net" if net_benefit > 0 else "Loss"]
        })
        st.dataframe(impact_data3, use_container_width=True, hide_index=True)
        
        if net_benefit > 0:
            st.success(f"✅ Net positive impact of **AED {net_benefit:,.0f}** from reduced discounts")
        else:
            st.warning(f"⚠️ Net negative impact - discount reduction may hurt more than help")
    
    st.markdown("---")
    
    # Summary Recommendations
    st.markdown("### 💡 Recommendations Summary")
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>Based on the What-If Analysis:</strong><br><br>
    
    1. <strong>Reduce Prep Time by 5 mins</strong> → Could reduce <strong>{int(late_reduction_count)} late deliveries</strong> and <strong>{int(complaint_reduction)} complaints</strong><br><br>
    
    2. <strong>Add 100 Riders</strong> → Could reduce <strong>{int(late_reduction_count2)} late deliveries</strong> at a cost of <strong>AED {100*5000:,}/month</strong><br><br>
    
    3. <strong>Reduce Discounts by 20%</strong> → Could save <strong>AED {current_discount * 0.2:,.0f}</strong> with minimal order impact<br><br>
    
    <strong>🎯 Priority:</strong> Focus on prep time reduction first (lowest cost, high impact), then consider fleet expansion for peak hours.
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application with view toggle"""
    
    # Load data
    try:
        customers, restaurants, riders, orders, delivery_events, complaints = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run `python data_generator.py` first to generate the data.")
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("# 🍔 BitesUAE")
    st.sidebar.markdown("### Analytics Dashboard")
    st.sidebar.markdown("---")
    
    # View Toggle (Executive vs Manager)
    st.sidebar.markdown("## 🔄 Dashboard View")
    view_mode = st.sidebar.radio(
        "Select View",
        ["📊 Executive View", "⚙️ Manager View", "🔍 Zone Drill-Down", "🔮 What-If Analysis"],
        key="view_toggle"
    )
    
    st.sidebar.markdown("---")
    
    # Render filters
    (date_range, selected_city, selected_zones, selected_cuisines, 
     selected_tiers, selected_segments, preset) = render_sidebar_filters(orders, restaurants, customers)
    
    # Apply time preset override
    if preset != "Custom":
        max_date = orders['order_time'].max().date()
        days_map = {'Last 7 Days': 7, 'Last 30 Days': 30, 'Last 90 Days': 90}
        min_date = max_date - timedelta(days=days_map[preset])
        date_range = (min_date, max_date)
    
    # Apply filters
    filtered_orders, filtered_delivery, filtered_complaints = apply_filters(
        orders, delivery_events, complaints, restaurants, riders,
        date_range, selected_city, selected_zones, selected_cuisines, 
        selected_tiers, selected_segments, customers
    )
    
    # Calculate KPIs on filtered data
    kpis = calculate_comprehensive_kpis(filtered_orders, filtered_delivery, filtered_complaints, customers)
    
    # Data info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Data Summary")
    st.sidebar.markdown(f"**Orders**: {len(filtered_orders):,}")
    st.sidebar.markdown(f"**Date Range**: {date_range[0]} to {date_range[1]}")
    st.sidebar.markdown(f"**Filters Active**: {sum([selected_city != 'All', 'All' not in selected_zones, 'All' not in selected_cuisines, 'All' not in selected_tiers, 'All' not in selected_segments])}")
    
    # Render selected view
    if view_mode == "📊 Executive View":
        render_executive_view(filtered_orders, filtered_delivery, restaurants, customers, 
                              filtered_complaints, riders, kpis)
    elif view_mode == "⚙️ Manager View":
        render_manager_view(filtered_orders, filtered_delivery, restaurants, customers, 
                            filtered_complaints, riders, kpis)
    elif view_mode == "🔍 Zone Drill-Down":
        render_drilldown_view(filtered_orders, filtered_delivery, restaurants, customers, 
                              filtered_complaints, riders, kpis)
    elif view_mode == "🔮 What-If Analysis":
        render_whatif_view(filtered_orders, filtered_delivery, restaurants, customers, 
                           filtered_complaints, riders, kpis)


if __name__ == "__main__":
    main()
