#!/usr/bin/env python3
"""
Trend Analysis Page for Instacart Dashboard
Temporal patterns, time series analysis, and customer behavior heatmaps
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Plotly config to suppress warnings
plotly_config = {'displayModeBar': False}
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

@st.cache_data
def load_instacart_data():
    """Load real Instacart data for trend analysis"""
    try:
        # Try multiple possible data directory paths
        possible_paths = [
            current_dir / "data" / "raw",
            Path(__file__).parent.parent.parent.parent / "data" / "raw",
            Path("data") / "raw"
        ]
        
        data_dir = None
        for path in possible_paths:
            if path.exists() and (path / "orders.csv").exists():
                data_dir = path
                break
        
        # Load orders data
        if data_dir and (data_dir / "orders.csv").exists():
            orders = pd.read_csv(data_dir / "orders.csv")
            logger.info(f"Loaded {len(orders)} orders from CSV")
            
            # Load products and departments for enriched analysis
            products = pd.read_csv(data_dir / "products.csv") if (data_dir / "products.csv").exists() else pd.DataFrame()
            departments = pd.read_csv(data_dir / "departments.csv") if (data_dir / "departments.csv").exists() else pd.DataFrame()
            aisles = pd.read_csv(data_dir / "aisles.csv") if (data_dir / "aisles.csv").exists() else pd.DataFrame()
            
            # Load order-products data for deeper analysis
            order_products = None
            if (data_dir / "order_products__prior.csv").exists():
                # Load subset for performance
                order_products = pd.read_csv(data_dir / "order_products__prior.csv", nrows=100000)
                logger.info(f"Loaded {len(order_products)} order-product records")
            
            return orders, products, departments, aisles, order_products
        else:
            logger.warning("Real data files not found, using sample data")
            return generate_sample_trend_data()
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return generate_sample_trend_data()

def generate_sample_trend_data():
    """Generate sample data for trend analysis"""
    np.random.seed(42)
    n_orders = 10000
    
    # Generate realistic orders data
    orders = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'user_id': np.random.randint(1, 3000, n_orders),
        'order_number': np.random.randint(1, 50, n_orders),
        'order_dow': np.random.randint(0, 7, n_orders),
        'order_hour_of_day': np.random.randint(6, 24, n_orders),
        'days_since_prior_order': np.random.exponential(7, n_orders)
    })
    
    # Other datasets
    products = pd.DataFrame()
    departments = pd.DataFrame()
    aisles = pd.DataFrame()
    order_products = pd.DataFrame()
    
    return orders, products, departments, aisles, order_products

def show_trend_analysis():
    """Main trend analysis page"""
    st.markdown("# 📈 Trend Analysis & Temporal Patterns")
    st.markdown("Analyze customer behavior patterns, seasonal trends, and time-based insights from Instacart data")
    
    # Load data
    orders, products, departments, aisles, order_products = load_instacart_data()
    
    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Analysis Filters")
    
    # Time period filter
    time_period = st.sidebar.selectbox(
        "Time Granularity",
        ["Hour of Day", "Day of Week", "Days Since Prior Order"]
    )
    
    # User behavior filter
    user_segment = st.sidebar.selectbox(
        "User Segment",
        ["All Users", "New Users (1-5 orders)", "Regular Users (6-20 orders)", "Heavy Users (20+ orders)"]
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⏰ Time Patterns", 
        "📅 Weekly Trends", 
        "🔄 Reorder Behavior", 
        "👥 User Journey"
    ])
    
    with tab1:
        show_time_patterns(orders, user_segment)
    
    with tab2:
        show_weekly_trends(orders, user_segment)
    
    with tab3:
        show_reorder_behavior(orders, user_segment)
    
    with tab4:
        show_user_journey(orders, user_segment)

def filter_by_user_segment(orders, segment):
    """Filter orders by user segment"""
    if segment == "All Users":
        return orders
    elif segment == "New Users (1-5 orders)":
        return orders[orders['order_number'] <= 5]
    elif segment == "Regular Users (6-20 orders)":
        return orders[(orders['order_number'] >= 6) & (orders['order_number'] <= 20)]
    elif segment == "Heavy Users (20+ orders)":
        return orders[orders['order_number'] >= 20]
    else:
        return orders

def show_time_patterns(orders, user_segment):
    """Show hourly and daily patterns"""
    st.markdown("### ⏰ Ordering Patterns by Time")
    
    # Filter data
    filtered_orders = filter_by_user_segment(orders, user_segment)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            peak_hour = filtered_orders['order_hour_of_day'].mode().iloc[0] if len(filtered_orders) > 0 and not filtered_orders['order_hour_of_day'].mode().empty else 14
        except:
            peak_hour = 14
        st.metric("Peak Hour", f"{peak_hour}:00", "Most orders")
    
    with col2:
        try:
            peak_dow = filtered_orders['order_dow'].mode().iloc[0] if len(filtered_orders) > 0 and not filtered_orders['order_dow'].mode().empty else 1
        except:
            peak_dow = 1
        dow_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        peak_dow = min(peak_dow, len(dow_names) - 1)  # Ensure valid index
        st.metric("Peak Day", dow_names[peak_dow], "Most active")
    
    with col3:
        try:
            avg_orders_per_user = filtered_orders['order_number'].mean() if len(filtered_orders) > 0 else 0
        except:
            avg_orders_per_user = 0
        st.metric("Avg Orders/User", f"{avg_orders_per_user:.1f}", "Orders")
    
    with col4:
        try:
            unique_users = filtered_orders['user_id'].nunique() if len(filtered_orders) > 0 else 0
        except:
            unique_users = 0
        st.metric("Active Users", f"{unique_users:,}", "Users")
    
    # Hourly patterns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Orders by Hour of Day")
        hourly_orders = filtered_orders.groupby('order_hour_of_day').size().reset_index()
        hourly_orders.columns = ['hour', 'orders']
        
        fig = px.bar(hourly_orders, x='hour', y='orders',
                    title="Order Distribution by Hour",
                    labels={'hour': 'Hour of Day', 'orders': 'Number of Orders'})
        fig.update_layout(showlegend=False)
        fig.update_traces(marker_color='lightblue')
        st.plotly_chart(fig, width='stretch', config=plotly_config)
    
    with col2:
        st.markdown("#### 📅 Orders by Day of Week")
        dow_orders = filtered_orders.groupby('order_dow').size().reset_index()
        dow_orders.columns = ['dow', 'orders']
        dow_orders['day_name'] = dow_orders['dow'].map({
            0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
            4: 'Thursday', 5: 'Friday', 6: 'Saturday'
        })
        
        fig = px.bar(dow_orders, x='day_name', y='orders',
                    title="Order Distribution by Day of Week",
                    labels={'day_name': 'Day of Week', 'orders': 'Number of Orders'})
        fig.update_layout(showlegend=False)
        fig.update_traces(marker_color='lightcoral')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, width='stretch', config=plotly_config)
    
    # Heatmap
    st.markdown("#### 🔥 Order Intensity Heatmap")
    
    # Create hour-dow heatmap
    heatmap_data = filtered_orders.groupby(['order_dow', 'order_hour_of_day']).size().unstack(fill_value=0)
    
    if not heatmap_data.empty and len(heatmap_data.columns) > 0:
        # Ensure we have the right dimensions
        available_hours = sorted(heatmap_data.columns)
        available_days = sorted(heatmap_data.index)
        
        # Create labels that match the actual data dimensions
        hour_labels = [f"{h}:00" for h in available_hours]
        day_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        day_labels = [day_labels[d] if d < len(day_labels) else f"Day {d}" for d in available_days]
        
        # Create plotly heatmap with matching dimensions
        fig = px.imshow(heatmap_data.values,
                       x=hour_labels,
                       y=day_labels,
                       title="Order Intensity by Day of Week and Hour",
                       labels={'color': 'Number of Orders'},
                       color_continuous_scale='YlOrRd')
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400
        )
        st.plotly_chart(fig, width='stretch', config=plotly_config)
    else:
        st.info("No data available for heatmap visualization")

def show_weekly_trends(orders, user_segment):
    """Show weekly ordering trends and patterns"""
    st.markdown("### 📅 Weekly Trends & Customer Behavior")
    
    filtered_orders = filter_by_user_segment(orders, user_segment)
    
    # Weekly analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Average Order Frequency by Day")
        
        # Calculate average orders per day of week
        dow_stats = filtered_orders.groupby('order_dow').agg({
            'order_id': 'count',
            'user_id': 'nunique',
            'order_number': 'mean'
        }).round(2)
        dow_stats.columns = ['Total Orders', 'Unique Users', 'Avg Order Number']
        dow_stats['Orders per User'] = (dow_stats['Total Orders'] / dow_stats['Unique Users']).round(2)
        
        # Add day names
        dow_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        dow_stats.index = [dow_names[i] for i in dow_stats.index]
        
        st.dataframe(dow_stats, width='stretch')
    
    with col2:
        st.markdown("#### 🎯 Peak vs Off-Peak Analysis")
        
        # Define peak hours (10-16) and peak days (Sat, Sun, Mon)
        peak_hours = list(range(10, 17))
        peak_days = [0, 1, 6]  # Sunday, Monday, Saturday
        
        # Categorize orders
        filtered_orders['is_peak_hour'] = filtered_orders['order_hour_of_day'].isin(peak_hours)
        filtered_orders['is_peak_day'] = filtered_orders['order_dow'].isin(peak_days)
        
        peak_analysis = pd.DataFrame({
            'Category': ['Peak Hours (10-16)', 'Off-Peak Hours', 'Peak Days (Sun,Mon,Sat)', 'Off-Peak Days'],
            'Orders': [
                filtered_orders[filtered_orders['is_peak_hour']].shape[0],
                filtered_orders[~filtered_orders['is_peak_hour']].shape[0],
                filtered_orders[filtered_orders['is_peak_day']].shape[0],
                filtered_orders[~filtered_orders['is_peak_day']].shape[0]
            ]
        })
        
        fig = px.pie(peak_analysis, values='Orders', names='Category',
                    title="Peak vs Off-Peak Distribution")
        st.plotly_chart(fig, width='stretch', config=plotly_config)
    
    # Time between orders analysis
    st.markdown("#### ⏱️ Time Between Orders Analysis")
    
    # Filter out missing values
    valid_gaps = filtered_orders[filtered_orders['days_since_prior_order'].notna()]
    
    if len(valid_gaps) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of days between orders
            fig = px.histogram(valid_gaps, x='days_since_prior_order',
                             title="Distribution of Days Between Orders",
                             labels={'days_since_prior_order': 'Days Since Prior Order', 'count': 'Frequency'},
                             nbins=30)
            st.plotly_chart(fig, width='stretch', config=plotly_config)
        
        with col2:
            # Box plot by order number
            valid_gaps = valid_gaps.copy()
            valid_gaps.loc[:, 'order_group'] = pd.cut(valid_gaps['order_number'], 
                                             bins=[0, 5, 10, 20, 100], 
                                             labels=['1-5', '6-10', '11-20', '20+'])
            
            fig = px.box(valid_gaps, x='order_group', y='days_since_prior_order',
                        title="Days Between Orders by Order Number Group",
                        labels={'order_group': 'Order Number Group', 'days_since_prior_order': 'Days Between Orders'})
            st.plotly_chart(fig, width='stretch', config=plotly_config)

def show_reorder_behavior(orders, user_segment):
    """Show reorder patterns and customer retention"""
    st.markdown("### 🔄 Reorder Behavior & Customer Retention")
    
    filtered_orders = filter_by_user_segment(orders, user_segment)
    
    # User retention analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Order Number Distribution")
        
        order_dist = filtered_orders['order_number'].value_counts().sort_index()
        
        fig = px.bar(x=order_dist.index, y=order_dist.values,
                    title="Distribution of Order Numbers",
                    labels={'x': 'Order Number', 'y': 'Number of Orders'})
        fig.update_traces(marker_color='lightgreen')
        st.plotly_chart(fig, width='stretch', config=plotly_config)
    
    with col2:
        st.markdown("#### 📈 Customer Journey Stages")
        
        # Categorize users by order patterns
        user_stats = filtered_orders.groupby('user_id').agg({
            'order_number': 'max',
            'days_since_prior_order': 'mean'
        }).reset_index()
        
        def categorize_customer(row):
            if row['order_number'] == 1:
                return "One-time Buyer"
            elif row['order_number'] <= 5:
                return "New Customer"
            elif row['order_number'] <= 15:
                return "Regular Customer"
            else:
                return "Loyal Customer"
        
        user_stats['category'] = user_stats.apply(categorize_customer, axis=1)
        category_dist = user_stats['category'].value_counts()
        
        fig = px.pie(values=category_dist.values, names=category_dist.index,
                    title="Customer Segments by Order History")
        st.plotly_chart(fig, width='stretch', config=plotly_config)
    
    # Cohort analysis simulation
    st.markdown("#### 🎯 Customer Retention Patterns")
    
    # Group users by their maximum order number
    retention_data = filtered_orders.groupby('user_id')['order_number'].max().value_counts().sort_index()
    retention_data = retention_data.cumsum() / retention_data.sum() * 100
    retention_data = 100 - retention_data  # Convert to retention rate
    
    fig = px.line(x=retention_data.index, y=retention_data.values,
                 title="Customer Retention by Order Number",
                 labels={'x': 'Order Number', 'y': 'Retention Rate (%)'})
    fig.update_traces(line_color='red', line_width=3)
    st.plotly_chart(fig, width='stretch', config=plotly_config)

def show_user_journey(orders, user_segment):
    """Show user journey and lifecycle analysis"""
    st.markdown("### 👥 User Journey & Lifecycle Analysis")
    
    filtered_orders = filter_by_user_segment(orders, user_segment)
    
    # User lifecycle metrics
    col1, col2, col3 = st.columns(3)
    
    user_metrics = filtered_orders.groupby('user_id').agg({
        'order_id': 'count',
        'order_number': 'max',
        'days_since_prior_order': 'mean'
    }).reset_index()
    user_metrics.columns = ['user_id', 'total_orders', 'max_order_num', 'avg_days_between']
    
    with col1:
        avg_orders = user_metrics['total_orders'].mean()
        st.metric("Avg Orders per User", f"{avg_orders:.1f}", "orders")
    
    with col2:
        avg_lifetime = user_metrics['max_order_num'].mean()
        st.metric("Avg Customer Lifetime", f"{avg_lifetime:.1f}", "order cycles")
    
    with col3:
        avg_frequency = user_metrics['avg_days_between'].mean()
        st.metric("Avg Purchase Frequency", f"{avg_frequency:.1f}", "days")
    
    # User journey visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛒 Order Frequency Distribution")
        
        fig = px.histogram(user_metrics, x='total_orders',
                         title="Distribution of Total Orders per User",
                         labels={'total_orders': 'Total Orders per User', 'count': 'Number of Users'},
                         nbins=20)
        st.plotly_chart(fig, width='stretch', config=plotly_config)
    
    with col2:
        st.markdown("#### ⏰ Purchase Frequency Distribution")
        
        valid_frequency = user_metrics[user_metrics['avg_days_between'].notna()]
        if len(valid_frequency) > 0:
            fig = px.histogram(valid_frequency, x='avg_days_between',
                             title="Distribution of Average Days Between Orders",
                             labels={'avg_days_between': 'Days Between Orders', 'count': 'Number of Users'},
                             nbins=20)
            st.plotly_chart(fig, width='stretch', config=plotly_config)
        else:
            st.info("No purchase frequency data available")
    
    # Customer value analysis
    st.markdown("#### 💰 Customer Value Segmentation")
    
    # Create RFM-like analysis with available data
    user_metrics['recency_score'] = pd.qcut(user_metrics['avg_days_between'].fillna(30), 
                                           q=4, labels=['High', 'Medium-High', 'Medium-Low', 'Low'])
    user_metrics['frequency_score'] = pd.qcut(user_metrics['total_orders'], 
                                            q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    # Create value matrix
    value_matrix = pd.crosstab(user_metrics['frequency_score'], user_metrics['recency_score'], 
                              margins=True, margins_name="Total")
    
    st.dataframe(value_matrix, width='stretch')
    
    # Insights
    st.markdown("#### 💡 Key Insights")
    
    insights = []
    
    # Calculate insights
    peak_hour = filtered_orders['order_hour_of_day'].mode().iloc[0] if len(filtered_orders) > 0 else 14
    peak_dow = filtered_orders['order_dow'].mode().iloc[0] if len(filtered_orders) > 0 else 1
    dow_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    insights.append(f"🕐 **Peak ordering time**: {peak_hour}:00 on {dow_names[peak_dow]}s")
    
    if len(user_metrics) > 0:
        loyal_customers = (user_metrics['total_orders'] >= 10).sum()
        total_customers = len(user_metrics)
        loyalty_rate = (loyal_customers / total_customers) * 100 if total_customers > 0 else 0
        insights.append(f"👑 **Customer loyalty**: {loyalty_rate:.1f}% of users have 10+ orders")
        
        avg_reorder_time = user_metrics['avg_days_between'].mean()
        insights.append(f"🔄 **Reorder frequency**: Customers typically reorder every {avg_reorder_time:.1f} days")
    
    for insight in insights:
        st.markdown(insight)

if __name__ == "__main__":
    st.set_page_config(page_title="Trend Analysis", layout="wide")
    show_trend_analysis()
