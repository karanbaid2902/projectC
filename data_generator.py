"""
BitesUAE Synthetic Data Generator
Generates realistic food delivery data for the dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_CUSTOMERS = 5000
NUM_RESTAURANTS = 200
NUM_RIDERS = 800
NUM_ORDERS = 15000
NUM_DAYS = 90

# Zone definitions
ZONES = {
    'Dubai': ['Marina', 'JBR', 'Downtown', 'Deira', 'Business Bay', 'JLT'],
    'Abu Dhabi': ['Corniche', 'Khalidiya', 'Yas Island', 'Al Reem'],
    'Sharjah': ['Al Nahda', 'Al Qasimia', 'Al Majaz'],
    'Ajman': ['Al Nuaimiya', 'Al Rashidiya']
}

# Flatten zones with city prefixes
ALL_ZONES = []
ZONE_TO_CITY = {}
for city, zones in ZONES.items():
    for zone in zones:
        full_zone = f"{zone} ({city})"
        ALL_ZONES.append(full_zone)
        ZONE_TO_CITY[full_zone] = city

# Restaurant categories
CUISINE_TYPES = ['Emirati', 'Indian', 'Asian', 'Western', 'Healthy']
RESTAURANT_TIERS = ['QSR', 'Casual', 'Premium', 'Fine Dining']

# Promo codes
PROMO_CODES = ['WELCOME10', 'SAVE20', 'FREESHIP', 'WEEKEND15', 'LOYALTY25', 'FLASH30', 'NEW50', None]

# Order statuses
ORDER_STATUSES = ['Delivered', 'Cancelled', 'In Progress']

# Cancellation reasons
CANCELLATION_REASONS = [
    'Customer Request',
    'Restaurant Unavailable', 
    'Rider Unavailable',
    'Long Wait Time',
    'Payment Issue',
    'Address Issue'
]

# Complaint categories
COMPLAINT_CATEGORIES = [
    'Late Delivery',
    'Cold Food',
    'Wrong Order',
    'Missing Items',
    'Damaged Package',
    'Rude Rider'
]


def generate_customers():
    """Generate customer data"""
    customers = []
    for i in range(1, NUM_CUSTOMERS + 1):
        customers.append({
            'customer_id': f'CUST{i:05d}',
            'customer_name': f'Customer_{i}',
            'registration_date': datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365)),
            'preferred_zone': random.choice(ALL_ZONES),
            'customer_segment': random.choices(['New', 'Regular', 'Premium', 'VIP'], 
                                               weights=[30, 40, 20, 10])[0]
        })
    return pd.DataFrame(customers)


def generate_restaurants():
    """Generate restaurant data"""
    restaurants = []
    tier_avg_prep_times = {'QSR': 10, 'Casual': 20, 'Premium': 30, 'Fine Dining': 45}
    
    for i in range(1, NUM_RESTAURANTS + 1):
        tier = random.choices(RESTAURANT_TIERS, weights=[40, 35, 18, 7])[0]
        restaurants.append({
            'restaurant_id': f'REST{i:04d}',
            'restaurant_name': f'Restaurant_{i}',
            'cuisine_type': random.choice(CUISINE_TYPES),
            'tier': tier,
            'zone': random.choice(ALL_ZONES),
            'avg_prep_time_mins': tier_avg_prep_times[tier] + random.randint(-5, 10),
            'rating': round(random.uniform(3.0, 5.0), 1),
            'is_active': random.choices([True, False], weights=[95, 5])[0]
        })
    return pd.DataFrame(restaurants)


def generate_riders():
    """Generate rider data"""
    riders = []
    for i in range(1, NUM_RIDERS + 1):
        riders.append({
            'rider_id': f'RIDER{i:04d}',
            'rider_name': f'Rider_{i}',
            'primary_zone': random.choice(ALL_ZONES),
            'vehicle_type': random.choices(['Bike', 'Motorcycle', 'Car'], weights=[40, 50, 10])[0],
            'rating': round(random.uniform(3.5, 5.0), 1),
            'join_date': datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730)),
            'is_active': random.choices([True, False], weights=[90, 10])[0]
        })
    return pd.DataFrame(riders)


def generate_orders(customers_df, restaurants_df, riders_df):
    """Generate order data using vectorized operations"""
    n = NUM_ORDERS
    
    # Date range for 90 days
    end_date = datetime(2025, 12, 31)
    start_date = end_date - timedelta(days=NUM_DAYS)
    
    # Customer order frequency (some customers order more)
    customer_weights = np.random.exponential(scale=1.5, size=len(customers_df))
    customer_weights = customer_weights / customer_weights.sum()
    
    # Generate all random selections at once
    customer_indices = np.random.choice(len(customers_df), size=n, p=customer_weights)
    restaurant_indices = np.random.choice(len(restaurants_df), size=n)
    active_riders = riders_df[riders_df['is_active'] == True].reset_index(drop=True)
    rider_indices = np.random.choice(len(active_riders), size=n)
    
    # Get IDs and tiers
    customer_ids = customers_df.iloc[customer_indices]['customer_id'].values
    restaurant_data = restaurants_df.iloc[restaurant_indices]
    restaurant_ids = restaurant_data['restaurant_id'].values
    zones = restaurant_data['zone'].values
    tiers = restaurant_data['tier'].values
    rider_ids = active_riders.iloc[rider_indices]['rider_id'].values
    
    # Order timing with peak hours bias
    hour_weights = np.array([1,1,1,1,1,1,2,3,4,5,6,8,10,6,5,5,6,7,9,10,8,6,4,2], dtype=float)
    hour_weights = hour_weights / hour_weights.sum()
    days = np.random.randint(0, NUM_DAYS, size=n)
    hours = np.random.choice(24, size=n, p=hour_weights)
    minutes = np.random.randint(0, 60, size=n)
    
    order_times = pd.to_datetime([
        start_date + timedelta(days=int(d), hours=int(h), minutes=int(m))
        for d, h, m in zip(days, hours, minutes)
    ])
    
    # Order values based on tier
    tier_multiplier = {'QSR': 1, 'Casual': 1.5, 'Premium': 2.5, 'Fine Dining': 4}
    base_values = np.random.uniform(30, 80, size=n)
    multipliers = np.array([tier_multiplier[t] for t in tiers])
    order_values = np.round(base_values * multipliers, 2)
    
    # Promo codes
    promo_weights = np.array([10, 15, 12, 10, 8, 5, 3, 37], dtype=float)
    promo_weights = promo_weights / promo_weights.sum()
    promo_indices = np.random.choice(len(PROMO_CODES), size=n, p=promo_weights)
    promo_codes = [PROMO_CODES[i] for i in promo_indices]
    
    # Calculate discounts
    discount_amounts = []
    for i, promo in enumerate(promo_codes):
        if promo:
            discount_pct = int(''.join(filter(str.isdigit, promo))) if any(c.isdigit() for c in promo) else 10
            discount_amounts.append(round(order_values[i] * discount_pct / 100, 2))
        else:
            discount_amounts.append(None)
    
    # Order status
    status_weights = np.array([85, 10, 5], dtype=float)
    status_weights = status_weights / status_weights.sum()
    status_indices = np.random.choice(len(ORDER_STATUSES), size=n, p=status_weights)
    statuses = [ORDER_STATUSES[i] for i in status_indices]
    
    # Cancellation reasons
    cancellation_reasons = [random.choice(CANCELLATION_REASONS) if s == 'Cancelled' else None for s in statuses]
    
    # Delivery fees
    delivery_fees = np.round(np.random.uniform(5, 15, size=n), 2)
    
    # Create dataframe
    df = pd.DataFrame({
        'order_id': [f'ORD{i:06d}' for i in range(1, n + 1)],
        'customer_id': customer_ids,
        'restaurant_id': restaurant_ids,
        'rider_id': rider_ids,
        'order_time': order_times,
        'zone': zones,
        'order_value': order_values,
        'promo_code': promo_codes,
        'discount_amount': discount_amounts,
        'delivery_fee': delivery_fees,
        'status': statuses,
        'cancellation_reason': cancellation_reasons
    })
    
    # Inject missing data as per requirements (~300 missing promo_code, ~100 missing discount_amount)
    missing_promo_idx = df.sample(300).index
    df.loc[missing_promo_idx, 'promo_code'] = None
    
    promo_not_null = df[df['promo_code'].notna()]
    if len(promo_not_null) >= 100:
        missing_discount_idx = promo_not_null.sample(100).index
        df.loc[missing_discount_idx, 'discount_amount'] = None
    
    return df


def generate_delivery_events(orders_df, restaurants_df):
    """Generate delivery event timestamps using vectorized operations"""
    n = len(orders_df)
    
    # Merge orders with restaurant prep times
    merged = orders_df.merge(restaurants_df[['restaurant_id', 'avg_prep_time_mins']], on='restaurant_id', how='left')
    
    # Generate random values for all events at once
    confirm_delays = np.random.randint(1, 6, size=n)  # 1-5 mins
    prep_variations = np.random.randint(-5, 16, size=n)  # -5 to 15 mins
    pickup_delays = np.random.randint(5, 21, size=n)  # 5-20 mins
    significant_delay_mask = np.random.random(n) < 0.15
    extra_pickup_delays = np.where(significant_delay_mask, np.random.randint(10, 31, size=n), 0)
    delivery_times = np.random.randint(10, 41, size=n)  # 10-40 mins
    traffic_delay_mask = np.random.random(n) < 0.1
    extra_delivery_times = np.where(traffic_delay_mask, np.random.randint(10, 26, size=n), 0)
    
    # Calculate times
    order_times = pd.to_datetime(merged['order_time'])
    restaurant_confirmed = order_times + pd.to_timedelta(confirm_delays, unit='m')
    
    prep_times = np.maximum(5, merged['avg_prep_time_mins'].fillna(20).astype(int) + prep_variations)
    food_ready = restaurant_confirmed + pd.to_timedelta(prep_times, unit='m')
    
    total_pickup_delay = pickup_delays + extra_pickup_delays
    rider_picked_up = food_ready + pd.to_timedelta(total_pickup_delay, unit='m')
    
    total_delivery_time = delivery_times + extra_delivery_times
    delivered = rider_picked_up + pd.to_timedelta(total_delivery_time, unit='m')
    
    expected_delivery = order_times + pd.to_timedelta(45, unit='m')
    
    # Create dataframe
    df = pd.DataFrame({
        'event_id': [f'EVT{i:06d}' for i in range(1, n + 1)],
        'order_id': merged['order_id'],
        'restaurant_confirmed_time': restaurant_confirmed,
        'food_ready_time': food_ready,
        'rider_picked_up_time': rider_picked_up,
        'delivered_time': delivered,
        'expected_delivery_time': expected_delivery
    })
    
    # Handle cancelled and in-progress orders
    cancelled_mask = merged['status'].isin(['Cancelled', 'In Progress'])
    df.loc[cancelled_mask, 'delivered_time'] = pd.NaT
    
    # Calculate on-time
    df['is_on_time'] = df['delivered_time'] <= df['expected_delivery_time']
    df.loc[df['delivered_time'].isna(), 'is_on_time'] = None
    
    return df


def generate_complaints(orders_df):
    """Generate customer complaints"""
    complaints = []
    
    # About 8% of orders get complaints
    complaint_orders = orders_df[orders_df['status'] == 'Delivered'].sample(frac=0.08)
    
    for _, order in complaint_orders.iterrows():
        complaint_time = order['order_time'] + timedelta(hours=random.randint(1, 24))
        
        complaints.append({
            'complaint_id': f'CMP{len(complaints)+1:05d}',
            'order_id': order['order_id'],
            'customer_id': order['customer_id'],
            'complaint_category': random.choice(COMPLAINT_CATEGORIES),
            'complaint_time': complaint_time,
            'zone': order['zone'],
            'is_resolved': random.choices([True, False], weights=[75, 25])[0],
            'resolution_time_hours': random.randint(1, 48) if random.random() > 0.25 else None
        })
    
    return pd.DataFrame(complaints)


def main():
    """Generate all data and save to CSV files"""
    print("Generating BitesUAE synthetic data...")
    
    # Generate data
    print("  - Generating customers...")
    customers_df = generate_customers()
    
    print("  - Generating restaurants...")
    restaurants_df = generate_restaurants()
    
    print("  - Generating riders...")
    riders_df = generate_riders()
    
    print("  - Generating orders...")
    orders_df = generate_orders(customers_df, restaurants_df, riders_df)
    
    print("  - Generating delivery events...")
    delivery_events_df = generate_delivery_events(orders_df, restaurants_df)
    
    print("  - Generating complaints...")
    complaints_df = generate_complaints(orders_df)
    
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    print("\nSaving data to CSV files...")
    customers_df.to_csv(f'{data_dir}/customers.csv', index=False)
    restaurants_df.to_csv(f'{data_dir}/restaurants.csv', index=False)
    riders_df.to_csv(f'{data_dir}/riders.csv', index=False)
    orders_df.to_csv(f'{data_dir}/orders.csv', index=False)
    delivery_events_df.to_csv(f'{data_dir}/delivery_events.csv', index=False)
    complaints_df.to_csv(f'{data_dir}/complaints.csv', index=False)
    
    print("\nData generation complete!")
    print(f"  - Customers: {len(customers_df)}")
    print(f"  - Restaurants: {len(restaurants_df)}")
    print(f"  - Riders: {len(riders_df)}")
    print(f"  - Orders: {len(orders_df)}")
    print(f"  - Delivery Events: {len(delivery_events_df)}")
    print(f"  - Complaints: {len(complaints_df)}")


if __name__ == "__main__":
    main()
