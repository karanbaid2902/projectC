# BitesUAE Analytics Dashboard

A comprehensive Streamlit dashboard for food delivery analytics, built for BitesUAE operations.

## Features

### 📊 Executive View
- GMV trends and performance tracking
- Zone and cuisine performance analysis
- Customer segmentation metrics
- Promo code effectiveness analysis
- Auto-generated insights

### ⚙️ Manager View
- Operational metrics (on-time rate, delivery time, cancellation rate)
- Top 10 problem areas table
- Delay analysis (prep time vs pickup delay)
- Peak hours performance analysis
- Cancellation and complaint analysis

### 🔍 Zone Drill-Down
- Zone-specific performance metrics
- Restaurant performance within zone
- Rider performance and retraining recommendations
- Detailed prep time and delivery analysis

### 🔮 What-If Analysis
- Prep time reduction scenarios
- Fleet expansion impact projections
- Promo optimization analysis
- High-value zone focus recommendations

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate sample data:
```bash
python data_generator.py
```

4. Run the dashboard:
```bash
streamlit run dashboard.py
```

## Data Structure

The dashboard uses 6 interconnected tables:

1. **customers.csv** - Customer information and segments
2. **restaurants.csv** - Restaurant details including tier and cuisine type
3. **riders.csv** - Rider information and ratings
4. **orders.csv** - Order transactions with promo and discount data
5. **delivery_events.csv** - Delivery timestamps for tracking delays
6. **complaints.csv** - Customer complaints and resolutions

## Key Metrics Tracked

- **GMV (Gross Merchandise Value)** - Total revenue from delivered orders
- **Repeat Rate** - Percentage of customers with multiple orders
- **On-Time Delivery Rate** - Percentage of orders delivered within expected time
- **Average Delivery Time** - Mean time from pickup to delivery
- **Cancellation Rate** - Percentage of cancelled orders
- **Discount Burn Rate** - Percentage of GMV lost to discounts
- **Complaint Rate** - Percentage of orders with complaints

## Business Questions Answered

1. What is our GMV trend over the last 90 days, and which zones contribute the most?
2. What is our discount burn rate, and which promo codes are driving orders vs. just reducing margins?
3. What percentage of customers are repeat buyers, and what is their average order frequency?
4. Which restaurant tier (QSR, Casual, Premium, Fine Dining) has the best unit economics?
5. What is our on-time delivery rate by zone, and which zones have the worst performance?
6. What are the top 3 cancellation reasons, and which zones have the highest cancellation rates?
7. Which riders have the longest average delivery times, and should we provide retraining?

## Project Structure

```
project/
├── data_generator.py     # Generates synthetic data
├── dashboard.py          # Main Streamlit dashboard
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── data/                 # Generated data files
    ├── customers.csv
    ├── restaurants.csv
    ├── riders.csv
    ├── orders.csv
    ├── delivery_events.csv
    └── complaints.csv
```

## Author

Built for BitesUAE Analytics Team
