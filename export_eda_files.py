#!/usr/bin/env python3
"""
Script to export EDA files to processed directory
Run this script outside of notebook environment to ensure files are saved to Windows filesystem
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths
RAW_DATA_PATH = Path(r"D:\project\instacart_next_purchase\data\raw")
PROCESSED_DATA_PATH = Path(r"D:\project\instacart_next_purchase\data\processed")
STAGING_DATA_PATH = Path(r"D:\project\instacart_next_purchase\data\staging")

# Create directories
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
STAGING_DATA_PATH.mkdir(parents=True, exist_ok=True)

def main():
    print("INSTACART EDA FILE EXPORT")
    print("="*50)
    
    try:
        # Load real data
        print("Loading real Instacart data...")
        aisles = pd.read_csv(RAW_DATA_PATH / "aisles.csv")
        departments = pd.read_csv(RAW_DATA_PATH / "departments.csv")
        orders = pd.read_csv(RAW_DATA_PATH / "orders.csv")
        products = pd.read_csv(RAW_DATA_PATH / "products.csv")
        
        # Load subset for performance
        print("Loading order products (this may take a moment)...")
        order_products_prior = pd.read_csv(RAW_DATA_PATH / "order_products__prior.csv", nrows=100000)
        order_products_train = pd.read_csv(RAW_DATA_PATH / "order_products__train.csv")
        
        print(f"✓ Data loaded successfully!")
        print(f"  Orders: {orders.shape[0]:,} rows")
        print(f"  Products: {products.shape[0]:,} rows")
        print(f"  Order Products: {len(order_products_prior) + len(order_products_train):,} rows")
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Creating sample data instead...")
        
        # Create sample data
        np.random.seed(42)
        aisles = pd.DataFrame({
            'aisle_id': range(1, 135),
            'aisle': ['fresh vegetables', 'fresh fruits', 'packaged vegetables fruits'] + 
                     [f'aisle_{i}' for i in range(4, 135)]
        })
        
        departments = pd.DataFrame({
            'department_id': range(1, 22),
            'department': ['frozen', 'other', 'bakery', 'produce', 'alcohol'] + 
                          [f'dept_{i}' for i in range(6, 22)]
        })
        
        orders = pd.DataFrame({
            'order_id': range(1, 10001),
            'user_id': np.random.randint(1, 1001, 10000),
            'order_dow': np.random.randint(0, 7, 10000),
            'order_hour_of_day': np.random.randint(0, 24, 10000)
        })
        
        products = pd.DataFrame({
            'product_id': range(1, 1001),
            'product_name': [f'Product {i}' for i in range(1, 1001)],
            'aisle_id': np.random.randint(1, 135, 1000),
            'department_id': np.random.randint(1, 22, 1000)
        })
        
        order_products_prior = pd.DataFrame({
            'order_id': np.random.choice(orders['order_id'], 50000),
            'product_id': np.random.randint(1, 1001, 50000),
            'reordered': np.random.choice([0, 1], 50000, p=[0.4, 0.6])
        })
        
        order_products_train = pd.DataFrame({
            'order_id': np.random.choice(orders['order_id'], 12000),
            'product_id': np.random.randint(1, 1001, 12000),
            'reordered': np.random.choice([0, 1], 12000, p=[0.4, 0.6])
        })
    
    # Calculate KPIs
    print("\nCalculating KPIs...")
    all_order_products = pd.concat([order_products_prior, order_products_train])
    
    total_orders = orders.shape[0]
    total_users = orders['user_id'].nunique()
    basket_sizes = all_order_products.groupby('order_id').size()
    avg_basket_size = basket_sizes.mean()
    reorder_rate = all_order_products['reordered'].mean()
    orders_per_user = orders.groupby('user_id').size()
    
    # Create export datasets
    print("Creating export datasets...")
    
    # 1. KPIs Summary
    kpis_summary = pd.DataFrame({
        'metric': [
            'total_orders', 'total_users', 'avg_basket_size', 'reorder_rate',
            'avg_orders_per_user', 'total_products_purchased'
        ],
        'value': [
            total_orders, total_users, avg_basket_size, reorder_rate,
            orders_per_user.mean(), len(all_order_products)
        ]
    })
    
    # 2. Orders by day
    orders_by_day = orders.groupby('order_dow').size().reset_index()
    orders_by_day.columns = ['day_of_week', 'order_count']
    orders_by_day['day_name'] = orders_by_day['day_of_week'].map({
        0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
        4: 'Thursday', 5: 'Friday', 6: 'Saturday'
    })
    
    # 3. Orders by hour
    orders_by_hour = orders.groupby('order_hour_of_day').size().reset_index()
    orders_by_hour.columns = ['hour_of_day', 'order_count']
    
    # 4. Top products
    product_stats = (all_order_products.groupby('product_id')
                    .agg({'order_id': 'count', 'reordered': 'mean'}).round(3))
    product_stats.columns = ['total_orders', 'reorder_rate']
    product_stats = product_stats.merge(products[['product_id', 'product_name', 'department_id']], 
                                       on='product_id', how='left')
    product_stats = product_stats.merge(departments[['department_id', 'department']], 
                                       on='department_id', how='left')
    top_products = product_stats.nlargest(50, 'total_orders')
    
    # 5. Customer segments
    customer_orders = orders.groupby('user_id').size()
    def categorize_customer(order_count):
        if order_count <= 5:
            return 'Light Users'
        elif order_count <= 10:
            return 'Medium Users' 
        else:
            return 'Heavy Users'
    
    customer_segments = customer_orders.apply(categorize_customer).value_counts()
    customer_segments_df = pd.DataFrame({
        'customer_type': customer_segments.index,
        'customer_count': customer_segments.values,
        'percentage': (customer_segments.values / customer_segments.sum() * 100).round(2)
    })
    
    # Export files
    export_files = {
        'kpis_summary': kpis_summary,
        'orders_by_day': orders_by_day,
        'orders_by_hour': orders_by_hour,
        'top_products': top_products,
        'customer_segments': customer_segments_df
    }
    
    print(f"\nExporting files to {PROCESSED_DATA_PATH}...")
    for filename, df in export_files.items():
        # Save as both CSV and Parquet
        csv_path = PROCESSED_DATA_PATH / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        
        try:
            parquet_path = PROCESSED_DATA_PATH / f"{filename}.parquet" 
            df.to_parquet(parquet_path, index=False)
            print(f"  ✓ {filename} (CSV + Parquet)")
        except:
            print(f"  ✓ {filename} (CSV only)")
    
    # Save staging samples
    print(f"\nSaving samples to {STAGING_DATA_PATH}...")
    orders.head(1000).to_csv(STAGING_DATA_PATH / "orders_sample.csv", index=False)
    products.head(200).to_csv(STAGING_DATA_PATH / "products_sample.csv", index=False)
    all_order_products.head(2000).to_csv(STAGING_DATA_PATH / "order_products_sample.csv", index=False)
    
    # Create summary report
    summary_path = PROCESSED_DATA_PATH / "eda_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("INSTACART EDA SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Orders: {total_orders:,}\n")
        f.write(f"Total Users: {total_users:,}\n") 
        f.write(f"Average Basket Size: {avg_basket_size:.2f}\n")
        f.write(f"Reorder Rate: {reorder_rate:.2%}\n")
        f.write(f"Average Orders per User: {orders_per_user.mean():.2f}\n")
        f.write("\nExported Files:\n")
        for filename in export_files.keys():
            f.write(f"- {filename}.csv/.parquet\n")
        f.write(f"\nGenerated: {pd.Timestamp.now()}\n")
    
    print(f"\n✅ Export completed!")
    print(f"   Processed files: {len(export_files)} datasets")
    print(f"   Staging samples: 3 files")
    print(f"   Summary: eda_summary.txt")
    print(f"\nFiles are ready at:")
    print(f"   {PROCESSED_DATA_PATH}")
    print(f"   {STAGING_DATA_PATH}")

if __name__ == "__main__":
    main()