import sqlite3
import random
from faker import Faker
from datetime import datetime, timedelta

def create_tables(cursor):
    """
    Creates the tables in SQLite.
    1) orders: order data
    2) products: product data
    """
    # Clear existing data in orders and products tables (if any)
    cursor.execute("DELETE FROM orders;")
    cursor.execute("DELETE FROM products;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        retailer_id INTEGER,
        store_id INTEGER,
        customer_id INTEGER,
        timestamp TEXT,
        product_id INTEGER,
        quantity INTEGER,
        regular_price REAL,
        sale_price REAL
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        retailer_id INTEGER,
        store_id INTEGER,
        product_name TEXT,
        product_description TEXT,
        category_name TEXT,
        department_name TEXT
    );
    """)

def generate_fake_data(n_orders=1000, n_products=5, n_customers=10):
    """
    Generates fake data using Faker and returns two lists of tuples:
    1) products_data: to insert into the products table
    2) orders_data: to insert into the orders table

    We use few products and few customers so that there are many combinations 
    (the same customer purchasing the same product at different times) with price variation.
    """
    fake = Faker('en_US')
    Faker.seed(42)
    
    # Generate product data (fixed number of products)
    products_data = []
    for product_id in range(1, n_products + 1):
        retailer_id = random.randint(1, 5)  # Example: 5 retailers
        store_id = random.randint(1, 20)    # Example: 20 stores
        product_name = f"Product_{product_id}"
        product_description = fake.sentence(nb_words=6)
        category_name = random.choice(["Electronics", "Fashion", "Food", "Books", "Toys"])
        department_name = random.choice(["Department_A", "Department_B", "Department_C"])
        
        products_data.append((
            product_id,
            retailer_id,
            store_id,
            product_name,
            product_description,
            category_name,
            department_name
        ))
    
    # Generate order data
    orders_data = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # Approximately 2 years
    
    # List of product IDs (from 1 to n_products)
    product_ids = [i for i in range(1, n_products + 1)]
    
    for _ in range(n_orders):
        retailer_id = random.randint(1, 5)
        store_id = random.randint(1, 20)
        # Limited customers: between 1 and n_customers
        customer_id = random.randint(1, n_customers)
        
        # Random date within the 2-year period
        random_date = fake.date_time_between_dates(datetime_start=start_date, datetime_end=end_date)
        
        product_id = random.choice(product_ids)
        quantity = random.randint(1, 10)
        
        # Regular price between 10 and 500
        regular_price = round(random.uniform(10, 500), 2)
        # 50% chance of a discount (sale price lower than the regular price)
        if random.random() < 0.5:
            sale_price = round(regular_price * random.uniform(0.5, 0.9), 2)
        else:
            sale_price = regular_price
        
        orders_data.append((
            retailer_id,
            store_id,
            customer_id,
            random_date.isoformat(),  # store as text in SQLite
            product_id,
            quantity,
            regular_price,
            sale_price
        ))
    
    return products_data, orders_data

def insert_data(cursor, products_data, orders_data):
    """
    Inserts the generated data into the products and orders tables.
    """
    # Insert products data
    cursor.executemany("""
        INSERT INTO products (
            product_id,
            retailer_id,
            store_id,
            product_name,
            product_description,
            category_name,
            department_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """, products_data)
    
    # Insert orders data
    cursor.executemany("""
        INSERT INTO orders (
            retailer_id,
            store_id,
            customer_id,
            timestamp,
            product_id,
            quantity,
            regular_price,
            sale_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, orders_data)

def main():
    # 1. Connect to SQLite (file example.db)
    conn = sqlite3.connect("/home/patrick/llm_pricing/example.db")
    cursor = conn.cursor()
    
    # 2. Create the tables (if they do not exist)
    create_tables(cursor)
    
    # 3. Generate fake data with few products and customers
    products_data, orders_data = generate_fake_data(
        n_orders=1000,   # many orders
        n_products=5,    # few products
        n_customers=10   # few customers
    )
    
    # 4. Insert data
    insert_data(cursor, products_data, orders_data)
    
    # 5. Commit the transaction
    conn.commit()
    
    # 6. Quick verification: print record counts
    cursor.execute("SELECT COUNT(*) FROM orders;")
    n_orders_db = cursor.fetchone()[0]
    print(f"Number of records in 'orders' table: {n_orders_db}")
    
    cursor.execute("SELECT COUNT(*) FROM products;")
    n_products_db = cursor.fetchone()[0]
    print(f"Number of records in 'products' table: {n_products_db}")
    
    # 7. Close the connection
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
