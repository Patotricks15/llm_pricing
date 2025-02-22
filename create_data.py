import sqlite3
import random
from faker import Faker
from datetime import datetime, timedelta

def create_tables(cursor):
    """
    Creates the tables in the SQLite database.
    Here we have:
    1) orders: order data
    2) products: product data
    """
    
    # Orders table
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
    
    # Products table
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

def generate_fake_data(n_orders=1000, n_products=50):
    """
    Generates fake data using Faker and returns lists of tuples
    that can be inserted into the orders and products tables.
    
    n_orders: number of records in the orders table
    n_products: number of records in the products table
    """
    
    fake = Faker('en_US')  # Faker with US locale
    Faker.seed(42)
    
    # Generate product data
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
    
    # Define date range (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years ~ 730 days
    
    product_ids = [i for i in range(1, n_products + 1)]
    
    for _ in range(n_orders):
        retailer_id = random.randint(1, 5)
        store_id = random.randint(1, 20)
        customer_id = random.randint(1000, 9999)
        
        # Generate a random date within the 2-year range
        random_date = fake.date_time_between_dates(datetime_start=start_date, datetime_end=end_date)
        timestamp = random_date.isoformat()
        
        product_id = random.choice(product_ids)
        quantity = random.randint(1, 10)
        
        # Generate prices
        # Example: Regular price between 10 and 500; sale_price can be equal or lower
        regular_price = round(random.uniform(10, 500), 2)
        # 50% chance of having a lower sale price to simulate a discount
        if random.random() < 0.5:
            sale_price = round(regular_price * random.uniform(0.5, 0.9), 2)
        else:
            sale_price = regular_price
        
        orders_data.append((
            retailer_id,
            store_id,
            customer_id,
            timestamp,
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
    
    # Insert product data into the products table
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
    
    # Insert order data into the orders table
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
    # 1. Connect to the database (in this example, SQLite in a local 'example.db' file)
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()
    
    # 2. Create the tables (if they do not exist)
    create_tables(cursor)
    
    # 3. Generate fake data
    products_data, orders_data = generate_fake_data(
        n_orders=1000,   # Adjust to the number of order records you want
        n_products=50    # Adjust to the number of products you want
    )
    
    # 4. Insert the data into the database
    insert_data(cursor, products_data, orders_data)
    
    # 5. Commit the changes
    conn.commit()
    
    # 6. Simple example query to verify the data
    cursor.execute("SELECT COUNT(*) FROM orders;")
    n_orders_db = cursor.fetchone()[0]
    print(f"Number of records in the 'orders' table: {n_orders_db}")
    
    cursor.execute("SELECT COUNT(*) FROM products;")
    n_products_db = cursor.fetchone()[0]
    print(f"Number of records in the 'products' table: {n_products_db}")
    
    # 7. Close the connection
    conn.close()

if __name__ == "__main__":
    main()
