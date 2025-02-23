import sqlite3
import random
from faker import Faker
from datetime import datetime, timedelta

def create_tables(cursor):
    """
    Cria as tabelas no SQLite.
    1) orders: dados de pedidos
    2) products: dados de produtos
    """

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
    Gera dados fake usando Faker e retorna dois arrays de tuplas:
    1) products_data: para inserir em products
    2) orders_data: para inserir em orders

    Usamos poucos produtos e poucos consumidores para que haja muitas
    combinações (mesmo consumidor comprando o mesmo produto em momentos diferentes)
    com variação de preços.
    """
    fake = Faker('en_US')
    Faker.seed(42)
    
    # Gerar dados de produtos (n_products fixos)
    products_data = []
    for product_id in range(1, n_products + 1):
        retailer_id = random.randint(1, 5)  # Exemplo: 5 retailers
        store_id = random.randint(1, 20)    # Exemplo: 20 stores
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
    
    # Gerar dados de orders
    orders_data = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 anos
    
    # Lista de produtos (1 a n_products)
    product_ids = [i for i in range(1, n_products + 1)]
    
    for _ in range(n_orders):
        retailer_id = random.randint(1, 5)
        store_id = random.randint(1, 20)
        # Consumidores limitados: entre 1 e n_customers
        customer_id = random.randint(1, n_customers)
        
        # Data aleatória dentro do período de 2 anos
        random_date = fake.date_time_between_dates(datetime_start=start_date, datetime_end=end_date)
        
        product_id = random.choice(product_ids)
        quantity = random.randint(1, 10)
        
        # Preço regular entre 10 e 500
        regular_price = round(random.uniform(10, 500), 2)
        # 50% de chance de ter um desconto
        if random.random() < 0.5:
            sale_price = round(regular_price * random.uniform(0.5, 0.9), 2)
        else:
            sale_price = regular_price
        
        orders_data.append((
            retailer_id,
            store_id,
            customer_id,
            random_date.isoformat(),  # armazenar como texto no SQLite
            product_id,
            quantity,
            regular_price,
            sale_price
        ))
    
    return products_data, orders_data

def insert_data(cursor, products_data, orders_data):
    """
    Insere os dados gerados nas tabelas products e orders.
    """
    # Inserir produtos
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
    
    # Inserir pedidos
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
    # 1. Conecta no SQLite (arquivo example.db)
    conn = sqlite3.connect("/home/patrick/llm_pricing/example.db")
    cursor = conn.cursor()
    
    # 2. Cria as tabelas (se não existirem)
    create_tables(cursor)
    
    # 3. Gera dados fake com poucos produtos e consumidores
    products_data, orders_data = generate_fake_data(
        n_orders=1000,   # muitas ordens
        n_products=5,    # poucos produtos
        n_customers=10   # poucos consumidores
    )
    
    # 4. Insere dados
    insert_data(cursor, products_data, orders_data)
    
    # 5. Commit
    conn.commit()
    
    # 6. Verifica rapidamente
    cursor.execute("SELECT COUNT(*) FROM orders;")
    n_orders_db = cursor.fetchone()[0]
    print(f"Number of records in 'orders' table: {n_orders_db}")
    
    cursor.execute("SELECT COUNT(*) FROM products;")
    n_products_db = cursor.fetchone()[0]
    print(f"Number of records in 'products' table: {n_products_db}")
    
    # 7. Fecha a conexão
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
