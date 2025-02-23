import sqlite3
import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DB_PATH = "/home/patrick/llm_pricing/example.db"  # Ajuste conforme seu caso

def create_result_tables():
    """
    Cria 3 tabelas (caso não existam) para armazenar as elasticidades:
      1. computed_product_elasticities
      2. computed_customer_elasticities
      3. computed_c_p_elasticities
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tabela de elasticidade por produto
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS computed_product_elasticities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            price_type TEXT,
            elasticity REAL
        );
    """)

    # Tabela de elasticidade por cliente
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS computed_customer_elasticities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            price_type TEXT,
            elasticity REAL
        );
    """)

    # Tabela de elasticidade por (cliente, produto)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS computed_c_p_elasticities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            product_id INTEGER,
            price_type TEXT,
            elasticity REAL
        );
    """)

    conn.commit()
    conn.close()

def compute_price_elasticity(df: pd.DataFrame, price_col: str) -> float:
    """
    Dado um DataFrame com colunas:
      - 'quantity'
      - <price_col> (ex.: 'regular_price' ou 'sale_price')
    faz a regressão log-log para obter a elasticidade:
        ln(quantity) = alpha + beta ln(price_col)
    Retorna o valor de 'beta' (a elasticidade).
    Levanta ValueError se não houver dados suficientes.
    """
    # Elimina linhas inválidas (quantity <=0 ou price <=0)
    df = df[(df["quantity"] > 0) & (df[price_col] > 0)].copy()
    if len(df) < 2:
        raise ValueError("Não há dados suficientes para calcular elasticidade (precisa de >=2 pontos).")

    df["log_quantity"] = df["quantity"].apply(math.log)
    df["log_price"] = df[price_col].apply(math.log)

    # Regressão OLS usando fórmula
    model = smf.ols("log_quantity ~ log_price", data=df).fit()
    elasticity = model.params["log_price"]
    return np.abs(elasticity)

def compute_elasticities():
    """
    1. Lê toda a tabela 'orders' do DB.
    2. Calcula elasticidade para cada product_id, cada customer_id e (customer_id, product_id).
    3. Grava resultados em 3 tabelas separadas.
    """

    # Conexão e leitura
    conn = sqlite3.connect(DB_PATH)
    df_orders = pd.read_sql_query("SELECT * FROM orders;", conn)
    conn.close()

    # 1) Identificar IDs distintos
    product_ids = df_orders["product_id"].dropna().unique().tolist()
    customer_ids = df_orders["customer_id"].dropna().unique().tolist()

    # Criar ou limpar as tabelas de resultados
    create_result_tables()

    # Abrir conexão p/ inserir resultados
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ----------------------------------------------------------------------------
    # ELASTICIDADE POR PRODUTO
    # ----------------------------------------------------------------------------
    price_types = ["regular_price", "sale_price"]  # correspondem às colunas no DB
    for pid in product_ids:
        # Filtra o DataFrame para apenas esse produto
        df_prod = df_orders[df_orders["product_id"] == pid]

        for pt in price_types:
            try:
                elasticity_val = compute_price_elasticity(df_prod, pt)
            except ValueError:
                # Se não houver dados suficientes, ignore ou salve como NULL
                elasticity_val = None

            # Insere na tabela
            cursor.execute("""
                INSERT INTO computed_product_elasticities (product_id, price_type, elasticity)
                VALUES (?, ?, ?);
            """, (pid, pt, elasticity_val))

    # ----------------------------------------------------------------------------
    # ELASTICIDADE POR CLIENTE
    # ----------------------------------------------------------------------------
    for cid in customer_ids:
        # Filtra o DataFrame para apenas esse cliente
        df_cust = df_orders[df_orders["customer_id"] == cid]

        for pt in price_types:
            try:
                elasticity_val = np.abs(compute_price_elasticity(df_cust, pt))
            except ValueError:
                elasticity_val = None

            cursor.execute("""
                INSERT INTO computed_customer_elasticities (customer_id, price_type, elasticity)
                VALUES (?, ?, ?);
            """, (cid, pt, elasticity_val))

    # ----------------------------------------------------------------------------
    # ELASTICIDADE POR (CLIENTE, PRODUTO)
    # ----------------------------------------------------------------------------
    # Em vez de gerar todas as combinações, vamos extrair diretamente do df_orders
    # ou poderíamos mesclar os sets, mas melhor usar groupby e iterar
    df_cust_prod = df_orders[["customer_id","product_id"]].drop_duplicates()
    # df_cust_prod tem pares únicos (customer_id, product_id)

    for _, row in df_cust_prod.iterrows():
        cid = row["customer_id"]
        pid = row["product_id"]
        # Filtrar
        df_sub = df_orders[(df_orders["customer_id"] == cid) & (df_orders["product_id"] == pid)]
        print(df_sub)

        for pt in price_types:
            try:
                elasticity_val = compute_price_elasticity(df_sub, pt)
            except ValueError:
                elasticity_val = None

            cursor.execute("""
                INSERT INTO computed_c_p_elasticities (customer_id, product_id, price_type, elasticity)
                VALUES (?, ?, ?, ?);
            """, (cid, pid, pt, elasticity_val))
            print(elasticity_val)

    # ----------------------------------------------------------------------------
    # Comitar e encerrar
    # ----------------------------------------------------------------------------
    conn.commit()
    conn.close()


if __name__ == "__main__":
    compute_elasticities()
    print("Elasticidades calculadas e armazenadas nas tabelas:")
    print(" - computed_product_elasticities")
    print(" - computed_customer_elasticities")
    print(" - computed_c_p_elasticities")
