import pandas as pd
import numpy as np
import statsmodels.api as sm

def model(dbt, session):
    """
    Calcula a elasticidade para cada combinação (cliente, produto).
    Ou seja, vê como variações no preço do produto X para um cliente Y 
    afetam a quantidade comprada pelo mesmo cliente, ao longo do tempo.
    """
    orders_df = dbt.source("raw_data", "orders").to_pandas()
    orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
    orders_df['week'] = orders_df['timestamp'].dt.to_period('W')

    # Agrega por (customer_id, product_id, week)
    cp_agg = (
        orders_df
        .groupby(['customer_id', 'product_id', 'week'], as_index=False)
        .agg({
            'quantity': 'sum',
            'sale_price': 'mean'
        })
    )

    # Exclui linhas de quantity=0 ou sale_price=0 (evitar log(0))
    cp_agg = cp_agg[
        (cp_agg['quantity'] > 0) & (cp_agg['sale_price'] > 0)
    ].copy()

    cp_agg['ln_qty'] = np.log(cp_agg['quantity'])
    cp_agg['ln_price'] = np.log(cp_agg['sale_price'])

    results = []
    for (cust_id, prod_id), group in cp_agg.groupby(['customer_id','product_id']):
        if len(group) < 2:
            elasticity = None
        else:
            X = sm.add_constant(group['ln_price'])
            y = group['ln_qty']
            model_fit = sm.OLS(y, X).fit()
            elasticity = model_fit.params['ln_price']

        results.append({
            'customer_id': cust_id,
            'product_id': prod_id,
            'elasticity': elasticity
        })

    cp_elasticity_df = pd.DataFrame(results)
    return cp_elasticity_df
