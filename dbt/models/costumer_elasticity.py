import pandas as pd
import numpy as np
import statsmodels.api as sm

def model(dbt, session):
    """
    Calcula a elasticidade por cliente, analisando:
      - A soma das quantidades (de todos os produtos) que o cliente comprou na semana
      - O preço médio (médio dos sale_prices) na semana
    """
    orders_df = dbt.ref("orders").to_pandas()
    orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
    orders_df['week'] = orders_df['timestamp'].dt.to_period('W')

    # Agrega por (customer_id, semana)
    cust_agg = (
        orders_df
        .groupby(['customer_id', 'week'], as_index=False)
        .agg({
            'quantity': 'sum',
            'sale_price': 'mean'
        })
    )

    # Remover linhas com quantity=0 ou sale_price=0
    cust_agg = cust_agg[
        (cust_agg['quantity'] > 0) & (cust_agg['sale_price'] > 0)
    ].copy()

    cust_agg['ln_qty'] = np.log(cust_agg['quantity'])
    cust_agg['ln_price'] = np.log(cust_agg['sale_price'])

    results = []
    for customer_id, group in cust_agg.groupby('customer_id'):
        # É necessário ao menos 2 pontos para estimar a regressão
        if len(group) < 2:
            elasticity = None
        else:
            X = sm.add_constant(group['ln_price'])
            y = group['ln_qty']
            model_fit = sm.OLS(y, X).fit()
            elasticity = model_fit.params['ln_price']

        results.append({
            'customer_id': customer_id,
            'elasticity': elasticity
        })

    customer_elasticity_df = pd.DataFrame(results)
    return customer_elasticity_df
