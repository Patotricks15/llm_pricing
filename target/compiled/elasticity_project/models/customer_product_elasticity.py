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


# This part is user provided model code
# you will need to copy the next section to run the code
# COMMAND ----------
# this part is dbt logic for get ref work, do not modify

def ref(*args, **kwargs):
    refs = {}
    key = '.'.join(args)
    version = kwargs.get("v") or kwargs.get("version")
    if version:
        key += f".v{version}"
    dbt_load_df_function = kwargs.get("dbt_load_df_function")
    return dbt_load_df_function(refs[key])


def source(*args, dbt_load_df_function):
    sources = {"raw_data.orders": "main.\"orders\""}
    key = '.'.join(args)
    return dbt_load_df_function(sources[key])


config_dict = {}


class config:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def get(key, default=None):
        return config_dict.get(key, default)

class this:
    """dbt.this() or dbt.this.identifier"""
    database = "/home/patrick/llm_pricing/example.db"
    schema = "main"
    identifier = "customer_product_elasticity"
    
    def __repr__(self):
        return 'main."customer_product_elasticity"'


class dbtObj:
    def __init__(self, load_df_function) -> None:
        self.source = lambda *args: source(*args, dbt_load_df_function=load_df_function)
        self.ref = lambda *args, **kwargs: ref(*args, **kwargs, dbt_load_df_function=load_df_function)
        self.config = config
        self.this = this()
        self.is_incremental = False

# COMMAND ----------


