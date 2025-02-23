import pandas as pd
import numpy as np
import statsmodels.api as sm

def model(dbt, session):
    """
    Calcula a elasticidade por cliente, analisando:
      - A soma das quantidades (de todos os produtos) que o cliente comprou na semana
      - O preço médio (médio dos sale_prices) na semana
    """
    orders_df = dbt.source("raw_data", "orders").to_pandas()
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
    identifier = "costumer_elasticity"
    
    def __repr__(self):
        return 'main."costumer_elasticity"'


class dbtObj:
    def __init__(self, load_df_function) -> None:
        self.source = lambda *args: source(*args, dbt_load_df_function=load_df_function)
        self.ref = lambda *args, **kwargs: ref(*args, **kwargs, dbt_load_df_function=load_df_function)
        self.config = config
        self.this = this()
        self.is_incremental = False

# COMMAND ----------


