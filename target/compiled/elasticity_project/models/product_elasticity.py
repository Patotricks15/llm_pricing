import pandas as pd
import numpy as np
import statsmodels.api as sm

def model(dbt, session):
    """
    Calcula a elasticidade de preço por produto (agrupando em períodos semanais)
    e retorna um dataframe com colunas: product_id, elasticity.
    """
    # 1. Ler dados da tabela "orders" referenciada no projeto dbt
    orders_df = dbt.source("raw_data", "orders").to_pandas()

    # 2. Converter timestamp para datetime (se já não estiver)
    orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])

    # 3. Agregar vendas por produto e por período (exemplo: semanal)
    orders_df['week'] = orders_df['timestamp'].dt.to_period('W')

    #    - quantidade total vendida (sum)
    #    - preço médio efetivamente pago (mean) -> "sale_price" 
    product_agg = (
        orders_df
        .groupby(['product_id', 'week'], as_index=False)
        .agg({
            'quantity': 'sum',
            'sale_price': 'mean'
        })
    )

    # 4. Log transform (cuidando para excluir casos de quantity=0 ou price=0)
    product_agg = product_agg[
        (product_agg['quantity'] > 0) & (product_agg['sale_price'] > 0)
    ].copy()
    product_agg['ln_qty'] = np.log(product_agg['quantity'])
    product_agg['ln_price'] = np.log(product_agg['sale_price'])

    # 5. Para cada produto, rodar regressão ln(Q) = alpha + beta ln(P)
    results = []
    for product_id, group in product_agg.groupby('product_id'):
        # Precisamos de pelo menos 2 pontos para estimar uma reta
        if len(group) < 2:
            elasticity = None
        else:
            X = sm.add_constant(group['ln_price'])
            y = group['ln_qty']
            model_fit = sm.OLS(y, X).fit()
            # O coeficiente de ln_price é a elasticidade
            elasticity = model_fit.params['ln_price']

        results.append({
            'product_id': product_id,
            'elasticity': elasticity
        })

    product_elasticity_df = pd.DataFrame(results)

    # Retornar como resultado do modelo
    return product_elasticity_df


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
    identifier = "product_elasticity"
    
    def __repr__(self):
        return 'main."product_elasticity"'


class dbtObj:
    def __init__(self, load_df_function) -> None:
        self.source = lambda *args: source(*args, dbt_load_df_function=load_df_function)
        self.ref = lambda *args, **kwargs: ref(*args, **kwargs, dbt_load_df_function=load_df_function)
        self.config = config
        self.this = this()
        self.is_incremental = False

# COMMAND ----------


