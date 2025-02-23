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
