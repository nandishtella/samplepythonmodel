import pandas as pd

def train(concrete_by_flyash):
    X = concrete_by_flyash.drop(['csMPa', 'did_pass'], axis=1)
    y = concrete_by_flyash.csMPa
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=20)
    model = LinearRegression()
    model.fit(qt.fit_transform(X), y)
    print(f'Linear model R^2 = {{model.score(X, y)}}')
    return (X.columns, qt, model)

def predict(model, query):
    columns, qt, model = model
    X = pd.DataFrame({c: [query[c]] for c in columns})
    y = model.predict(qt.transform(X))[0]
    return {'csMPa': float(y)}
