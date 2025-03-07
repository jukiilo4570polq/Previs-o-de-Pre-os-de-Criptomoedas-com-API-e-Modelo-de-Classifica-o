# Previs-o-de-Pre-os-de-Criptomoedas-com-API-e-Modelo-de-Classifica-o
Coletar dados históricos de preços de criptomoedas usando uma API.  Salvar os dados em um arquivo CSV.  Criar uma variável alvo (target) que indica se o preço no próximo dia será maior ou menor.  Treinar um modelo de classificação (ex: RandomForest).  Avaliar a acurácia e visualizar a importância das variáveis.
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Função para coletar dados da API CoinGecko
def fetch_crypto_data(crypto_id, days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop('timestamp', axis=1)
    return df

# Coletando dados do Bitcoin
df = fetch_crypto_data('bitcoin', days=365)

# Salvando em CSV
df.to_csv('bitcoin_prices.csv', index=False)

# Criando a variável alvo (target)
df['price_next_day'] = df['price'].shift(-1)
df['target'] = (df['price_next_day'] > df['price']).astype(int)
df = df.dropna()

# Separando variáveis independentes (X) e dependente (y)
X = df[['price']]
y = df['target']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

# Visualizando a importância das variáveis
importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.title('Importância das Variáveis')
plt.show()
