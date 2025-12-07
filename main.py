import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import tqdm

# --- CONFIGURACIÓN ---
tickers_bvc = ["ECOPETROL.CL", "ISA.CL", "GRUPOARGOS.CL", "GEB.CL"]
FECHA_CORTE = "2023-12-31"  # El modelo NO verá nada después de esta fecha para entrenar


# --- FUNCIONES (Mismas del modelo avanzado) ---
def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def preparar_datos(df):
    df_ml = df.copy()
    # Objetivo: Retorno del día siguiente
    df_ml["Target_Return"] = df_ml["Close"].pct_change().shift(-1)

    # Features (Indicadores)
    df_ml["Return_1d"] = df_ml["Close"].pct_change()
    df_ml["Return_5d"] = df_ml["Close"].pct_change(5)
    df_ml["SMA_10"] = df_ml["Close"].rolling(window=10).mean()
    df_ml["Dist_SMA_10"] = df_ml["Close"] / df_ml["SMA_10"]
    df_ml["Volatility"] = df_ml["Close"].rolling(window=10).std()
    df_ml["RSI"] = calcular_rsi(df_ml["Close"])

    df_ml = df_ml.dropna()
    return df_ml


print(f"--- Iniciando Entrenamiento (Corte: {FECHA_CORTE}) ---")

rd_models = {}

for t in tickers_bvc:
    print(f"\nProcesando: {t}...")

    # Descargamos suficiente historia para entrenar bien
    dfabi = yf.download(
        t, period="max", interval="1d", auto_adjust=True, progress=False
    )

    if len(dfabi) > 0:
        if isinstance(dfabi.columns, pd.MultiIndex):
            dfabi.columns = dfabi.columns.get_level_values(0)

        df_processed = preparar_datos(dfabi)

        features = ["Return_1d", "Return_5d", "Dist_SMA_10", "Volatility", "RSI"]

        # --- AQUÍ ESTÁ EL CAMBIO CLAVE: CORTE POR FECHA ---
        # Máscaras booleanas para separar el tiempo
        mask_train = df_processed.index <= FECHA_CORTE

        X_train = df_processed.loc[mask_train, features]
        y_train = df_processed.loc[mask_train, "Target_Return"]

        print(f"   -> Datos Entrenamiento (Hasta 2024): {len(X_train)} días")

        # Entrenar Modelo
        model = RandomForestRegressor(
            n_estimators=150, max_depth=5, min_samples_leaf=5, random_state=42
        )
        model.fit(X_train, y_train)

        rd_models.update({t: model})
    else:
        print(f"No se pudieron descargar datos para {t}")


market_data_store = {}

print("\n--- Preparando datos para el Entorno de RL ---")
for t in tickers_bvc:
    # Descargamos de nuevo para persistencia (optimizacion simple)
    df_temp = yf.download(
        t, period="max", interval="1d", auto_adjust=True, progress=False
    )
    if len(df_temp) > 0:
        if isinstance(df_temp.columns, pd.MultiIndex):
            df_temp.columns = df_temp.columns.get_level_values(0)

        # Procesamos con TU función original
        df_proc = preparar_datos(df_temp)
        market_data_store[t] = df_proc
        print(f"Datos cargados para RL: {t} - Filas: {len(df_proc)}")

# --- OPTIMIZACIÓN: PRE-CALCULAR PREDICCIONES ---
print("--- Optimizando datos (Pre-calculando RF) ---")

# Diccionario optimizado
market_data_opt = {}

for t, model in rd_models.items():
    df = market_data_store[t].copy()

    # 1. Predecimos TODO el histórico de una vez
    # (El modelo ya fue entrenado con estos datos o similares,
    #  aquí solo generamos la columna 'signal' para que el RL la lea rápido)
    features_cols = ["Return_1d", "Return_5d", "Dist_SMA_10", "Volatility", "RSI"]

    # Predecimos
    all_predictions = model.predict(df[features_cols])

    # 2. Guardamos la predicción como una columna más en el DataFrame
    df["RF_Prediction"] = all_predictions

    market_data_opt[t] = df

print(
    "Datos optimizados. El entorno RL ahora solo leerá columnas, no ejecutará modelos."
)


class TradingEnvFast:
    def __init__(self, df, features_list, initial_balance=10_000_000):
        self.df = df
        # Ya no necesitamos model_rf aquí
        self.features = features_list
        self.initial_balance = initial_balance

        self.n_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.net_worth_history = [self.initial_balance]

        # Convertimos el DataFrame a numpy array al inicio para velocidad máxima
        # Esto evita usar .iloc (que es lento) dentro del bucle
        # El orden de columnas será: [Features... , RF_Prediction, Close, Target_Return]
        self.obs_data = self.df[self.features].values
        self.rf_preds = self.df["RF_Prediction"].values
        self.prices = self.df["Close"].values
        self.targets = self.df["Target_Return"].values
        self.max_steps = len(df) - 1

    def reset(self, start_index=0):
        self.n_step = start_index
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.net_worth_history = [self.initial_balance]
        return self._get_observation()

    def _get_observation(self):
        # Acceso directo a arrays de Numpy (C++ speed)
        obs = self.obs_data[self.n_step]  # Features técnicos
        rf_pred = self.rf_preds[self.n_step]  # Predicción pre-calculada

        has_shares = 1.0 if self.shares_held > 0 else 0.0

        extra_info = np.array([rf_pred, has_shares])

        state = np.concatenate((obs, extra_info))
        return torch.FloatTensor(state)

    def step(self, action):
        current_price = self.prices[self.n_step]

        if action == 1:  # Comprar
            if self.balance >= current_price:
                shares_to_buy = self.balance // current_price
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy

        elif action == 2:  # Vender
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0

        self.n_step += 1

        next_price = self.prices[self.n_step]
        self.net_worth = self.balance + (self.shares_held * next_price)
        self.net_worth_history.append(self.net_worth)

        prev_net_worth = self.net_worth_history[-2]
        reward = (self.net_worth - prev_net_worth) / prev_net_worth
        reward = reward * 100

        done = self.n_step >= (self.max_steps - 1)

        next_state = (
            self._get_observation()
            if not done
            else torch.zeros_like(self._get_observation())
        )

        return next_state, reward, done


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=2000)

        self.epsilon = 1.0  # Exploración inicial
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Factor de descuento
        self.batch_size = 32

    def act(self, state, is_training=True):
        if is_training and np.random.rand() <= self.epsilon:
            return random.randrange(3)  # Acción aleatoria
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)

        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (
                    reward
                    + self.gamma * torch.max(self.target_model(next_state)).item()
                )

            target_f = self.model(state)

            # Actualizamos solo el Q-value de la acción tomada
            # Convertimos action a tensor para indexar si fuera necesario, aqui lo hacemos manual:
            target_vector = target_f.tolist()
            target_vector[action] = target

            # Recalculamos loss y optimizamos
            prediction = self.model(state)
            target_tensor = torch.FloatTensor(target_vector)

            loss = self.loss_fn(prediction, target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / self.batch_size

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename="modelo_dqn.pth"):
        torch.save(self.model.state_dict(), filename)
        print(f"Modelo guardado exitosamente en: {filename}")

    def load(self, filename="modelo_dqn.pth"):
        # map_location='cpu' asegura que funcione aunque no tengas GPU
        self.model.load_state_dict(
            torch.load(filename, map_location=torch.device("cpu"))
        )
        self.model.eval()  # Pone el modelo en modo 'evaluación' (fija los pesos)
        print(f"Modelo cargado desde: {filename}")


# --- CONFIGURACIÓN DE ENTRENAMIENTO ---
features_rf = ["Return_1d", "Return_5d", "Dist_SMA_10", "Volatility", "RSI"]
# Estado = Features RF + Prediccion_RF + Has_Shares = 5 + 1 + 1 = 7 inputs
state_dim = len(features_rf) + 2
action_dim = 3  # Hold, Buy, Sell

agent = Agent(state_dim, action_dim)
episodes = 200  # Aumentar para mejor resultado (ej. 200)

print(f"\n--- Iniciando Entrenamiento DQN (Episodios: {episodes}) ---")
loss_history = []

for e in range(episodes):
    print("Episode: ", e)
    # Elegimos una acción aleatoria para entrenar en este episodio (Generalización)
    ticker_train = random.choice(list(market_data_opt.keys()))
    df_train = market_data_opt[ticker_train]  # Usamos el diccionario optimizado

    df_train_cut = df_train[df_train.index <= FECHA_CORTE]
    if len(df_train_cut) < 50:
        continue

    # YA NO PASAMOS EL MODELO RF, SOLO LOS DATOS
    env = TradingEnvFast(df_train_cut, features_rf)
    state = env.reset()

    total_reward = 0
    done = False

    with tqdm(
        total=env.max_steps,
        desc=f"Episodio {e + 1}/{episodes} ({ticker_train})",
        unit="días",
    ) as pbar:
        while not done:
            action = agent.act(state, is_training=True)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.replay()

            pbar.update(1)

            # Muestra el Profit acumulado y el Epsilon actual
            pbar.set_postfix(
                {"Profit": f"{total_reward:.1f}%", "Eps": f"{agent.epsilon:.2f}"}
            )

    if e % 5 == 0:
        agent.update_target_network()
        print(
            f"Episodio {e}/{episodes} | Ticker: {ticker_train} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}"
        )


print("\nEntrenamiento finalizado.")
agent.save("modelo_trader.pth")
