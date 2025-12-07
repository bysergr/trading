import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import tqdm
import requests
import joblib


# --- CONFIGURACIÓN ---
tickers_bvc = ["ECOPETROL.CL", "ISA.CL", "GRUPOARGOS.CL", "GEB.CL"]
FECHA_CORTE = "2023-12-31"  # El modelo NO verá nada después de esta fecha para entrenar
features_rf = ["Return_1d", "Return_5d", "Dist_SMA_10", "Volatility", "RSI"]
# Estado = Features RF + Prediccion_RF + Has_Shares = 5 + 1 + 1 = 7 inputs
state_dim = len(features_rf) + 2
action_dim = 3  # Hold, Buy, Sell

tries = [50, 100]


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

        # Máscaras booleanas para separar el tiempo
        mask_train = df_processed.index <= FECHA_CORTE

        X_train = df_processed.loc[mask_train, features_rf]
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


print("\n--- Calibrando Normalizador (StandardScaler) ---")

scaler = StandardScaler()
all_training_data = []

# 1. Recolectamos TODOS los datos de entrenamiento de todos los tickers
for t in tickers_bvc:
    df = market_data_opt[t]

    # IMPORTANTE: Solo "aprendemos" de los datos anteriores a la fecha de corte
    # para evitar "Data Leakage" (que el modelo vea el futuro)
    df_train_cut = df[df.index <= FECHA_CORTE]

    # Extraemos solo las columnas numéricas que usa la red neuronal
    # features_rf = ["Return_1d", "Return_5d", "Dist_SMA_10", "Volatility", "RSI"]
    if len(df_train_cut) > 0:
        all_training_data.append(df_train_cut[features_rf].values)

# 2. Unimos todo en una matriz gigante
full_matrix = np.concatenate(all_training_data, axis=0)

# 3. El Scaler calcula la Media y Desviación Estándar de todo el mercado
scaler.fit(full_matrix)

print("Scaler calibrado exitosamente.")
print(f"Media de cada feature: {scaler.mean_}")
print(f"Varianza de cada feature: {scaler.var_}")

# Guardamos el scaler para usarlo luego en producción (Telegram/Live Trading)
joblib.dump(scaler, "scaler_trader.pkl")

print(
    "Datos optimizados. El entorno RL ahora solo leerá columnas, no ejecutará modelos."
)


class TradingEnvFast:
    def __init__(self, df, features_list, scaler, initial_balance=10_000_000):
        self.df = df
        self.features = features_list
        self.scaler = scaler
        self.initial_balance = initial_balance

        self.n_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.net_worth_history = [self.initial_balance]

        # Datos en numpy para velocidad
        self.obs_data = self.df[self.features].values
        self.rf_preds = self.df["RF_Prediction"].values
        self.prices = self.df["Close"].values
        self.max_steps = len(df) - 1

    def reset(self, start_index=0):
        self.n_step = start_index
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.net_worth_history = [self.initial_balance]
        return self._get_observation()

    def _get_observation(self):
        # 1. Obtenemos los features crudos (raw)
        raw_obs = self.obs_data[self.n_step].copy()

        # 2. --- NORMALIZACIÓN PROFESIONAL ---
        # El scaler espera una matriz 2D (1 fila, N columnas), por eso el reshape(1, -1)
        # transform devuelve una matriz, así que tomamos el índice [0] para volver a tener un vector
        normalized_obs = self.scaler.transform(raw_obs.reshape(1, -1))[0]

        # 3. Obtenemos la predicción del RF
        rf_pred = self.rf_preds[self.n_step]

        # La predicción del RF suele ser un % pequeño (ej: 0.01).
        # Podemos multiplicarlo por 10 o 100 para que tenga una escala similar a la normalización (que suele ir de -3 a 3)
        rf_pred_scaled = rf_pred * 100.0

        has_shares = 1.0 if self.shares_held > 0 else 0.0

        # Concatenamos: [Features Normalizados] + [Prediccion RF] + [Tengo Acciones?]
        extra_info = np.array([rf_pred_scaled, has_shares])

        state = np.concatenate((normalized_obs, extra_info))

        return torch.FloatTensor(state)

    def step(self, action):
        current_price = self.prices[self.n_step]

        # Ejecutar Acción
        if action == 1:  # BUY
            if self.balance >= current_price:
                # Comprar todo lo posible (Simplificación agresiva para forzar movimientos)
                shares_to_buy = self.balance // current_price
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy

        elif action == 2:  # SELL
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0

        # Avanzar el tiempo
        self.n_step += 1
        next_price = self.prices[self.n_step]

        # Calcular Patrimonio Nuevo
        self.net_worth = self.balance + (self.shares_held * next_price)
        self.net_worth_history.append(self.net_worth)

        # --- CÁLCULO DE RECOMPENSA (ALPHA) ---

        # 1. Retorno del Agente
        prev_net_worth = self.net_worth_history[-2]
        agent_return = (self.net_worth - prev_net_worth) / prev_net_worth

        # 2. Retorno del Mercado (Benchmark)
        market_return = (next_price - current_price) / current_price

        # 3. Alpha (Diferencia)
        # Multiplicamos por 100 para que los gradientes de la red neuronal sean significativos
        reward = (agent_return - market_return) * 100

        # 3. Penalización por inactividad extrema
        # Si lleva mucho tiempo en cash y el mercado sube, castigamos más fuerte
        if self.shares_held == 0 and market_return > 0:
            reward -= 0.2

        if agent_return < 0:
            reward -= 1.0

        done = self.n_step >= (self.max_steps - 1)
        next_state = (
            self._get_observation()
            if not done
            else torch.zeros_like(self._get_observation())
        )

        return next_state, reward, done


# --- CLASES DEL AGENTE Y ENTORNO ---


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        return self.fc4(x)


class Agent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=2000)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98
        self.gamma = 0.95
        self.batch_size = 32

    def act(self, state, is_training=True):
        if is_training and np.random.rand() <= self.epsilon:
            return random.randrange(3)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)

        # Vectorización (Mucho más rápido)
        states = torch.cat([s.unsqueeze(0) for s, a, r, n, d in minibatch])
        actions = torch.tensor([a for s, a, r, n, d in minibatch]).unsqueeze(1)
        rewards = torch.tensor([r for s, a, r, n, d in minibatch], dtype=torch.float32)
        next_states = torch.cat([n.unsqueeze(0) for s, a, r, n, d in minibatch])
        dones = torch.tensor([d for s, a, r, n, d in minibatch], dtype=torch.float32)

        q_values = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)


# --- LÓGICA DE ENTRENAMIENTO INCREMENTAL ---

# 1. Definimos las metas acumulativas
# El modelo entrenará hasta llegar al 50, parará, enviará, y seguirá hasta el 100, etc.
metas_episodios = [50, 100, 150, 200]

# 2. Inicializamos el agente UNA SOLA VEZ fuera del bucle
print("\n--- Inicializando Agente (Cerebro Nuevo) ---")
agent = Agent(state_dim, action_dim)

# Variable para llevar la cuenta de dónde vamos
episodios_completados = 0


# Función de Telegram
def enviar_a_telegram(archivo_path, token, chat_id, descripcion):
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    print(f"📤 Subiendo {archivo_path} a Telegram...")
    try:
        with open(archivo_path, "rb") as f:
            files = {"document": f}
            data = {"chat_id": chat_id, "caption": descripcion}
            requests.post(url, files=files, data=data)
            print("✅ Enviado.")
    except Exception as e:
        print(f"❌ Error al enviar: {e}")


# Credenciales (Pon las tuyas aquí)
TOKEN = "8208409663:AAFgU_1DsRBan3lpBu4YcGx_50uqx0GiSEo"
CHAT_ID = "6912858224"

# Primero enviamos el Scaler una sola vez (es necesario para usar cualquier versión del modelo)
try:
    enviar_a_telegram(
        "scaler_trader.pkl",
        TOKEN,
        CHAT_ID,
        "👓 Gafas (Scaler) - Único para todas las versiones",
    )
except Exception as e:
    print(f"Error al enviar el Scaler: {e}")
    print("Asegurate de haber ejecutado la parte del Scaler primero.")


# 3. Bucle de Metas
for meta in metas_episodios:
    # Calculamos cuántos faltan para la siguiente meta
    episodios_a_entrenar = meta - episodios_completados

    if episodios_a_entrenar <= 0:
        continue  # Ya pasamos esa meta

    print(
        f"\n>>> Entrenando desde episodio {episodios_completados} hasta {meta} ({episodios_a_entrenar} nuevos) <<<"
    )

    for e in range(episodios_a_entrenar):
        num_episodio_global = episodios_completados + e + 1

        # Selección de datos
        ticker_train = random.choice(list(market_data_opt.keys()))
        df_train = market_data_opt[ticker_train]
        df_train_cut = df_train[df_train.index <= FECHA_CORTE]

        if len(df_train_cut) < 50:
            continue

        # Creamos entorno con el Scaler
        env = TradingEnvFast(df_train_cut, features_rf, scaler)
        state = env.reset()

        total_reward = 0
        done = False
        action_counts = {0: 0, 1: 0, 2: 0}

        with tqdm(
            total=env.max_steps,
            desc=f"Episodio Global {num_episodio_global}/{meta}",
            unit="días",
        ) as pbar:
            while not done:
                action = agent.act(state, is_training=True)
                action_counts[action] += 1
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                agent.replay()
                pbar.update(1)
                pbar.set_postfix(
                    {"Profit": f"{total_reward:.1f}%", "Eps": f"{agent.epsilon:.2f}"}
                )

        agent.decay_epsilon()

        # Actualizar red objetivo cada 5 episodios
        if num_episodio_global % 5 == 0:
            agent.update_target_network()

    # --- AL FINALIZAR LA META ---
    episodios_completados = meta  # Actualizamos el contador

    # Guardamos esta versión
    nombre_modelo = f"modelo_dqn_v{meta}.pth"
    agent.save(nombre_modelo)

    # Enviamos a Telegram
    mensaje = f"🚀 Checkpoint alcanzado: {meta} Episodios.\nEpsilon actual: {agent.epsilon:.3f}"
    enviar_a_telegram(nombre_modelo, TOKEN, CHAT_ID, mensaje)

print("\n🏁 Entrenamiento completo de todas las fases.")
