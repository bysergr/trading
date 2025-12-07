import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


# --- 1. DEFINICIÓN DE ARQUITECTURA ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        return self.fc4(x)


class Agent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)

    def act(self, state):
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def load(self, filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # map_location asegura que cargue en CPU si no tienes GPU a mano
        self.model.load_state_dict(torch.load(filename, map_location=device))
        self.model.eval()
        print(f"✅ Modelo cargado exitosamente: {filename}")


# --- 2. FUNCIONES DE PROCESAMIENTO ---
def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def preparar_datos_test(ticker):
    print(f"📥 Descargando datos para: {ticker}...")
    df = yf.download(
        ticker, period="5y", interval="1d", auto_adjust=True, progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if len(df) == 0:
        raise ValueError(f"No se encontraron datos para {ticker}")

    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["Dist_SMA_10"] = df["Close"] / df["SMA_10"]
    df["Volatility"] = df["Close"].rolling(window=10).std()
    df["RSI"] = calcular_rsi(df["Close"])
    df["Target_Return"] = df["Close"].pct_change().shift(-1)
    df = df.dropna()
    return df


# --- 3. CONFIGURACIÓN ---
TICKER = "ECOPETROL.CL"
MODELO_FILE = "modelo_dqn_v200.pth"  # Asegúrate que coincida con el nombre del archivo
SCALER_FILE = "scaler_trader.pkl"
FECHA_INICIO_TEST = "2024-01-01"

# --- 4. EJECUCIÓN PRINCIPAL ---
if not os.path.exists(MODELO_FILE) or not os.path.exists(SCALER_FILE):
    print(
        f"❌ ERROR: Faltan archivos. Modelo: {os.path.exists(MODELO_FILE)}, Scaler: {os.path.exists(SCALER_FILE)}"
    )
    exit()

print("Loading Scaler...")
scaler = joblib.load(SCALER_FILE)

# A. Preparar datos
df = preparar_datos_test(TICKER)
features = ["Return_1d", "Return_5d", "Dist_SMA_10", "Volatility", "RSI"]

# B. Generar Señal Random Forest
print("🌲 Generando señales del Random Forest auxiliar...")
mask_train = df.index < FECHA_INICIO_TEST
mask_test = df.index >= FECHA_INICIO_TEST
X_train_rf = df.loc[mask_train, features]
y_train_rf = df.loc[mask_train, "Target_Return"]

rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

df_test = df.loc[mask_test].copy()
if len(df_test) == 0:
    print("❌ Error: Sin datos de test.")
    exit()
df_test["RF_Prediction"] = rf_model.predict(df_test[features])

# C. Inicializar Agente
# IMPORTANTE: Ahora el state_dim es +3 (Features + RF + Shares + PnL)
state_dim = len(features) + 3
agent = Agent(state_dim, 3)
agent.load(MODELO_FILE)

# D. Loop de Simulación
saldo = 10_000_000
acciones = 0
avg_buy_price = 0.0  # <--- NUEVA VARIABLE DE RASTREO
historial_valor = []
log_operaciones = []
puntos_compra = []
puntos_venta = []

print(f"\n🚀 Iniciando Test en {TICKER} desde {FECHA_INICIO_TEST}...")

for i in range(len(df_test)):
    row = df_test.iloc[i]
    precio_hoy = row["Close"]

    # --- CONSTRUCCIÓN DEL ESTADO (NUEVA LÓGICA) ---

    # 1. Normalización técnica
    raw_obs = row[features].values.reshape(1, -1)
    obs_norm = scaler.transform(raw_obs)[0]

    # 2. Datos extra
    rf_pred_scaled = row["RF_Prediction"] * 100.0
    has_shares = 1.0 if acciones > 0 else 0.0

    # 3. Cálculo de PnL no realizado (Lo que ve el agente)
    unrealized_pnl = 0.0
    if acciones > 0 and avg_buy_price > 0:
        unrealized_pnl = (precio_hoy - avg_buy_price) / avg_buy_price

    # 4. Unir todo: [Features, RF, Shares, PnL]
    state_input = np.concatenate(
        (obs_norm, [rf_pred_scaled, has_shares, unrealized_pnl])
    )
    state_tensor = torch.FloatTensor(state_input)

    # 5. Decisión
    action = agent.act(state_tensor)

    # Debug visual
    if i % 30 == 0:
        with torch.no_grad():
            q = agent.model(state_tensor)
        print(
            f"Día {i} | PnL: {unrealized_pnl * 100:.2f}% | Q: {q.numpy()} -> Act: {action}"
        )

    fecha = df_test.index[i]

    # --- EJECUCIÓN (CON RASTREO DE PRECIO PROMEDIO) ---
    if action == 1:  # COMPRAR
        if saldo >= precio_hoy:
            cantidad = saldo // precio_hoy
            if cantidad > 0:
                costo_total = cantidad * precio_hoy

                # Actualizar precio promedio ponderado
                valor_actual_holding = acciones * avg_buy_price
                nuevo_valor_holding = valor_actual_holding + costo_total
                acciones += cantidad
                avg_buy_price = nuevo_valor_holding / acciones

                saldo -= costo_total
                puntos_compra.append((fecha, precio_hoy))
                log_operaciones.append(f"{fecha.date()} | COMPRA | ${precio_hoy:.0f}")

    elif action == 2:  # VENDER
        if acciones > 0:
            ingreso = acciones * precio_hoy

            # Calcular ganancia real de esta operación para log
            pnl_operacion = (precio_hoy - avg_buy_price) / avg_buy_price
            log_operaciones.append(
                f"{fecha.date()} | VENTA  | ${precio_hoy:.0f} | Profit: {pnl_operacion * 100:.2f}%"
            )

            saldo += ingreso
            acciones = 0
            avg_buy_price = 0.0  # Reset
            puntos_venta.append((fecha, precio_hoy))

    valor_total = saldo + (acciones * precio_hoy)
    historial_valor.append(valor_total)

# --- 5. RESULTADOS ---
retorno_total = ((historial_valor[-1] - 10_000_000) / 10_000_000) * 100
bh_retorno = (
    (df_test["Close"].iloc[-1] - df_test["Close"].iloc[0]) / df_test["Close"].iloc[0]
) * 100

print("\n" + "=" * 40)
print(f"RESULTADO FINAL ({TICKER})")
print("=" * 40)
print(f"Saldo Final:   ${historial_valor[-1]:,.2f}")
print(f"Rentabilidad IA:      {retorno_total:.2f}%")
print(f"Rentabilidad Mercado: {bh_retorno:.2f}%")
print("=" * 40)

# Gráficas
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax1.plot(df_test.index, df_test["Close"], label="Precio", color="gray", alpha=0.5)
if puntos_compra:
    idx, val = zip(*puntos_compra)
    ax1.scatter(idx, val, marker="^", color="green", s=100, label="Compra")
if puntos_venta:
    idx, val = zip(*puntos_venta)
    ax1.scatter(idx, val, marker="v", color="red", s=100, label="Venta")
ax1.set_title("Operaciones")
ax1.legend()

ax2.plot(df_test.index, historial_valor, label="IA", color="blue")
factor = 10_000_000 / df_test["Close"].iloc[0]
ax2.plot(
    df_test.index,
    df_test["Close"] * factor,
    label="Buy&Hold",
    color="orange",
    linestyle="--",
)
ax2.set_title("Curva de Capital")
ax2.legend()
plt.tight_layout()
plt.show()
