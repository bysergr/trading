import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import joblib  # Necesario para cargar el Scaler
import os


# --- 1. DEFINICIÓN DE ARQUITECTURA (IDÉNTICA al entrenamiento) ---
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
        # En testeo, no usamos Epsilon (siempre la mejor decisión = argmax)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def load(self, filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Features Técnicos
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["Dist_SMA_10"] = df["Close"] / df["SMA_10"]
    df["Volatility"] = df["Close"].rolling(window=10).std()
    df["RSI"] = calcular_rsi(df["Close"])

    # Objetivo para entrenar el RF auxiliar
    df["Target_Return"] = df["Close"].pct_change().shift(-1)
    df = df.dropna()
    return df


# --- 3. CONFIGURACIÓN ---
TICKER = "ECOPETROL.CL"  # ¡Prueba con diferentes!
MODELO_FILE = "modelo_dqn_v150.pth"  # Usa la versión que prefieras (v50, v100, etc.)
SCALER_FILE = "scaler_trader.pkl"  # INDISPENSABLE
FECHA_INICIO_TEST = "2024-01-01"  # Fecha desde donde empieza a simular

# --- 4. EJECUCIÓN PRINCIPAL ---

# Verificación de archivos
if not os.path.exists(MODELO_FILE) or not os.path.exists(SCALER_FILE):
    print("❌ ERROR: Faltan archivos.")
    print(
        f"Buscando Modelo: {MODELO_FILE} -> {'Encontrado' if os.path.exists(MODELO_FILE) else 'FALTA'}"
    )
    print(
        f"Buscando Scaler: {SCALER_FILE} -> {'Encontrado' if os.path.exists(SCALER_FILE) else 'FALTA'}"
    )
    exit()

# Cargar Scaler
print("Loading Scaler...")
scaler = joblib.load(SCALER_FILE)

# A. Preparar datos
df = preparar_datos_test(TICKER)
features = ["Return_1d", "Return_5d", "Dist_SMA_10", "Volatility", "RSI"]

# B. Generar Señal Random Forest (Simulación del "Oráculo")
print("🌲 Generando señales del Random Forest auxiliar...")
# Entrenamos solo con datos ANTERIORES al test para no hacer trampa
mask_train = df.index < FECHA_INICIO_TEST
mask_test = df.index >= FECHA_INICIO_TEST

X_train_rf = df.loc[mask_train, features]
y_train_rf = df.loc[mask_train, "Target_Return"]

if len(X_train_rf) < 50:
    print(
        "⚠️ Advertencia: Pocos datos para entrenar el RF. El resultado puede ser inestable."
    )

rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Predecimos sobre el conjunto de TEST
df_test = df.loc[mask_test].copy()
if len(df_test) == 0:
    print(
        f"❌ Error: La fecha de inicio {FECHA_INICIO_TEST} es posterior a los datos disponibles."
    )
    exit()

df_test["RF_Prediction"] = rf_model.predict(df_test[features])

# C. Inicializar Agente y Cargar Modelo
state_dim = len(features) + 2  # Features + RF_Pred + Has_Shares
agent = Agent(state_dim, 3)
agent.load(MODELO_FILE)

# D. Loop de Simulación (Backtesting)
saldo = 10_000_000  # 10 Millones COP
acciones = 0
historial_valor = []
log_operaciones = []
puntos_compra = []
puntos_venta = []

print(f"\n🚀 Iniciando Test en {TICKER} desde {FECHA_INICIO_TEST}...")

for i in range(len(df_test)):
    row = df_test.iloc[i]

    # 1. Obtener datos crudos
    raw_obs = row[features].values.reshape(1, -1)

    # 2. NORMALIZACIÓN CORRECTA (Usando el Scaler cargado)
    obs_norm = scaler.transform(raw_obs)[0]

    # 3. Preparar resto de inputs
    rf_pred = row["RF_Prediction"]
    # IMPORTANTE: Usar el mismo multiplicador que en el entrenamiento (100.0)
    rf_pred_scaled = rf_pred * 100.0

    has_shares = 1.0 if acciones > 0 else 0.0

    # 4. Unir todo
    state_input = np.concatenate((obs_norm, [rf_pred_scaled, has_shares]))
    state_tensor = torch.FloatTensor(state_input)

    # 5. El agente decide
    action = agent.act(state_tensor)

    # Debug: Ver qué piensa la red neuronal
    with torch.no_grad():
        q_values = agent.model(state_tensor)

    # Mostrar detalle cada 30 días
    if i % 30 == 0:
        print(
            f"Día {i} | Q-Values: [Hold: {q_values[0]:.2f}, Buy: {q_values[1]:.2f}, Sell: {q_values[2]:.2f}] -> Acción: {action}"
        )

    precio_hoy = row["Close"]
    fecha = df_test.index[i]

    # Ejecutar Acción
    if action == 1:  # COMPRAR
        if saldo >= precio_hoy:
            cantidad = saldo // precio_hoy
            costo = cantidad * precio_hoy
            saldo -= costo
            acciones += cantidad
            puntos_compra.append((fecha, precio_hoy))
            log_operaciones.append(f"{fecha.date()} | COMPRA | ${precio_hoy:.0f}")

    elif action == 2:  # VENDER
        if acciones > 0:
            ingreso = acciones * precio_hoy
            saldo += ingreso
            acciones = 0
            puntos_venta.append((fecha, precio_hoy))
            log_operaciones.append(f"{fecha.date()} | VENTA  | ${precio_hoy:.0f}")

    # Valor total portafolio
    valor_total = saldo + (acciones * precio_hoy)
    historial_valor.append(valor_total)

# --- 5. RESULTADOS Y VISUALIZACIÓN ---

retorno_total = ((historial_valor[-1] - 10_000_000) / 10_000_000) * 100
bh_retorno = (
    (df_test["Close"].iloc[-1] - df_test["Close"].iloc[0]) / df_test["Close"].iloc[0]
) * 100

print("\n" + "=" * 40)
print(f"RESULTADO FINAL ({TICKER})")
print("=" * 40)
print("Saldo Inicial: $10,000,000")
print(f"Saldo Final:   ${historial_valor[-1]:,.2f}")
print(f"Rentabilidad IA:      {retorno_total:.2f}%")
print(f"Rentabilidad Mercado: {bh_retorno:.2f}%")
print("=" * 40)

# Gráfica
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Gráfico 1: Precio y Señales
ax1.plot(df_test.index, df_test["Close"], label="Precio", color="gray", alpha=0.5)
if puntos_compra:
    idx_b, prices_b = zip(*puntos_compra)
    ax1.scatter(
        idx_b, prices_b, marker="^", color="green", s=100, label="Compra", zorder=5
    )
if puntos_venta:
    idx_s, prices_s = zip(*puntos_venta)
    ax1.scatter(
        idx_s, prices_s, marker="v", color="red", s=100, label="Venta", zorder=5
    )
ax1.set_title(f"{TICKER} - Operaciones del Agente")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Evolución del Portafolio
ax2.plot(
    df_test.index, historial_valor, label="IA Portafolio", color="blue", linewidth=2
)
factor_norm = 10_000_000 / df_test["Close"].iloc[0]
ax2.plot(
    df_test.index,
    df_test["Close"] * factor_norm,
    label="Buy & Hold",
    color="orange",
    linestyle="--",
    alpha=0.7,
)

ax2.set_title(f"Rentabilidad Acumulada: {retorno_total:.2f}%")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
