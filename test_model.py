import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor


# --- 1. DEFINICIÓN DE ARQUITECTURA (Debe ser IDÉNTICA al entrenamiento) ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Aumentamos neuronas a 128 para darle más capacidad
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)

        # Usamos LeakyReLU en lugar de ReLU normal para evitar muerte neuronal
        self.leaky_relu = nn.LeakyReLU(0.01)

        # Inicialización de pesos (Evita que empiece con sesgos tontos)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Inicialización He (Kaiming) optimizada para LeakyReLU
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

    def act(self, state):
        # En testeo, no usamos Epsilon (siempre la mejor decisión)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def load(self, filename):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(filename))
        else:
            self.model.load_state_dict(
                torch.load(filename, map_location=torch.device("cpu"))
            )
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
TICKER = "GRUPOARGOS.CL"  # Cambia esto por la acción que quieras probar
MODELO_FILE = "modelo_trader_alpha_50.pth"
FECHA_INICIO_TEST = "2024-01-01"  # Probaremos desde 2024 hasta hoy (o 2025)

# --- 4. EJECUCIÓN PRINCIPAL ---

# A. Preparar datos
df = preparar_datos_test(TICKER)
features = ["Return_1d", "Return_5d", "Dist_SMA_10", "Volatility", "RSI"]

# B. Generar Señal Random Forest (Simulación del "Oráculo")
# Entrenamos un RF rápido con datos PREVIOS a la fecha de test para no hacer trampa
print("🌲 Generando señales del Random Forest auxiliar...")
mask_train = df.index < FECHA_INICIO_TEST
mask_test = df.index >= FECHA_INICIO_TEST

X_train_rf = df.loc[mask_train, features]
y_train_rf = df.loc[mask_train, "Target_Return"]

rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Predecimos sobre el conjunto de TEST
df_test = df.loc[mask_test].copy()
if len(df_test) == 0:
    raise ValueError(f"No hay datos para la fecha {FECHA_INICIO_TEST}")

df_test["RF_Prediction"] = rf_model.predict(df_test[features])

# C. Inicializar Agente y Cargar Modelo
state_dim = len(features) + 2  # Features + RF_Pred + Has_Shares
agent = Agent(state_dim, 3)

try:
    agent.load(MODELO_FILE)
except FileNotFoundError:
    print(
        f"❌ Error: No encuentro el archivo '{MODELO_FILE}'. Asegúrate de haber entrenado primero."
    )
    exit()

# D. Loop de Simulación (Backtesting)
saldo = 10_000_000  # 10 Millones COP
acciones = 0
historial_valor = []
log_operaciones = []
puntos_compra = []
puntos_venta = []

print(f"\n🚀 Iniciando Test en {TICKER} desde {FECHA_INICIO_TEST}...")

for i in range(len(df_test)):
    # Construir estado
    row = df_test.iloc[i]
    obs_tech = row[features].values.astype(float)
    rf_pred = row["RF_Prediction"]
    has_shares = 1.0 if acciones > 0 else 0.0

    obs_norm = obs_tech.copy()

    # 2. Aplicamos las mismas reglas matemáticas
    # Indices asumidos: [Return_1d, Return_5d, Dist_SMA_10, Volatility, RSI]

    obs_norm[2] = obs_norm[2] - 1.0  # Dist_SMA: Centrar en 0
    obs_norm[3] = obs_norm[3] / 1000.0  # Volatility: Escalar
    obs_norm[4] = obs_norm[4] / 100.0  # RSI: Pasar de 0-100 a 0-1

    rf_pred_norm = rf_pred * 10.0

    state_input = np.concatenate((obs_norm, [rf_pred_norm, has_shares]))

    if i % 50 == 0:
        print(f"DEBUG INPUT DIA {i}: {state_input}")

    state_tensor = torch.FloatTensor(state_input)

    # El agente decide
    action = agent.act(state_tensor)

    with torch.no_grad():
        q_values = agent.model(state_tensor)

    action = torch.argmax(q_values).item()

    # SOLO PARA DEBUG: Imprimir los valores cada 20 días
    if i % 20 == 0:
        print(
            f"Día {i} | Precios: {row['Close']:.0f} | Q-Values: HOLD={q_values[0]:.4f}, BUY={q_values[1]:.4f}, SELL={q_values[2]:.4f} -> Acción: {action}"
        )

    precio_hoy = row["Close"]
    fecha = df_test.index[i]

    # Ejecutar Acción
    tipo_accion = "MANTENER"

    if action == 1:  # COMPRAR
        if saldo >= precio_hoy:
            cantidad = saldo // precio_hoy
            costo = cantidad * precio_hoy
            saldo -= costo
            acciones += cantidad
            tipo_accion = "COMPRA"
            puntos_compra.append((fecha, precio_hoy))
            log_operaciones.append(
                f"{fecha.date()} | COMPRA | Precio: ${precio_hoy:.2f} | Cant: {cantidad}"
            )

    elif action == 2:  # VENDER
        if acciones > 0:
            ingreso = acciones * precio_hoy
            saldo += ingreso
            acciones = 0
            tipo_accion = "VENTA"
            puntos_venta.append((fecha, precio_hoy))
            log_operaciones.append(
                f"{fecha.date()} | VENTA  | Precio: ${precio_hoy:.2f} | Total: ${ingreso:.2f}"
            )

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
print(f"Saldo Inicial: $10,000,000")
print(f"Saldo Final:   ${historial_valor[-1]:,.2f}")
print(f"Rentabilidad Modelo: {retorno_total:.2f}%")
print(f"Rentabilidad Mercado (Buy&Hold): {bh_retorno:.2f}%")
print("=" * 40)

# Mostrar últimos 5 logs
print("\n--- Últimas 5 Operaciones ---")
for log in log_operaciones[-5:]:
    print(log)

# Gráfica
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Gráfico 1: Precio y Señales
ax1.plot(
    df_test.index, df_test["Close"], label="Precio Acción", color="gray", alpha=0.5
)
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
ax1.set_title(f"{TICKER} - Decisiones de Compra/Venta")
ax1.set_ylabel("Precio (COP)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Evolución del Portafolio
ax2.plot(
    df_test.index,
    historial_valor,
    label="Mi Portafolio (IA)",
    color="blue",
    linewidth=2,
)
# Comparativa con Buy & Hold (normalizado al capital inicial)
factor_norm = 10_000_000 / df_test["Close"].iloc[0]
ax2.plot(
    df_test.index,
    df_test["Close"] * factor_norm,
    label="Mercado (Buy & Hold)",
    color="orange",
    linestyle="--",
    alpha=0.7,
)

ax2.set_title(f"Evolución del Capital (Rentabilidad: {retorno_total:.2f}%)")
ax2.set_ylabel("Valor Portafolio (COP)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
