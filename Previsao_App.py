"""
============================================================================
APLICAÇÃO STREAMLIT - PREVISÃO DE SÉRIES TEMPORAIS COM VARIÁVEIS EXÓGENAS
============================================================================

Aplicação para carregar modelos de previsão (SARIMAX, LSTM, GRU, LightGBM,
Híbridos, etc.), gerar previsões para 6 meses com 4 a 6 variáveis exógenas,
avaliar métricas de desempenho e analisar resíduos do modelo.

Estrutura de pastas esperada (ajustável na constante MODELS_DIR):

    modelos/
        gru_model.keras
        gru_model_residuals.csv
        sarimax_model.pkl
        sarimax_model_residuals.csv
        lightgbm_model.joblib
        lightgbm_model_residuals.csv

O ficheiro de resíduos deve conter, no mínimo, as colunas:
    data, y_real, y_previsto, residuo

Autor: Estrutura gerada para fins de TCC / produção.
============================================================================
"""

import os
import io
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------------------------------------------------------------
# Importação condicional do Keras (apenas necessária para modelos .keras)
# ----------------------------------------------------------------------------
try:
    from tensorflow.keras.models import load_model as keras_load_model
    KERAS_DISPONIVEL = True
except ImportError:
    KERAS_DISPONIVEL = False

# ----------------------------------------------------------------------------
# Importação condicional do joblib (necessário para .joblib)
# ----------------------------------------------------------------------------
try:
    import joblib
    JOBLIB_DISPONIVEL = True
except ImportError:
    JOBLIB_DISPONIVEL = False


# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

MODELS_DIR = "modelos"          # Pasta onde ficam os modelos e os _residuals.csv
HORIZONTE_MESES = 6              # Horizonte fixo de previsão
EXTENSOES_VALIDAS = (".pkl", ".joblib", ".keras")

st.set_page_config(
    page_title="Previsão de Séries Temporais",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# FUNÇÕES DE MÉTRICAS
# ============================================================================

def smape(y_real: np.ndarray, y_previsto: np.ndarray) -> float:
    """
    Calcula o sMAPE (Symmetric Mean Absolute Percentage Error).

    O sMAPE não existe nativamente no scikit-learn, por isso é implementado
    manualmente. Fórmula:

        sMAPE = (100 / n) * soma( |real - previsto| / ((|real| + |previsto|) / 2) )

    Pontos onde o denominador é zero são ignorados para evitar divisão por zero.
    """
    y_real = np.asarray(y_real, dtype=float)
    y_previsto = np.asarray(y_previsto, dtype=float)

    denominador = (np.abs(y_real) + np.abs(y_previsto)) / 2.0
    mascara = denominador != 0

    if not mascara.any():
        return float("nan")

    erro = np.abs(y_real[mascara] - y_previsto[mascara]) / denominador[mascara]
    return float(np.mean(erro) * 100)


def mape(y_real: np.ndarray, y_previsto: np.ndarray) -> float:
    """
    Calcula o MAPE (Mean Absolute Percentage Error), ignorando valores
    reais iguais a zero para evitar divisão por zero.
    """
    y_real = np.asarray(y_real, dtype=float)
    y_previsto = np.asarray(y_previsto, dtype=float)

    mascara = y_real != 0
    if not mascara.any():
        return float("nan")

    erro = np.abs((y_real[mascara] - y_previsto[mascara]) / y_real[mascara])
    return float(np.mean(erro) * 100)


def calcular_metricas(y_real: np.ndarray, y_previsto: np.ndarray) -> dict:
    """Calcula RMSE, MAE, MAPE e sMAPE para um par (real, previsto)."""
    rmse = float(np.sqrt(mean_squared_error(y_real, y_previsto)))
    mae = float(mean_absolute_error(y_real, y_previsto))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape(y_real, y_previsto),
        "sMAPE (%)": smape(y_real, y_previsto),
    }


# ============================================================================
# FUNÇÕES DE CARREGAMENTO (CACHEADAS)
# ============================================================================

@st.cache_resource(show_spinner="A carregar o modelo selecionado...")
def carregar_modelo(caminho_modelo: str):
    """
    Carrega o modelo a partir do disco, de acordo com a extensão do ficheiro.

    Suporta:
        - .pkl      -> pickle padrão (ex.: SARIMAX/statsmodels)
        - .joblib   -> joblib (ex.: LightGBM, modelos híbridos serializados)
        - .keras    -> Keras/TensorFlow (ex.: LSTM, GRU)

    O resultado é cacheado pelo Streamlit (st.cache_resource), pelo que o
    modelo só é lido do disco uma vez por sessão/caminho.
    """
    extensao = os.path.splitext(caminho_modelo)[1].lower()

    if extensao == ".pkl":
        with open(caminho_modelo, "rb") as f:
            return pickle.load(f)

    elif extensao == ".joblib":
        if not JOBLIB_DISPONIVEL:
            raise ImportError("A biblioteca 'joblib' não está instalada.")
        return joblib.load(caminho_modelo)

    elif extensao == ".keras":
        if not KERAS_DISPONIVEL:
            raise ImportError("O TensorFlow/Keras não está instalado.")
        return keras_load_model(caminho_modelo)

    else:
        raise ValueError(f"Extensão de modelo não suportada: {extensao}")


@st.cache_data(show_spinner="A carregar os resíduos do modelo...")
def carregar_residuos(caminho_residuos: str) -> pd.DataFrame:
    """
    Carrega o ficheiro CSV de resíduos correspondente ao modelo.

    Colunas esperadas: data, y_real, y_previsto, residuo
    """
    df = pd.read_csv(caminho_residuos)
    df.columns = [c.strip().lower() for c in df.columns]

    colunas_esperadas = {"data", "y_real", "y_previsto", "residuo"}
    if not colunas_esperadas.issubset(set(df.columns)):
        st.warning(
            f"O ficheiro de resíduos não contém todas as colunas esperadas "
            f"({colunas_esperadas}). Colunas encontradas: {list(df.columns)}"
        )
    else:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        df = df.sort_values("data")

    return df


def listar_modelos_disponiveis(pasta: str) -> list:
    """Lista os ficheiros de modelo válidos (.pkl, .joblib, .keras) na pasta."""
    if not os.path.isdir(pasta):
        return []
    return sorted(
        f for f in os.listdir(pasta)
        if f.lower().endswith(EXTENSOES_VALIDAS)
    )


def caminho_residuos_para_modelo(caminho_modelo: str) -> str:
    """
    Gera o caminho do ficheiro de resíduos seguindo a convenção:
        [nome_do_modelo]_residuals.csv
    """
    pasta = os.path.dirname(caminho_modelo)
    nome_base = os.path.splitext(os.path.basename(caminho_modelo))[0]
    return os.path.join(pasta, f"{nome_base}_residuals.csv")


# ============================================================================
# FUNÇÃO DE PREVISÃO GENÉRICA
# ============================================================================

def gerar_previsao(modelo, exogenas_futuras: pd.DataFrame, ultimo_valor_serie: float = None):
    """
    Gera a previsão para o horizonte definido, tentando detectar
    automaticamente a interface do modelo carregado.

    Suporta três padrões comuns:
        1. statsmodels (SARIMAX): método .get_forecast(steps, exog=...)
        2. scikit-learn / LightGBM: método .predict(X)
        3. Keras (LSTM/GRU): método .predict(X), com X em formato 3D
           (amostras, passos_temporais, variáveis) — aqui assume-se que
           os dados já chegam preparados no shape correto.

    NOTA IMPORTANTE: Modelos Keras (LSTM/GRU) e modelos híbridos exigem,
    em geral, normalização e formatação 3D dos dados antes da previsão
    (a mesma usada no treino). Ajuste esta função conforme o pré-processamento
    real do seu pipeline (ex.: aplicar o mesmo MinMaxScaler/StandardScaler
    salvo durante o treino) antes de a colocar em produção.
    """
    X = exogenas_futuras.values

    # --- Caso 1: modelos statsmodels (SARIMAX) ---
    if hasattr(modelo, "get_forecast"):
        resultado = modelo.get_forecast(steps=len(exogenas_futuras), exog=X)
        previsao = resultado.predicted_mean
        return np.asarray(previsao)

    # --- Caso 2: modelos com .predict() (sklearn, LightGBM, Keras) ---
    elif hasattr(modelo, "predict"):
        previsao = modelo.predict(X)
        previsao = np.asarray(previsao).flatten()
        return previsao

    else:
        raise AttributeError(
            "O modelo carregado não possui método 'predict' nem 'get_forecast'. "
            "Adapte a função gerar_previsao() à interface do seu modelo."
        )


# ============================================================================
# BARRA LATERAL - SELEÇÃO DO MODELO
# ============================================================================

st.sidebar.title("⚙️ Configuração")

modelos_disponiveis = listar_modelos_disponiveis(MODELS_DIR)

if not modelos_disponiveis:
    st.sidebar.error(
        f"Nenhum modelo encontrado na pasta '{MODELS_DIR}/'. "
        f"Coloque ficheiros .pkl, .joblib ou .keras nessa pasta."
    )
    st.stop()

nome_modelo_selecionado = st.sidebar.selectbox(
    "Selecione o modelo de previsão",
    options=modelos_disponiveis,
    help="Modelos suportados: SARIMAX (.pkl), LightGBM/Híbrido (.joblib), LSTM/GRU (.keras)",
)

caminho_modelo_selecionado = os.path.join(MODELS_DIR, nome_modelo_selecionado)
caminho_residuos_selecionado = caminho_residuos_para_modelo(caminho_modelo_selecionado)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Modelo ativo:** `{nome_modelo_selecionado}`")

# Carregamento do modelo (cacheado)
try:
    modelo = carregar_modelo(caminho_modelo_selecionado)
    st.sidebar.success("Modelo carregado com sucesso.")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Carregamento automático dos resíduos (cacheado)
df_residuos = pd.DataFrame()
if os.path.isfile(caminho_residuos_selecionado):
    df_residuos = carregar_residuos(caminho_residuos_selecionado)
else:
    st.sidebar.warning(
        f"Ficheiro de resíduos não encontrado: `{os.path.basename(caminho_residuos_selecionado)}`"
    )


# ============================================================================
# CABEÇALHO
# ============================================================================

st.title("📈 Previsão de Séries Temporais com Variáveis Exógenas")
st.caption(
    "Análise, previsão e diagnóstico de modelos com 4 a 6 variáveis exógenas "
    f"e horizonte fixo de {HORIZONTE_MESES} meses."
)

aba_previsao, aba_metricas, aba_residuos = st.tabs(
    ["🔮 Executar Previsão", "📊 Métricas de Desempenho", "📉 Análise de Resíduos"]
)


# ============================================================================
# ABA 1 - EXECUTAR PREVISÃO
# ============================================================================

with aba_previsao:
    st.subheader("1. Dados históricos da série")

    arquivo_historico = st.file_uploader(
        "Carregue o histórico da série (Excel ou CSV) com colunas: data, valor",
        type=["xlsx", "csv"],
        key="upload_historico",
    )

    df_historico = None
    if arquivo_historico is not None:
        if arquivo_historico.name.endswith(".csv"):
            df_historico = pd.read_csv(arquivo_historico)
        else:
            df_historico = pd.read_excel(arquivo_historico)

        df_historico.columns = [c.strip().lower() for c in df_historico.columns]

        if {"data", "valor"}.issubset(df_historico.columns):
            df_historico["data"] = pd.to_datetime(df_historico["data"], errors="coerce")
            df_historico = df_historico.sort_values("data")
            st.success(f"Histórico carregado: {len(df_historico)} observações.")
        else:
            st.error("O ficheiro deve conter as colunas 'data' e 'valor'.")
            df_historico = None

    st.markdown("---")
    st.subheader("2. Variáveis exógenas para os próximos 6 meses")

    nomes_exogenas_texto = st.text_input(
        "Nomes das variáveis exógenas (separados por vírgula, de 4 a 6 variáveis)",
        value="preco_petroleo, taxa_cambio, ipi_ch, dummy_estrutural",
        help="Exemplo: preco_petroleo, taxa_cambio, ipi_ch, dummy_estrutural",
    )
    nomes_exogenas = [n.strip() for n in nomes_exogenas_texto.split(",") if n.strip()]

    if not (4 <= len(nomes_exogenas) <= 6):
        st.warning("Defina entre 4 e 6 variáveis exógenas para continuar.")

    modo_entrada = st.radio(
        "Como deseja inserir os valores futuros das exógenas?",
        options=["Upload de ficheiro Excel", "Preenchimento manual (tabela)"],
        horizontal=True,
    )

    df_exogenas_futuras = None

    if modo_entrada == "Upload de ficheiro Excel":
        arquivo_exogenas = st.file_uploader(
            f"Carregue um Excel com {HORIZONTE_MESES} linhas (meses) e colunas: {', '.join(nomes_exogenas)}",
            type=["xlsx"],
            key="upload_exogenas",
        )
        if arquivo_exogenas is not None:
            df_exogenas_futuras = pd.read_excel(arquivo_exogenas)
            df_exogenas_futuras.columns = [c.strip().lower() for c in df_exogenas_futuras.columns]

    else:
        # Tabela editável: 6 linhas (meses futuros) x N colunas (exógenas)
        df_modelo_vazio = pd.DataFrame(
            np.zeros((HORIZONTE_MESES, len(nomes_exogenas))),
            columns=nomes_exogenas,
        )
        df_modelo_vazio.insert(0, "mes_futuro", [f"M+{i+1}" for i in range(HORIZONTE_MESES)])

        st.caption("Edite diretamente os valores na tabela abaixo:")
        df_editado = st.data_editor(
            df_modelo_vazio,
            num_rows="fixed",
            use_container_width=True,
            key="editor_exogenas",
        )
        df_exogenas_futuras = df_editado.drop(columns=["mes_futuro"])

    st.markdown("---")
    st.subheader("3. Gerar previsão")

    pode_prever = (
        df_historico is not None
        and df_exogenas_futuras is not None
        and len(df_exogenas_futuras) == HORIZONTE_MESES
        and 4 <= len(nomes_exogenas) <= 6
    )

    if st.button("🚀 Executar previsão", disabled=not pode_prever, type="primary"):
        try:
            previsao = gerar_previsao(modelo, df_exogenas_futuras[nomes_exogenas])

            ultima_data = df_historico["data"].max()
            datas_futuras = pd.date_range(
                start=ultima_data, periods=HORIZONTE_MESES + 1, freq="MS"
            )[1:]

            df_previsao = pd.DataFrame({"data": datas_futuras, "valor": previsao})

            # --- Gráfico Plotly: histórico + previsão ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_historico["data"], y=df_historico["valor"],
                mode="lines", name="Histórico real",
                line=dict(color="#1f77b4"),
            ))
            fig.add_trace(go.Scatter(
                x=df_previsao["data"], y=df_previsao["valor"],
                mode="lines+markers", name="Previsão (6 meses)",
                line=dict(color="#d62728", dash="dash"),
            ))
            fig.update_layout(
                title=f"Histórico e Previsão — {nome_modelo_selecionado}",
                xaxis_title="Data", yaxis_title="Valor",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Valores previstos")
            st.dataframe(df_previsao, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao gerar a previsão: {e}")
    elif not pode_prever:
        st.info(
            "Carregue o histórico, defina as exógenas e preencha os "
            f"{HORIZONTE_MESES} meses futuros para habilitar a previsão."
        )


# ============================================================================
# ABA 2 - MÉTRICAS DE DESEMPENHO
# ============================================================================

with aba_metricas:
    st.subheader(f"Métricas de validação — {nome_modelo_selecionado}")

    if df_residuos.empty or not {"y_real", "y_previsto"}.issubset(df_residuos.columns):
        st.warning(
            "Não foi possível calcular as métricas: o ficheiro de resíduos "
            "está vazio ou não contém as colunas 'y_real' e 'y_previsto'."
        )
    else:
        metricas = calcular_metricas(
            df_residuos["y_real"].values, df_residuos["y_previsto"].values
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{metricas['RMSE']:.4f}")
        col2.metric("MAE", f"{metricas['MAE']:.4f}")
        col3.metric("MAPE", f"{metricas['MAPE (%)']:.2f}%")
        col4.metric("sMAPE", f"{metricas['sMAPE (%)']:.2f}%")

        st.markdown("---")
        st.subheader("Tabela resumo")
        df_metricas = pd.DataFrame([metricas]).T.rename(columns={0: "Valor"})
        st.dataframe(df_metricas, use_container_width=True)

        st.markdown("---")
        st.subheader("Amostra dos dados de validação (real vs. previsto)")
        st.dataframe(
            df_residuos[["data", "y_real", "y_previsto", "residuo"]].tail(20),
            use_container_width=True,
        )


# ============================================================================
# ABA 3 - ANÁLISE DE RESÍDUOS
# ============================================================================

with aba_residuos:
    st.subheader(f"Análise de resíduos — {nome_modelo_selecionado}")

    if df_residuos.empty or "residuo" not in df_residuos.columns:
        st.warning("Ficheiro de resíduos indisponível ou sem a coluna 'residuo'.")
    else:
        # --- Gráfico 1: evolução temporal dos resíduos ---
        fig_linha = go.Figure()
        fig_linha.add_trace(go.Scatter(
            x=df_residuos["data"], y=df_residuos["residuo"],
            mode="lines+markers", name="Resíduo",
            line=dict(color="#2ca02c"),
        ))
        fig_linha.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_linha.update_layout(
            title="Evolução temporal dos resíduos",
            xaxis_title="Data", yaxis_title="Resíduo",
        )
        st.plotly_chart(fig_linha, use_container_width=True)

        # --- Gráfico 2: distribuição dos resíduos (histograma) ---
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_residuos["residuo"], nbinsx=30,
            marker_color="#9467bd", name="Distribuição",
        ))
        fig_hist.update_layout(
            title="Distribuição dos resíduos",
            xaxis_title="Resíduo", yaxis_title="Frequência",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Média dos resíduos", f"{df_residuos['residuo'].mean():.4f}")
        col2.metric("Desvio padrão", f"{df_residuos['residuo'].std():.4f}")
        col3.metric("Mínimo / Máximo", f"{df_residuos['residuo'].min():.2f} / {df_residuos['residuo'].max():.2f}")
