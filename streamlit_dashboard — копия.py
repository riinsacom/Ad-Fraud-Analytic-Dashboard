import streamlit as st
# from streamlit_autorefresh import st_autorefresh  # Удалено, больше не используется
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
import traceback
try:
    from scipy import stats
except ImportError:
    stats = None  # Fallback if scipy is not available

# Настройка темной темы с улучшенным дизайном
st.set_page_config(
    page_title="Аналитика Фрода",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔍"
)

# Добавляем CSS для масштабирования
st.markdown("""
    <style>
        .main .block-container {
            max-width: 80%;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Применяем современную темную тему через CSS с градиентами и анимациями
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1d29 50%, #0f1419 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stMarkdown {
        color: #ffffff;
    }
    
    .stMetric {
        background: linear-gradient(145deg, #1e2139 0%, #2a2d47 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stMetric > div {
        background: transparent !important;
    }
    
    .stMetric label {
        color: #a0a9c0 !important;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.8rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .stDataFrame {
        background: linear-gradient(145deg, #1e2139 0%, #2a2d47 100%);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(145deg, #1e2139 0%, #2a2d47 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #ffffff;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .stCheckbox > label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(145deg, #1e2139 0%, #2a2d47 100%);
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0a9c0;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #1a1d29 0%, #0f1419 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSidebar .stSelectbox > div > div,
    .stSidebar .stSlider > div > div,
    .stSidebar .stCheckbox {
        background: linear-gradient(145deg, #2a2d47 0%, #1e2139 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .chart-container {
        background: linear-gradient(145deg, #1e2139 0%, #2a2d47 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.2rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(145deg, #ff6b6b22 0%, #ff8e8e22 100%);
        border-left: 4px solid #ff6b6b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 107, 107, 0.2);
    }
    
    .success-box {
        background: linear-gradient(145deg, #51cf6622 0%, #6bcf7f22 100%);
        border-left: 4px solid #51cf66;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(81, 207, 102, 0.2);
    }
    
    .info-box {
        background: linear-gradient(145deg, #339af022 0%, #74c0fc22 100%);
        border-left: 4px solid #339af0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(51, 154, 240, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #667eea11 0%, #764ba211 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    .pattern-alert {
        background: linear-gradient(145deg, #ff6b6b15 0%, #ff8e8e15 100%);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
    }
    
    .modern-table {
        background: linear-gradient(145deg, #1e2139 0%, #2a2d47 100%);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    </style>
    """, unsafe_allow_html=True)

# Заголовок с градиентом
st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; text-align: center;">
            Дашборд для анализа и мониторинга фрод-активности
        </h1>
    </div>
    """, unsafe_allow_html=True)

# Определение улучшенной цветовой схемы для графиков
COLORS = {
    'background': 'rgba(15, 20, 25, 0.8)',
    'paper_bgcolor': 'rgba(30, 33, 57, 0.9)',
    'text': '#ffffff',
    'grid': 'rgba(255, 255, 255, 0.1)',
    'primary': '#667eea',
    'secondary': '#764ba2', 
    'tertiary': '#51cf66',
    'warning': '#ff6b6b', # Existing general warning, can be used for high fraud
    'info': '#339af0',
    'accent': '#f783ac',
    'success': '#51cf66', # Existing general success, can be used for low fraud (within threshold)
    
    # Цвета для светофора фрода
    'traffic_red': '#ff4757',  # Очень красный для высокой опасности
    'traffic_yellow': '#ffa502', # Оранжево-желтый для средней опасности
    'traffic_green': '#2ed573', # Более мягкий зеленый для низкой опасности (но все еще фрод)
    'traffic_below_threshold': '#747d8c', # Серый для значений ниже порога

    'gradient_colors': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
    'fraud_colors': ['#ff4757', '#ff6b6b', '#ffa502', '#2ed573', '#1e90ff'],
    'modern_palette': [
        '#667eea', '#764ba2', '#f093fb', '#f5576c', 
        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
        '#667eea', '#764ba2', '#ffecd2', '#fcb69f'
    ],
    'pie_colors': [
        '#667eea', '#764ba2', '#f093fb', '#f5576c', 
        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
        '#667eea', '#764ba2', '#ffecd2', '#fcb69f'
    ]
}

# Базовый шаблон для графиков
def get_plot_template():
    return {
        'layout': {
            'plot_bgcolor': COLORS['background'],
            'paper_bgcolor': COLORS['paper_bgcolor'],
            'font': {'color': COLORS['text']},
            'xaxis': {
                'gridcolor': COLORS['grid'],
                'zerolinecolor': COLORS['grid']
            },
            'yaxis': {
                'gridcolor': COLORS['grid'],
                'zerolinecolor': COLORS['grid']
            }
        }
    }

# --- Загрузка и объединение данных ---
@st.cache_data
def load_data():
    # Загружаем все строки без ограничения nrows
    test = pd.read_csv('test_small.csv')
    pred = pd.read_csv('Frod_Predict_small.csv')
    df = pd.merge(test, pred, on='click_id', how='left')
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['is_attributed'] = pd.to_numeric(df['is_attributed'], errors='coerce').fillna(0.0)
    return df

# --- Вспомогательные функции ---

def get_fraud_traffic_light_info(fraud_prob, threshold):
    """Определяет категорию и цвет светофора для уровня фрода."""
    if fraud_prob < threshold:
        return {'text': 'Ниже порога', 'color': COLORS['traffic_below_threshold'], 'category': 'below_threshold', 'style': f"color: {COLORS['traffic_below_threshold']};"}
    
    if threshold >= 1.0: # Если порог 100%, все что выше (невозможно) или равно - красное
        return {'text': 'Критический (Красная зона)', 'color': COLORS['traffic_red'], 'category': 'red', 'style': f"background-color: {COLORS['traffic_red']}; color: white; font-weight: bold;"}

    segment_size = (1.0 - threshold) / 3.0
    
    green_upper_bound = threshold + segment_size
    yellow_upper_bound = threshold + 2 * segment_size

    if fraud_prob < green_upper_bound:
        return {'text': f'Низкий риск ({threshold*100:.0f}-{green_upper_bound*100:.0f}%)', 
                'color': COLORS['traffic_green'], 
                'category': 'green_fraud', 
                'style': f"background-color: {COLORS['traffic_green']}; color: black;"}
    elif fraud_prob < yellow_upper_bound:
        return {'text': f'Средний риск ({green_upper_bound*100:.0f}-{yellow_upper_bound*100:.0f}%)', 
                'color': COLORS['traffic_yellow'], 
                'category': 'yellow_fraud',
                'style': f"background-color: {COLORS['traffic_yellow']}; color: black; font-weight: bold;"}
    else: # fraud_prob >= yellow_upper_bound
        return {'text': f'Высокий риск ({yellow_upper_bound*100:.0f}-100%)', 
                'color': COLORS['traffic_red'], 
                'category': 'red_fraud',
                'style': f"background-color: {COLORS['traffic_red']}; color: white; font-weight: bold;"}

def get_related_clicks(df, click_id, field):
    """Получить связанные клики по заданному полю"""
    target_value = df[df['click_id'] == click_id][field].iloc[0]
    return df[df[field] == target_value]

@st.cache_data
def get_suspicious_patterns_cached(df, threshold):
    """Выявление подозрительных паттернов (кэшированная версия)"""
    patterns = []
    if df.empty or 'ip' not in df.columns or 'is_attributed' not in df.columns:
        return patterns
        
    suspicious_ips = df.groupby('ip').agg(
        clicks_count=('is_attributed', 'count'),
        fraud_prob=('is_attributed', 'mean')
    ).reset_index()
    
    if suspicious_ips.empty:
        return patterns
        
    quantile_val = suspicious_ips['clicks_count'].quantile(0.95)
    
    patterns.extend([
        f"IP {row['ip']}: {int(row['clicks_count'])} кликов, вероятность фрода {row['fraud_prob']:.2f}"
        for _, row in suspicious_ips[
            (suspicious_ips['clicks_count'] > quantile_val) & 
            (suspicious_ips['fraud_prob'] > threshold)
        ].iterrows()
    ])
    return patterns

def create_pie_chart(data, values, names, title, show_legend=False):
    """Создание современной круговой диаграммы с градиентами и анимациями"""
    colors = COLORS['pie_colors'][:len(values)]
    fig = go.Figure(data=[go.Pie(
        labels=names,
        values=values,
        hole=.4,
        marker=dict(
            colors=colors,
            line=dict(color='rgba(255, 255, 255, 0.2)', width=2)
        ),
        textfont=dict(size=12, color='white', family='Inter'),
        textposition='inside',  # подписи только внутри
        textinfo='label',       # только название категории
        hovertemplate='<b>%{label}</b><br>' +
                      'Значение: %{value}<br>' +
                      'Процент: %{percent}<br>' +
                      '<extra></extra>',
        rotation=45,
        sort=False
    )])
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16, color='white', family='Inter', weight=600)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        showlegend=False,  # Легенда всегда скрыта
        margin=dict(t=50, b=20, l=20, r=20),
        annotations=[
            dict(
                text=f'<b>Всего<br>{sum(values):,}</b>',
                x=0.5, y=0.5,
                font_size=14,
                font_color='white',
                font_family='Inter',
                showarrow=False
            )
        ]
    )
    fig.update_traces(
        hoverlabel=dict(
            bgcolor='rgba(30, 33, 57, 0.9)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            font_size=12,
            font_family='Inter'
        ),
        marker_line_width=2,
        opacity=0.9
    )
    return fig

data = load_data()

# --- Сайдбар: только общие фильтры ---
st.sidebar.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
           padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;">
    <h2 style="margin: 0; color: white; font-size: 1.4rem; font-weight: 600;">
         Панель управления
    </h2>
    <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.9rem;">
        Глобальные настройки системы
    </p>
</div>
""", unsafe_allow_html=True)

alert_threshold = st.sidebar.slider(
    "Порог вероятности для алерта (фрода)", 0.0, 1.0, 0.5, 0.01,
    help="Устанавливает глобальный порог вероятности, выше которого событие считается подозрительным. Помогает быстро фильтровать и анализировать только потенциально мошеннические записи."
)

# --- Симуляция реального времени ---
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

if 'realtime_mode' not in st.session_state:
    st.session_state['realtime_mode'] = False
if 'realtime_current_sim_time' not in st.session_state: # Переименовано из realtime_time
    st.session_state['realtime_current_sim_time'] = None
if 'realtime_speed' not in st.session_state:
    st.session_state['realtime_speed'] = 60  # Старое значение, будет заменено множителем
if 'simulation_speed_multiplier' not in st.session_state:
    st.session_state['simulation_speed_multiplier'] = 1.0 # Новый множитель скорости, 1x по умолчанию
if 'realtime_start_actual_time' not in st.session_state: # Переименовано из realtime_start_time
    st.session_state['realtime_start_actual_time'] = None
if 'simulated_data_accumulator' not in st.session_state:
    st.session_state['simulated_data_accumulator'] = pd.DataFrame()
if 'last_processed_sim_time' not in st.session_state:
    st.session_state['last_processed_sim_time'] = None

st.sidebar.markdown("""
<div style="background: linear-gradient(145deg, #2a2d47 0%, #1e2139 90%);
           padding: 1.5rem; 
           border-radius: 12px; 
           margin: 1.5rem 0 1rem 0;
           border: 1px solid rgba(255, 255, 255, 0.1); text-align: center;">
    <h3 style="margin: 0 0 0.75rem 0; color: white; font-size: 1.3rem; font-weight: 600;">
         Симуляция <span style="font-weight: 300;">потока данных</span>
    </h3>
    <p style="margin: 0.5rem 0 1rem 0; color: rgba(255,255,255,0.85); font-size: 0.9rem;">
        Запустите или остановите эмуляцию событий в реальном времени и настройте её скорость.
    </p>
</div>
""", unsafe_allow_html=True)

col_sim1, col_sim2 = st.sidebar.columns(2)
with col_sim1:
    if st.button("▶️ Старт симуляции", use_container_width=True, key="start_simulation_button_styled"):
        st.session_state['realtime_mode'] = True
        st.session_state['realtime_current_sim_time'] = None # Сброс текущего времени симуляции
        st.session_state['realtime_start_actual_time'] = None # Сброс времени старта
        # Инициализация simulated_data_accumulator с правильными dtypes
        if not data.empty:
            st.session_state['simulated_data_accumulator'] = data.iloc[0:0].copy()
            # Сохраняем исходные типы данных
            st.session_state['original_dtypes'] = data.dtypes.to_dict()
        else:
            st.session_state['simulated_data_accumulator'] = pd.DataFrame()
            st.session_state['original_dtypes'] = {}
        st.session_state['last_processed_sim_time'] = None # Сброс времени последней обработки
        st.rerun()
with col_sim2:
    if st.button("⏹️ Стоп симуляции", use_container_width=True, key="stop_simulation_button_styled"):
        st.session_state['realtime_mode'] = False
        st.session_state['realtime_current_sim_time'] = None
        st.session_state['realtime_start_actual_time'] = None
        # simulated_data_accumulator и last_processed_sim_time не нужно сбрасывать здесь,
        # так как при следующем старте они инициализируются заново.
        # А при простое они не используются.
        st.rerun()

realtime_speed_label = "Скорость симуляции (старый selectbox, будет удален или изменен)"
# Удаляем старый selectbox для realtime_speed
# realtime_speed = st.sidebar.selectbox(
#     "Скорость симуляции (секунда = ... минут)", [1, 5, 10, 30, 60, 120], index=2,
#     help="Чем больше значение, тем быстрее проходят события. 1 секунда = столько минут данных.",
#     key="realtime_speed_select"
# )
# st.session_state['realtime_speed'] = realtime_speed

# Новый слайдер для множителя скорости
st.sidebar.markdown("<p style='margin-top: 1.2rem; margin-bottom: 0.3rem; font-size:0.95rem; color: rgba(255,255,255,0.9); text-align:left;'>Настройте скорость эмуляции:</p>", unsafe_allow_html=True)
st.session_state['simulation_speed_multiplier'] = st.sidebar.slider(
    "Множитель скорости симуляции",
    min_value=1.0, max_value=120.0, value=st.session_state.get('simulation_speed_multiplier', 1.0), step=1.0,
    help="Ускоряет течение симулированного времени. 1x = реальное время, 60x = 1 реальная секунда равна 1 симулированной минуте."
)

# --- Автообновление страницы только во время симуляции ---
if st.session_state.get('realtime_mode', False): # Проверяем только режим
    if st_autorefresh is not None:
        st_autorefresh(interval=2000, key="realtime_autorefresh_key_v3") # Обновление каждые 2 секунды
        if st.session_state.get('realtime_current_sim_time'):
            st.sidebar.info(f"Время симуляции: {st.session_state['realtime_current_sim_time'].strftime('%Y-%m-%d %H:%M:%S')}")

        # Диагностический счетчик для проверки работы autorefresh
        if 'autorefresh_diagnostic_counter' not in st.session_state:
            st.session_state.autorefresh_diagnostic_counter = 0
        st.session_state.autorefresh_diagnostic_counter += 1
        st.sidebar.caption(f"Авто-обновление тикает: #{st.session_state.autorefresh_diagnostic_counter}")
    else:
        st.sidebar.warning("Модуль `streamlit-autorefresh` не найден или не импортирован. "
                           "Для автоматического обновления данных в реальном времени, пожалуйста, "
                           "установите его: `pip install streamlit-autorefresh` и перезапустите приложение.")
        if st.sidebar.button("Обновить данные симуляции вручную", key="manual_refresh_sim_button"):
            st.rerun()

# --- Логика фильтрации данных для симуляции ---
if st.session_state.get('realtime_mode', False) and not data.empty:
    time_min_data = data['click_time'].min().to_pydatetime()
    time_max_data = data['click_time'].max().to_pydatetime()

    if st.session_state.get('realtime_start_actual_time') is None:
        st.session_state['realtime_start_actual_time'] = datetime.now() 
        st.session_state['realtime_current_sim_time'] = time_min_data 
        st.session_state['last_processed_sim_time'] = time_min_data - timedelta(seconds=1) # чтобы первая порция захватилась
        # Инициализация simulated_data_accumulator и original_dtypes, если еще не было
        if 'original_dtypes' not in st.session_state or not st.session_state['original_dtypes']:
            if not data.empty:
                st.session_state['simulated_data_accumulator'] = data.iloc[0:0].copy()
                st.session_state['original_dtypes'] = data.dtypes.to_dict()
            else:
                st.session_state['simulated_data_accumulator'] = pd.DataFrame()
                st.session_state['original_dtypes'] = {}
        elif data.empty and isinstance(st.session_state.get('simulated_data_accumulator'), pd.DataFrame) and st.session_state['simulated_data_accumulator'].empty:
             pass # Dtypes и аккумулятор уже установлены
        elif not data.empty and (not isinstance(st.session_state.get('simulated_data_accumulator'), pd.DataFrame) or st.session_state['simulated_data_accumulator'].empty):
            st.session_state['simulated_data_accumulator'] = data.iloc[0:0].copy()
    
    elapsed_actual_seconds = (datetime.now() - st.session_state['realtime_start_actual_time']).total_seconds()
    simulated_seconds_passed = elapsed_actual_seconds * st.session_state.get('simulation_speed_multiplier', 1.0)
    current_sim_time_boundary = time_min_data + timedelta(seconds=simulated_seconds_passed)

    new_data_chunk = data[(data['click_time'] > st.session_state['last_processed_sim_time']) & (data['click_time'] <= current_sim_time_boundary)]

    if not new_data_chunk.empty:
        st.session_state['simulated_data_accumulator'] = pd.concat(
            [st.session_state['simulated_data_accumulator'], new_data_chunk],
            ignore_index=True
        )
        if st.session_state.get('original_dtypes'):
            try:
                st.session_state['simulated_data_accumulator'] = st.session_state['simulated_data_accumulator'].astype(st.session_state['original_dtypes'])
            except Exception as e:
                st.error(f"Ошибка приведения типов данных: {e}")
    
    st.session_state['last_processed_sim_time'] = current_sim_time_boundary
    st.session_state['realtime_current_sim_time'] = current_sim_time_boundary # Обновляем для отображения

    # ВАЖНО: не фильтруем по времени! Просто берем все накопленные данные
    filtered_data_base = st.session_state['simulated_data_accumulator'].copy()

    # Если достигли конца и обработали все данные
    if current_sim_time_boundary >= time_max_data and st.session_state['last_processed_sim_time'] >= time_max_data:
        if st.session_state['realtime_mode']: # Проверяем, что все еще в режиме, прежде чем выключать
            st.sidebar.success("Симуляция завершена! Все данные обработаны.")
            st.session_state['realtime_mode'] = False
            st.session_state['realtime_start_actual_time'] = None # Сброс времени старта
            # simulated_data_accumulator и last_processed_sim_time можно оставить или сбросить по желанию
            st.rerun() # <--- Добавляем rerun для немедленного обновления UI
    st.sidebar.slider(
        "Временной диапазон (симуляция активна)",
        min_value=time_min_data, max_value=time_max_data,
        value=(time_min_data, st.session_state['realtime_current_sim_time']), format="YYYY-MM-DD HH:mm:ss",
        disabled=True
    )

elif not data.empty:
    time_min_data = data['click_time'].min().to_pydatetime()
    time_max_data = data['click_time'].max().to_pydatetime()
    if 'time_range_value' not in st.session_state: 
        default_start = time_max_data - timedelta(hours=1)
        default_end = time_max_data
        st.session_state['time_range_value'] = (default_start, default_end)
    time_range_value = st.sidebar.slider(
        "Временной диапазон",
        min_value=time_min_data, max_value=time_max_data,
        value=st.session_state['time_range_value'], format="YYYY-MM-DD HH:mm:ss",
        help="Позволяет анализировать данные за выбранный период. Это помогает выявлять всплески мошенничества, сезонные аномалии и сравнивать разные временные интервалы.",
        key="main_time_slider",
        on_change=lambda: st.session_state.update(time_range_value=st.session_state.main_time_slider)
    )
    filtered_data_base = data[(data['click_time'] >= time_range_value[0]) & (data['click_time'] <= time_range_value[1])].copy()
else:
    st.error("Нет данных для отображения после загрузки. Проверьте исходные файлы.")
    filtered_data_base = pd.DataFrame(columns=data.columns)
    dt_now = datetime.now()
    time_range = st.sidebar.slider("Временной диапазон", min_value=dt_now - timedelta(days=1), max_value=dt_now, 
                                  value=(dt_now - timedelta(days=1), dt_now), format="YYYY-MM-DD HH:mm:ss")

# Показываем общую статистику в сайдбаре с красивым дизайном
st.sidebar.markdown("""
<div style="background: linear-gradient(145deg, #1e2139 0%, #2a2d47 100%); 
           padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;
           border: 1px solid rgba(255, 255, 255, 0.1);">
    <h3 style="margin: 0 0 1rem 0; color: white; font-size: 1.2rem; font-weight: 600;">
         Общая статистика
    </h3>
</div>
""", unsafe_allow_html=True)

if not filtered_data_base.empty:
    total_records = len(filtered_data_base)
    fraud_records = (filtered_data_base['is_attributed'] > alert_threshold).sum()
    fraud_rate = fraud_records / total_records if total_records > 0 else 0
    
    # Красивые метрики в сайдбаре
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(145deg, #667eea11 0%, #764ba211 100%); 
               padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
               border: 1px solid rgba(102, 126, 234, 0.2);">
        <div style="color: #667eea; font-size: 1.8rem; font-weight: 700;">
            {total_records:,}
        </div>
        <div style="color: #a0a9c0; font-size: 0.8rem;">
             ВСЕГО ЗАПИСЕЙ
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    fraud_color = "#ff6b6b" if fraud_rate > 0.1 else "#ffa502" if fraud_rate > 0.05 else "#51cf66"
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(145deg, #667eea11 0%, #764ba211 100%); 
               padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
               border: 1px solid rgba(102, 126, 234, 0.2);">
        <div style="color: {fraud_color}; font-size: 1.8rem; font-weight: 700;">
            {fraud_records:,}
        </div>
        <div style="color: #a0a9c0; font-size: 0.8rem;">
             ПОДОЗРИТЕЛЬНЫХ
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(145deg, #667eea11 0%, #764ba211 100%); 
               padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
               border: 1px solid rgba(102, 126, 234, 0.2);">
        <div style="color: {fraud_color}; font-size: 1.8rem; font-weight: 700;">
            {fraud_rate:.1%}
        </div>
        <div style="color: #a0a9c0; font-size: 0.8rem;">
             ДОЛЯ ФРОДА
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Основной DataFrame для вкладок ---
current_df = filtered_data_base.copy()

# --- Tabs ---
# Сохранение и восстановление активной вкладки
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Главная" # Имя первой вкладки по умолчанию

def on_tab_change():
    # st.session_state.active_tab будет автоматически обновлен Streamlit благодаря параметру key в st.tabs
    pass

tabs_list = ["Главная", "Категории", "Связи/Графы", "Корреляции", "Алерты", "Последние события"]
# st.tabs теперь должен принимать key, чтобы его состояние сохранялось автоматически
# Однако, st.tabs не имеет параметра on_change в привычном виде и key не сохраняет активную вкладку между rerun-ами st_autorefresh.
# Streamlit управляет активной вкладкой через query params в URL, если вкладкам даны уникальные имена.

# Попробуем установить выбранную вкладку через selected_tab параметр, если он существует
# st.experimental_set_query_params не сохраняет состояние между st_autorefresh,
# поэтому мы будем полагаться на стандартное поведение st.tabs, если имена вкладок уникальны.

# Вместо selected_tab, мы будем использовать немного другой подход.
# Streamlit >1.17.0 сохраняет состояние виджетов, включая st.tabs, при st.rerun(), если у них есть уникальный key.
# Однако, st_autorefresh может вести себя иначе.
# Давайте убедимся, что активная вкладка сохраняется с помощью session_state и выбора по умолчанию.

# Мы не можем напрямую задать активную вкладку для st.tabs() после его создания без query_params.
# Сохранение ключа активной вкладки:
if 'selected_tab_key' not in st.session_state:
    st.session_state.selected_tab_key = tabs_list[0]

# Эта функция будет вызываться при смене вкладки
def _set_active_tab():
    st.session_state.selected_tab_key = st.session_state.query_params_tab_key # query_params_tab_key - это ключ виджета st.tabs

# tabs = st.tabs(tabs_list, key="query_params_tab_key", on_change=_set_active_tab)
# on_change в st.tabs не работает так, как ожидается для этой цели.
# Streamlit должен сам запоминать активную вкладку, если у виджета st.tabs есть `key`.

# Однако, st_autorefresh сбрасывает это.
# Попробуем программно выбрать вкладку через JavaScript, если стандартные методы не работают.
# Это очень хакки, и может быть не стабильно.
# Простой способ - сохранить индекс.

if 'active_tab_index' not in st.session_state:
    st.session_state.active_tab_index = 0

def handle_tab_change():
    # st.session_state.tab_key - это ключ, который мы дадим st.tabs
    # Найдем индекс выбранной вкладки по имени (которое возвращает st.tabs с key)
    # st.tabs возвращает имя выбранной вкладки, если ему передать key
    # Эта логика здесь избыточна, если Streamlit сам сохраняет состояние по key.
    # Проблема в том, что st_autorefresh вызывает st.rerun(), который может сбрасывать UI состояние.
    # У Streamlit нет прямого способа установить активную вкладку через Python после ее рендеринга, кроме как через query params.
    # Но query params тоже могут сбрасываться.
    
    # Лучший способ - это передать default в st.radio или st.selectbox, если бы мы использовали их для навигации.
    # Для st.tabs, если key задан, он должен сам это делать.
    
    # Если `st_autorefresh` все равно сбрасывает, то это ограничение.
    # Давайте проверим, сохраняется ли состояние `st.tabs` с `key` при `st_autorefresh`.
    # Если нет, то это сложно обойти без JS хаков или изменения логики навигации (например, на st.radio в сайдбаре).

    # На данный момент, просто присвоим key и посмотрим.
    # Если вкладка все равно сбрасывается, то единственный надежный способ - убрать autorefresh
    # или смириться с этим поведением, т.к. autorefresh по сути перезапускает скрипт.
    pass


tab_key_val = "main_tabs_selector" # Уникальный ключ для виджета вкладок

# Получаем текущую выбранную вкладку (если она уже была установлена)
# Если ключа нет в session_state (первый запуск), Streamlit выберет первую вкладку.
# Если ключ есть, Streamlit попытается восстановить его.
# Проблема в том, что st_autorefresh вызывает st.rerun(), и состояние виджета st.tabs может не всегда корректно восстанавливаться
# только лишь по `key` в таком сценарии динамического обновления.

# Попробуем управлять этим через query parameters, что является более официальным способом Streamlit
# для установки состояния виджетов через URL.
# Закомментируем этот блок, так как он сложен в реализации с st_autorefresh

# tab_names = ["Главная", "Категории", "Связи/Графы", "Корреляции", "Алерты", "Последние события"]
# query_params = st.experimental_get_query_params()
# current_query_tab = query_params.get("tab", [None])[0]

# active_tab_name = st.session_state.get("active_tab_name", tab_names[0])

# if current_query_tab and current_query_tab != active_tab_name and current_query_tab in tab_names:
# st.session_state.active_tab_name = current_query_tab
# active_tab_name = current_query_tab
# elif not current_query_tab: # Если в URL нет параметра tab, устанавливаем его
# st.experimental_set_query_params(tab=active_tab_name)


# def update_active_tab_from_query_params():
    # """Обновляет активную вкладку в session_state из query params."""
    # query_params = st.experimental_get_query_params()
    # query_tab = query_params.get("tab", [None])[0]
    # if query_tab and query_tab in tab_names:
        # st.session_state.active_tab_name = query_tab

# update_active_tab_from_query_params() # Вызываем при каждом rerun

# selected_tab = st.tabs(
    # tab_names,
    # key="main_tabs_widget" #  Уникальный ключ для st.tabs
# )

# # Обновляем query param при смене вкладки пользователем
# # st.tabs возвращает имя выбранной вкладки.
# # Мы не можем использовать on_change для st.tabs напрямую, чтобы вызвать set_query_params.
# # Это должно происходить автоматически, если Streamlit правильно обрабатывает key.

# # Чтобы это работало с st_autorefresh, нужно, чтобы st_autorefresh не сбрасывал URL.
# # Проблема в том, что st.experimental_get_query_params() и st.experimental_set_query_params()
# # могут не всегда надежно работать с st_autorefresh в плане сохранения состояния UI между авто-реранами.

# Простой подход: сохранить индекс активной вкладки в session_state.
# st.tabs не возвращает индекс напрямую, а имя. Мы можем найти индекс по имени.

# Если 'active_tab_name' не в session_state, инициализируем его.
if 'active_tab_name' not in st.session_state:
    st.session_state.active_tab_name = tabs_list[0]

# Создаем вкладки. Streamlit должен запоминать активную вкладку по `key` при `st.rerun`.
# Пользовательский ввод (смена вкладки) должен обновлять `st.session_state.active_tab_name`.
# Мы не можем использовать `on_change` для `st.tabs`.
# Вместо этого, мы прочитаем состояние виджета `st.tabs` после его рендеринга.

# `st.tabs` сам по себе должен сохранять состояние при наличии `key` в рамках одного сеанса и обычных `st.rerun`.
# Проблема именно с `st_autorefresh`.

# Давайте попробуем самый простой подход: дать `st.tabs` ключ.
# Если это не сработает с `st_autorefresh`, то это ограничение Streamlit.
# И единственный способ - это навигация через `st.radio` или `st.selectbox` в сайдбаре,
# где мы можем явно контролировать выбранное значение через `st.session_state`.

# _tabs_instance = st.tabs(tabs_list, key="main_tabs_control") # Убираем key
_tabs_instance = st.tabs(tabs_list)

# После того как st.tabs отрисован, его текущее значение (имя активной вкладки)
# будет доступно через st.session_state.main_tabs_control (если Streamlit < 1.18)
# или просто сам факт выбора будет сохранен Streamlit (>= 1.18)
# Мы не можем активно *установить* вкладку через Python здесь без query_params.

# Если realtime_mode активен, мы хотим, чтобы выбранная вкладка СОХРАНЯЛАСЬ.
# Streamlit обычно делает это автоматически для виджетов с ключами.
# Проблема с st_autorefresh заключается в том, что он может вести себя как "более жесткий" rerun.

# Вывод: Самый надежный способ управлять состоянием вкладок при автообновлении -
# это использовать query parameters. Однако, это может сделать URL длиннее.
# Второй по надежности способ - если st.tabs с key="unique_key" сам сохраняет состояние.
# Если это не работает с st_autorefresh, то остается только навигация через другие виджеты (radio/selectbox),
# чье состояние мы полностью контролируем через session_state.

# Текущая реализация `tabs = st.tabs(...)` должна быть заменена на `tab1, tab2, ... = st.tabs(...)`
# или мы должны использовать индекс для доступа.

tabs = _tabs_instance

# Чтобы определить, какая вкладка активна, мы можем перебирать их.
# Это не идеальный подход. Если st.tabs с key работает как надо, этого не нужно.

# Давайте положимся на то, что Streamlit > 1.18+ сам корректно обрабатывает key для st.tabs при st.rerun.
# Проблема может быть специфична для st_autorefresh.

# --- Главная ---
# with tabs[0]: # Старый способ
with tabs[0]: # Возвращаемся к индексам
    # Красивый заголовок секции
    st.markdown('<div class="section-header">Ключевые показатели эффективности</div>', unsafe_allow_html=True)
    
    # Настройки для главной вкладки
    col_settings, col_spacer = st.columns([3, 1])
    with col_settings:
        top_n_main = st.selectbox(
            "Количество элементов в топах", [5, 10, 15, 20], index=1,
            key="top_n_main",
            help="**Что это?** Настройка количества отображаемых верхних позиций в различных рейтингах и распределениях на главной вкладке (например, топ приложений, топ каналов).\n\n**Зачем он нужен?** Позволяет сфокусировать анализ на наиболее значимых или активных сегментах данных, отсекая менее релевантные.\n\n**Как им пользоваться?** Выберите желаемое число из списка (5, 10, 15 или 20). Графики и списки на этой вкладке автоматически обновятся, показывая выбранное количество топовых элементов.\n\n**Чем он полезен?** Улучшает читаемость дашборда, особенно при большом количестве категорий. Помогает быстро выделить ключевые сущности, требующие внимания, и избежать перегруженности визуализаций."
        )
    
    # Улучшенные метрики с градиентом
    total_clicks = len(current_df)
    avg_fraud_prob = current_df['is_attributed'].mean() if total_clicks > 0 else 0
    fraud_clicks = (current_df['is_attributed'] > alert_threshold).sum()
    fraud_share = fraud_clicks / total_clicks if total_clicks > 0 else 0
    
    # Создаем красивые метрики
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        # Определяем статус
        status_color = "#51cf66" if total_clicks > 1000 else "#ffa502" if total_clicks > 100 else "#ff6b6b"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {status_color}; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                {total_clicks:,}
            </div>
            <div style="color: #a0a9c0; font-size: 0.9rem; font-weight: 500;">
                 ВСЕГО КЛИКОВ
            </div>
            <div style="color: {status_color}; font-size: 0.8rem; margin-top: 0.3rem;">
                {"Высокая активность" if total_clicks > 1000 else "Умеренная активность" if total_clicks > 100 else "Низкая активность"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        fraud_status = "КРИТИЧЕСКИЙ" if avg_fraud_prob > 0.3 else "ПОВЫШЕННЫЙ" if avg_fraud_prob > 0.1 else "НОРМАЛЬНЫЙ"
        fraud_color = "#ff6b6b" if avg_fraud_prob > 0.3 else "#ffa502" if avg_fraud_prob > 0.1 else "#51cf66"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {fraud_color}; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                {avg_fraud_prob:.1%}
            </div>
            <div style="color: #a0a9c0; font-size: 0.9rem; font-weight: 500;">
                СРЕДНИЙ УРОВЕНЬ ФРОДА
            </div>
            <div style="color: {fraud_color}; font-size: 0.8rem; margin-top: 0.3rem;">
                {fraud_status} РИСК
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        alert_color = "#ff6b6b" if fraud_clicks > 100 else "#ffa502" if fraud_clicks > 10 else "#51cf66"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {alert_color}; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                {fraud_clicks:,}
            </div>
            <div style="color: #a0a9c0; font-size: 0.9rem; font-weight: 500;">
                АЛЕРТОВ (>{alert_threshold:.0%})
            </div>
            <div style="color: {alert_color}; font-size: 0.8rem; margin-top: 0.3rem;">
                {"Много угроз" if fraud_clicks > 100 else "Есть угрозы" if fraud_clicks > 10 else "Мало угроз"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col4:
        share_color = "#ff6b6b" if fraud_share > 0.1 else "#ffa502" if fraud_share > 0.05 else "#51cf66"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {share_color}; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                {fraud_share:.1%}
            </div>
            <div style="color: #a0a9c0; font-size: 0.9rem; font-weight: 500;">
                ДОЛЯ МОШЕННИЧЕСТВА
            </div>
            <div style="color: {share_color}; font-size: 0.8rem; margin-top: 0.3rem;">
                {"Высокая доля" if fraud_share > 0.1 else "Умеренная доля" if fraud_share > 0.05 else "Низкая доля"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Улучшенные круговые диаграммы с настройками
    st.markdown('<div class="section-header">Общее распределение кликов</div>', unsafe_allow_html=True)
    
    # Настройки для круговых диаграмм
    chart_height = 500
    show_legend_main_cb = False
    
    # Контейнер для графиков
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    pie_cols = st.columns(4)
    
    if not current_df.empty:
        with pie_cols[0]:
            device_stats = current_df['device'].value_counts().head(top_n_main)
            fig_device = create_pie_chart(
                current_df, 
                device_stats.values, 
                [f"Device {idx}" for idx in device_stats.index],
                'Распределение по устройствам'
            )
            fig_device.update_layout(height=chart_height)
            st.plotly_chart(fig_device, use_container_width=True, config={'displayModeBar': False})
        
        with pie_cols[1]:
            app_stats = current_df['app'].value_counts().head(top_n_main)
            fig_app = create_pie_chart(
                current_df,
                app_stats.values,
                [f"App {idx}" for idx in app_stats.index],
                f'Топ-{top_n_main} приложений'
            )
            fig_app.update_layout(height=chart_height)
            st.plotly_chart(fig_app, use_container_width=True, config={'displayModeBar': False})
        
        with pie_cols[2]:
            channel_stats = current_df['channel'].value_counts().head(top_n_main)
            fig_channel = create_pie_chart(
                current_df,
                channel_stats.values,
                [f"Channel {idx}" for idx in channel_stats.index],
                f'Топ-{top_n_main} каналов'
            )
            fig_channel.update_layout(height=chart_height)
            st.plotly_chart(fig_channel, use_container_width=True, config={'displayModeBar': False})
        
        with pie_cols[3]:
            fraud_stats = pd.Series({
                'Не фрод': total_clicks - fraud_clicks,
                'Фрод': fraud_clicks
            })
            fig_fraud = create_pie_chart(
                current_df,
                fraud_stats.values,
                fraud_stats.index,
                'Распределение фрод/не фрод'
            )
            fig_fraud.update_traces(marker_colors=['#51cf66', '#ff6b6b'])
            fig_fraud.update_layout(height=chart_height)
            st.plotly_chart(fig_fraud, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown('<div class="info-box">ℹ️ Нет данных для отображения круговых диаграмм.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Подробное пояснение для секции паттернов
    st.markdown("""
    <div style='margin-bottom: 1.5rem; padding: 1rem; background: linear-gradient(90deg, #667eea22 0%, #764ba222 100%); border-radius: 10px;'>
    <b>Что такое подозрительные паттерны активности?</b><br>
    <ul style='margin-top: 0.5rem;'>
      <li>Паттерн — это повторяющаяся аномалия в поведении пользователей, устройств или IP-адресов, которая может указывать на мошенничество.</li>
      <li>Система автоматически ищет группы с необычно высокой активностью и вероятностью фрода.</li>
      <li><b>Порог для паттернов</b> позволяет фильтровать только самые подозрительные группы: чем выше порог, тем строже фильтрация.</li>
      <li>Анализ паттернов помогает выявлять массовые атаки, ботнеты, скликивание и другие схемы мошенничества, которые не видны при обычном просмотре данных.</li>
    </ul>
    <i>Используйте этот инструмент для быстрого обнаружения новых угроз и оценки эффективности антифрод-мер.</i>
    </div>
    """, unsafe_allow_html=True)

    # Подозрительные паттерны с настройками
    st.markdown('<div class="section-header">Подозрительные паттерны активности</div>', unsafe_allow_html=True)
    
    # Настройки для анализа паттернов
    pattern_settings_col1, pattern_settings_col2 = st.columns(2)
    with pattern_settings_col1:
        pattern_threshold = st.slider(
            "Порог для паттернов", 0.0, 1.0, alert_threshold, 0.05,
            key="pattern_threshold",
            help="**Что это?** Уровень отсечения по вероятности фрода для выявления групп подозрительной активности (паттернов).\n\n**Зачем он нужен?** Помогает отфильтровать и показать только те группы (например, IP-адреса), где средняя вероятность фрода превышает заданное значение, при условии аномально высокой активности.\n\n**Как им пользоваться?** Передвиньте слайдер для установки порога. Чем выше значение, тем более строгие критерии применяются для выявления паттернов, и тем меньше их будет показано. Паттерны также учитывают 95-й квантиль по количеству кликов.\n\n**Чем он полезен?** Позволяет сфокусироваться на самых очевидных и потенциально опасных аномалиях, связанных с высокой концентрацией фрода и активностью."
        )
    with pattern_settings_col2:
        max_patterns = st.selectbox(
            "Максимум паттернов", [3, 5, 10], index=1, key="max_patterns",
            help="**Что это?** Ограничение на количество одновременно отображаемых подозрительных паттернов на главной вкладке.\n\n**Зачем он нужен?** Предотвращает перегрузку интерфейса информацией, если обнаружено много паттернов.\n\n**Как им пользоваться?** Выберите желаемое максимальное количество из списка. Будут показаны наиболее значимые паттерны в рамках этого лимита.\n\n**Чем он полезен?** Обеспечивает фокусировку на самых приоритетных угрозах, выявленных через анализ паттернов, и сохраняет наглядность дашборда."
        )
    
    patterns = get_suspicious_patterns_cached(current_df, pattern_threshold)
    if patterns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        for i, pattern_text in enumerate(patterns[:max_patterns]):
            # Извлекаем вероятность фрода из текста паттерна для светофора
            # Пример текста: "IP 123.45.67.89: 150 кликов, вероятность фрода 0.75"
            try:
                fraud_prob_text = pattern_text.split("вероятность фрода ")[-1]
                fraud_prob_value = float(fraud_prob_text)
                # Используем pattern_threshold для светофора, так как он определяет, что считать паттерном
                traffic_light = get_fraud_traffic_light_info(fraud_prob_value, pattern_threshold) 
            except Exception: # Изменено с except: на except Exception:
                # Фоллбэк, если не удалось распарсить вероятность
                traffic_light = {'text': 'Неопределенный риск', 'style': "background-color: #747d8c; color: white;", 'category': 'unknown'}

            header_text = ""
            icon = ""
            if traffic_light['category'] == 'red_fraud':
                header_text = f"КРИТИЧЕСКАЯ УГРОЗА #{i+1} ({traffic_light['text']})"
                container_class = "pattern-alert"
            elif traffic_light['category'] == 'yellow_fraud':
                header_text = f"ВЫСОКИЙ РИСК #{i+1} ({traffic_light['text']})"
                container_class = "warning-box"
            elif traffic_light['category'] == 'green_fraud':
                header_text = f"ПОДОЗРИТЕЛЬНАЯ АКТИВНОСТЬ #{i+1} ({traffic_light['text']})"
                container_class = "info-box"
            else: # ниже порога или неизвестно
                header_text = f"ЗАМЕЧАНИЕ #{i+1} ({traffic_light['text']})"
                container_class = "info-box"
            
            st.markdown(f"""
            <div class="{container_class}" style="border-left-color: {traffic_light['color']};">
                <strong>{header_text}</strong><br>
                {pattern_text}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            ✅ <strong>БЕЗОПАСНОСТЬ</strong><br>
            Подозрительных паттернов не обнаружено. Система функционирует в штатном режиме.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Топ мошеннических сущностей</div>', unsafe_allow_html=True)
    
    # Настройки для топ сущностей
    # Убрана третья колонка и слайдер для entity_threshold
    entity_settings_col1, entity_settings_col2 = st.columns(2)
    with entity_settings_col1:
        top_n_entities = st.slider(
            "Количество топ сущностей", 3, 20, 10, key="top_n_entities",
            help="**Что это?** Определяет, сколько IP-адресов и приложений будет показано в списках \"Топ мошеннических сущностей\".\n\n**Зачем он нужен?** Позволяет настроить глубину анализа наиболее активных или подозрительных IP и приложений.\n\n**Как им пользоваться?** Передвиньте слайдер, чтобы выбрать желаемое количество. Таблицы топ-сущностей обновятся.\n\n**Чем он полезен?** Помогает выявить ключевые источники или цели фрода, ограничивая вывод наиболее релевантными записями для упрощения анализа."
        )
    
    # Локальный entity_threshold удален, теперь используется alert_threshold из сайдбара
    # entity_threshold = alert_threshold # Это присваивание больше не нужно здесь, так как alert_threshold используется напрямую

    with entity_settings_col2:
        sort_by = st.selectbox(
            "Сортировать по", ["Количество кликов", "Средний фрод"],
            index=0, key="sort_by_entities",
            help=("**Что это?** Критерий для ранжирования IP-адресов и приложений в списках \"Топ мошеннических сущностей\".\n\n"
                   "**Зачем он нужен?** Позволяет анализировать сущности либо по их общей активности (массовости), либо по степени их предполагаемой вредоносности.\n\n"
                   "**Как им пользоваться?**\n"
                   "- **Количество кликов:** Сущности с наибольшим числом событий будут наверху.\n"
                   "- **Средний фрод:** Сущности с наивысшей средней вероятностью фрода будут наверху.\n\n"
                   "**Чем он полезен?** Предоставляет два разных взгляда на данные: выявление наиболее активных участников (потенциально ботнеты) или наиболее \"токсичных\" (с высоким риском фрода).")
        )
    
    # Используем глобальный alert_threshold напрямую
    current_entity_threshold = alert_threshold 

    if not current_df.empty and 'is_attributed' in current_df.columns and 'ip' in current_df.columns and 'app' in current_df.columns:
        # Фильтруем по current_entity_threshold (который теперь равен глобальному alert_threshold)
        high_fraud_df = current_df[current_df['is_attributed'] > current_entity_threshold]
        
        if high_fraud_df.empty:
            st.markdown(f'<div class="info-box">ℹ️ Нет кликов выше глобального порога фрода {current_entity_threshold:.1%}. Показываются топ по всем данным с P(фрод) > 0.01.</div>', unsafe_allow_html=True)
            high_fraud_df = current_df[current_df['is_attributed'] > 0.01]
            if high_fraud_df.empty:
                 high_fraud_df = current_df # Если и таких нет, показываем по всему датафрейму

        # Топ IP адресов
        suspicious_ips_agg = high_fraud_df.groupby('ip').agg(
            click_count=('click_id', 'count'),
            avg_fraud_prob=('is_attributed', 'mean')
        ).reset_index()
        
        sort_column_ip = 'click_count' if sort_by == "Количество кликов" else 'avg_fraud_prob'
        suspicious_ips_table = suspicious_ips_agg.sort_values(by=sort_column_ip, ascending=False).head(top_n_entities)
        suspicious_ips_table.columns = ['IP', 'Количество кликов', 'Средняя P(фрод)']
        
        # Топ приложений
        suspicious_apps_agg = high_fraud_df.groupby('app').agg(
            click_count=('click_id', 'count'),
            avg_fraud_prob=('is_attributed', 'mean')
        ).reset_index()
        sort_column_app = 'click_count' if sort_by == "Количество кликов" else 'avg_fraud_prob'
        suspicious_apps_table = suspicious_apps_agg.sort_values(by=sort_column_app, ascending=False).head(top_n_entities)
        suspicious_apps_table.columns = ['App ID', 'Количество кликов', 'Средняя P(фрод)']

        col_ip_fraud, col_app_fraud = st.columns(2)
        with col_ip_fraud:
            st.write(f"**Топ-{top_n_entities} IP адресов** (глоб. порог: {current_entity_threshold:.1%}, сорт: {sort_by.lower()}):")
            
            # Определение функции create_styled_table_html непосредственно перед использованием
            def create_styled_table_html(df, fraud_column_name, threshold_for_traffic_light):
                """Создает HTML-таблицу со стилизацией светофора для колонки фрода."""
                headers = "".join(f"<th>{col}</th>" for col in df.columns)
                rows_html = ""
                for _, row in df.iterrows():
                    row_html = "<tr>"
                    for col_name, cell_value in row.items():
                        style = ""
                        display_value = cell_value
                        if col_name == fraud_column_name:
                            # Предполагается, что get_fraud_traffic_light_info определена глобально
                            traffic_light_info = get_fraud_traffic_light_info(cell_value, threshold_for_traffic_light)
                            style = traffic_light_info['style']
                            display_value = f"{cell_value:.3f}"
                        elif isinstance(cell_value, float):
                            display_value = f"{cell_value:.3f}"
                        
                        row_html += f'<td style="{style}">{display_value}</td>'
                    row_html += "</tr>"
                    rows_html += row_html

                table_html = f"""
                <div class="modern-table">
                    <table>
                        <thead><tr>{headers}</tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
                """
                return table_html

            html_table_ips = create_styled_table_html(suspicious_ips_table, 'Средняя P(фрод)', current_entity_threshold)
            st.markdown(html_table_ips, unsafe_allow_html=True)
            
        with col_app_fraud:
            st.write(f"**Топ-{top_n_entities} приложений** (глоб. порог: {current_entity_threshold:.1%}, сорт: {sort_by.lower()}):")
            html_table_apps = create_styled_table_html(suspicious_apps_table, 'Средняя P(фрод)', current_entity_threshold)
            st.markdown(html_table_apps, unsafe_allow_html=True)

# --- Категории ---
# with tabs[1]:
with tabs[1]:
    st.subheader("Анализ по категориям")
    
    # Настройки для категорий (перенесены из сайдбара)
    cat_settings_col1, cat_settings_col2, cat_settings_col3 = st.columns(3)
    
    with cat_settings_col1:
        cat_options = ['ip', 'app', 'device', 'channel']
        cat1 = st.selectbox(
            "Категория 1 (ось X)", cat_options, index=0, key="cat1_main",
            help="**Что это?** Основная категория (например, IP, приложение, устройство, канал), которая будет использоваться для первичной агрегации и как главная ось (X) на графиках в этой секции.\n\n**Зачем он нужен?** Позволяет выбрать главный признак для анализа распределения кликов и среднего уровня фрода.\n\n**Как им пользоваться?** Выберите одну из доступных категорий из списка. Графики \"Топ по количеству кликов\" и \"Топ по ср. вероятности фрода\" обновятся в соответствии с вашим выбором.\n\n**Чем он полезен?** Помогает быстро оценить, какие значения выбранной категории наиболее активны или наиболее подвержены фроду."
        )
    with cat_settings_col2:
        cat2 = st.selectbox(
            "Категория 2 (группировка)", cat_options, index=1, key="cat2_main",
            help="**Что это?** Вторая категория, используемая для построения тепловой карты пересечений с Категорией 1.\n\n**Зачем он нужен?** Позволяет анализировать взаимодействие и совместную встречаемость двух разных категорий, чтобы выявить их взаимное влияние или совместные паттерны.\n\n**Как им пользоваться?** Выберите категорию, отличную от Категории 1. Тепловая карта ниже покажет количество совместных кликов для пар значений из Категории 1 и Категории 2.\n\n**Чем он полезен?** Помогает обнаружить неочевидные связи, например, какие приложения часто используются с определенных IP-адресов или какие каналы приводят к установке конкретных приложений."
        )
    with cat_settings_col3:
        top_n_categories = st.slider(
            "Количество топ элементов", 5, 30, 10, key="top_n_categories",
            help="**Что это?** Определяет, сколько наиболее частых значений для Категории 1 будет отображено на гистограммах топов.\n\n**Зачем он нужен?** Ограничивает вывод на графиках наиболее популярными значениями, чтобы избежать перегруженности и улучшить читаемость.\n\n**Как им пользоваться?** Передвиньте слайдер, чтобы выбрать желаемое число. Гистограммы для Категории 1 обновятся.\n\n**Чем он полезен?** Позволяет сфокусироваться на самых значимых сегментах внутри выбранной категории."
        )
    
    cat_analysis_df = current_df 
    
    st.subheader(f"Статистика по категории: {cat1}")
    stats_cols = st.columns(2)
    
    with stats_cols[0]:
        if not cat_analysis_df.empty and cat1 in cat_analysis_df.columns:
            top_cats = cat_analysis_df[cat1].value_counts().nlargest(top_n_categories)
            if not top_cats.empty:
                fig_top = go.Figure(data=[
                    go.Bar(
                        x=top_cats.index.astype(str),
                        y=top_cats.values,
                        marker_color=COLORS['primary'],
                        text=top_cats.values,
                        textposition='auto'
                    )
                ])
                fig_top.update_layout(
                    title=f'Топ-{top_n_categories} по количеству кликов ({cat1})',
                    plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['paper_bgcolor'],
                    font=dict(color=COLORS['text']),
                    height=450,
                    xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], type='category'),
                    yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
                )
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.info(f"Нет данных для категории {cat1} для отображения топа по кликам.")
        else:
            st.info(f"Категория {cat1} не найдена или нет данных.")
    
    with stats_cols[1]:
        if not cat_analysis_df.empty and cat1 in cat_analysis_df.columns and 'is_attributed' in cat_analysis_df.columns:
            avg_fraud = cat_analysis_df.groupby(cat1)['is_attributed'].mean().nlargest(top_n_categories)
            if not avg_fraud.empty:
                fig_avg = go.Figure(data=[
                    go.Bar(
                        x=avg_fraud.index.astype(str),
                        y=avg_fraud.values,
                        marker_color=COLORS['secondary'],
                        text=[f"{v:.1%}" for v in avg_fraud.values],
                        textposition='auto'
                    )
                ])
                fig_avg.update_layout(
                    title=f'Топ-{top_n_categories} по ср. вероятности фрода ({cat1})',
                    plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['paper_bgcolor'],
                    font=dict(color=COLORS['text']),
                    height=450,
                    xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], type='category'),
                    yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], range=[0,1])
                )
                st.plotly_chart(fig_avg, use_container_width=True)
            else:
                st.info(f"Нет данных для категории {cat1} для отображения топа по фроду.")
        else:
            st.info(f"Категория {cat1} или колонка 'is_attributed' не найдена, или нет данных.")

    # Анализ связей между категориями с настройками
    st.markdown(f'<div class="section-header">🔗 Связи между категориями: {cat1.upper()} и {cat2.upper()}</div>', unsafe_allow_html=True)
    
    # Настройки для тепловой карты
    # Убран выбор цветовой схемы, используется 'RdYlBu' по умолчанию
    heatmap_settings_col1, heatmap_settings_col2 = st.columns(2)
    # with heatmap_settings_col1:
    #     color_scales = ['Viridis', 'Cividis', 'Plasma', 'Blues', 'Greens', 'Reds', 'RdYlBu']
    #     selected_color_scale = st.selectbox("Цветовая схема", color_scales, index=6, key="heatmap_color_scale")
    with heatmap_settings_col1: # Ранее была col2
        heatmap_height = st.selectbox(
            "Высота карты", [400, 500, 600, 700], index=1, key="heatmap_height_cat",
            help="**Что это?** Настройка высоты отображаемой тепловой карты в пикселях.\n\n**Зачем он нужен?** Позволяет адаптировать размер карты под количество данных и разрешение экрана для лучшего визуального восприятия.\n\n**Как им пользоваться?** Выберите желаемую высоту из списка. Тепловая карта изменит свой размер.\n\n**Чем он полезен?** Улучшает читаемость и детализацию тепловой карты, особенно если одна из категорий имеет много уникальных значений."
        )
    with heatmap_settings_col2: # Ранее была col3
        show_annotations = st.checkbox(
            "Показать значения", True, key="show_annotations_cat",
            help="**Что это?** Опция для отображения числовых значений (количество пересечений) непосредственно в ячейках тепловой карты.\n\n**Зачем он нужен?** Предоставляет точные данные о количестве совместных кликов для каждой пары категорий.\n\n**Как им пользоваться?** Установите или снимите флажок. Тепловая карта обновится, показывая или скрывая числовые аннотации.\n\n**Чем он полезен?** Облегчает точную интерпретацию данных на тепловой карте без необходимости наводить курсор на каждую ячейку."
        )
    
    default_heatmap_color_scale = 'RdYlBu' # Одна схема по умолчанию

    if not cat_analysis_df.empty and cat1 in cat_analysis_df.columns and cat2 in cat_analysis_df.columns:
        if cat1 == cat2:
            st.info(f"Для тепловой карты выберите различные категории.")
        else:
            # Ограничиваем количество категорий для читаемости
            top_cat1_values = cat_analysis_df[cat1].value_counts().head(15).index
            top_cat2_values = cat_analysis_df[cat2].value_counts().head(15).index
            
            filtered_df = cat_analysis_df[
                cat_analysis_df[cat1].isin(top_cat1_values) & 
                cat_analysis_df[cat2].isin(top_cat2_values)
            ]
            
            pivot = pd.crosstab(filtered_df[cat1], filtered_df[cat2])
            if not pivot.empty:
                fig_heatmap = px.imshow(
                    pivot,
                    title=None, # Убираем заголовок, так как он есть в markdown
                    color_continuous_scale=default_heatmap_color_scale, # Используем схему по умолчанию
                    text_auto=show_annotations
                )
                fig_heatmap.update_layout(
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['paper_bgcolor'],
                    font=dict(color=COLORS['text']),
                    height=heatmap_height,
                    xaxis_type='category', # Явное указание типа оси X
                    yaxis_type='category'  # Явное указание типа оси Y
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info(f"Нет данных для построения тепловой карты между {cat1} и {cat2}.")
    else:
        st.info(f"Одна или обе выбранные категории ({cat1}, {cat2}) не найдены или нет данных.")

    # Временной анализ с настройками
    st.markdown(f'<div class="section-header">⏳ Временной анализ для категории: {cat1.upper()}</div>', unsafe_allow_html=True)
    
    # Настройки для временного анализа
    # Убран выбор типа визуализации, используется Box plot по умолчанию
    time_analysis_col1 = st.columns(1)[0] # Одна колонка для группировки
    with time_analysis_col1:
        time_grouping = st.selectbox(
            "Группировка по времени", ["Часы", "Дни недели", "Дни месяца"],
            index=0, key="time_grouping_cat",
            help="**Что это?** Определяет, как будут агрегированы данные для временного анализа фрода в разрезе Категории 1 (часы суток, дни недели или дни месяца).\n\n**Зачем он нужен?** Помогает выявить временные закономерности и пики фродовой активности для различных значений Категории 1.\n\n**Как им пользоваться?** Выберите один из вариантов группировки. Box plot ниже покажет распределение вероятности фрода для топ-8 значений Категории 1 в соответствии с выбранной временной единицей.\n\n**Чем он полезен?** Позволяет обнаружить, например, определенные часы или дни недели, когда конкретные IP или приложения демонстрируют аномально высокую фродовую активность."
        )
    # with time_analysis_col2:
    #     viz_type_time = st.selectbox("Тип визуализации", ["Box plot", "Violin plot", "Line plot"], 
    #                                 index=0, key="viz_type_time_cat")
    
    default_viz_type_time_cat = "Box plot" # Один тип визуализации по умолчанию

    if not cat_analysis_df.empty and cat1 in cat_analysis_df.columns and 'click_time' in cat_analysis_df.columns and 'is_attributed' in cat_analysis_df.columns:
        temp_df_time_cat = cat_analysis_df.copy()
        
        if time_grouping == "Часы":
            temp_df_time_cat['time_unit'] = temp_df_time_cat['click_time'].dt.hour
            x_title = "Час дня"
        elif time_grouping == "Дни недели":
            temp_df_time_cat['time_unit'] = temp_df_time_cat['click_time'].dt.day_name()
            x_title = "День недели"
        else:  # Дни месяца
            temp_df_time_cat['time_unit'] = temp_df_time_cat['click_time'].dt.day
            x_title = "День месяца"
        
        # Ограичиваем количество категорий
        top_cat1_for_time = temp_df_time_cat[cat1].value_counts().nlargest(8).index
        df_for_time_plot = temp_df_time_cat[temp_df_time_cat[cat1].isin(top_cat1_for_time)]

        if not df_for_time_plot.empty:
            # Используем default_viz_type_time_cat вместо переменной viz_type_time
            if default_viz_type_time_cat == "Box plot":
                fig_time_cat = px.box(
                    df_for_time_plot,
                    x='time_unit',
                    y='is_attributed',
                    color=cat1,
                    title=f'Распределение P(фрод) по {time_grouping.lower()} и топ-8 значениям {cat1}',
                    color_discrete_sequence=COLORS['pie_colors']
                )
            elif default_viz_type_time_cat == "Violin plot": # Эта ветка теперь не будет выполняться, но оставим для целостности, если решим вернуть
                fig_time_cat = px.violin(
                    df_for_time_plot,
                    x='time_unit',
                    y='is_attributed',
                    color=cat1,
                    title=f'Распределение P(фрод) по {time_grouping.lower()} и топ-8 значениям {cat1}',
                    color_discrete_sequence=COLORS['pie_colors']
                )
            else:  # Line plot - Эта ветка также не будет выполняться
                agg_time_cat = df_for_time_plot.groupby(['time_unit', cat1])['is_attributed'].mean().reset_index()
                fig_time_cat = px.line(
                    agg_time_cat,
                    x='time_unit',
                    y='is_attributed',
                    color=cat1,
                    title=f'Средняя P(фрод) по {time_grouping.lower()} и топ-8 значениям {cat1}',
                    color_discrete_sequence=COLORS['pie_colors'],
                    markers=True
                )
                
            fig_time_cat.update_layout(
                plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['paper_bgcolor'],
                font=dict(color=COLORS['text']),
                height=500,
                xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], title=x_title),
                yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], range=[0,1], title="Вероятность фрода")
            )
            st.plotly_chart(fig_time_cat, use_container_width=True)
        else:
            st.info(f"Нет данных для временного анализа категории {cat1}.")
    else:
        st.info(f"Необходимые колонки ({cat1}, click_time, is_attributed) отсутствуют или нет данных.")

# --- Связи/Графы ---
# with tabs[2]:
with tabs[2]:
    st.header("Анализ сетей мошенничества")
    
    # Подробное объяснение функциональности
    st.info("""
    **Назначение инструмента:**
    
    Данный граф помогает выявить мошеннические сети - связи между подозрительными IP-адресами, устройствами, приложениями и каналами.
    
    **Принцип работы:**
    - **Узлы** представляют отдельные сущности (IP-адреса, устройства, приложения)  
    - **Связи** отображают совместную активность между сущностями
    - **Цвет узла** показывает уровень мошенничества (красный = высокий риск)
    - **Размер узла** отражает активность или важность в сети
    - **Толщина связи** пропорциональна количеству взаимодействий
    """)
    
    # Настройки в логичном порядке
    st.subheader("Настройки анализа")
    
    # Основные настройки
    settings_col1, settings_col2, settings_col3, settings_col4 = st.columns(4)
    
    with settings_col1:
        st.markdown("**Выбор данных для анализа:**")
        # Получаем список всех колонок
        if not current_df.empty:
            all_columns = current_df.columns.tolist()
            excluded_cols_for_graph = ['click_id', 'click_time']
            graph_node_options = [col for col in all_columns if col not in excluded_cols_for_graph]
        else:
            graph_node_options = ['ip', 'app', 'device', 'channel']
        
        graph_node1_attr = st.selectbox("Тип узлов A", graph_node_options, 
                                       index=graph_node_options.index('ip') if 'ip' in graph_node_options else 0,
                                       key="graph_node1",
                                       help="**Что это?** Первый тип сущностей (например, IP-адрес, приложение), которые будут представлены как узлы на графе.\n\n**Зачем он нужен?** Определяет одну из категорий данных для анализа их связей с другим типом сущностей (Тип узлов B).\n\n**Как им пользоваться?** Выберите из списка атрибут, который хотите анализировать. Например, 'ip'.\n\n**Чем он полезен?** Позволяет визуализировать, как выбранные сущности (например, IP-адреса) взаимодействуют с другими (например, устройствами).")
        graph_node2_attr = st.selectbox("Тип узлов B", graph_node_options,
                                       index=graph_node_options.index('device') if 'device' in graph_node_options else 1,
                                       key="graph_node2",
                                       help="**Что это?** Второй тип сущностей, которые будут представлены как узлы и связаны с Типом узлов A.\n\n**Зачем он нужен?** Определяет вторую категорию данных для анализа их взаимных связей с Типом узлов A.\n\n**Как им пользоваться?** Выберите атрибут, отличный от Типа узлов A. Например, если Узлы А - это 'ip', здесь можно выбрать 'device'.\n\n**Чем он полезен?** Позволяет исследовать взаимосвязи между двумя различными типами данных, например, какие устройства (Узлы B) использовались с определенных IP-адресов (Узлы A).")
    
    with settings_col2:
        st.markdown("**Параметры визуализации:**")
        graph_dimension = st.radio("Режим отображения", ('2D (быстрый)', '3D (интерактивный)'), index=0, 
                                   key="graph_dim",
                                   help="**Что это?** Выбор между двухмерным и трехмерным представлением графа связей.\n\n**Зачем он нужен?** 2D-режим обычно работает быстрее и проще для восприятия на больших графах. 3D-режим предлагает более наглядное интерактивное исследование структуры сети, позволяя вращать граф.\n\n**Как им пользоваться?** Выберите желаемый режим. 2D графы строятся быстрее.\n\n**Чем он полезен?** Предоставляет выбор между скоростью и наглядностью в зависимости от задачи анализа и размера графа.")
        
        layout_options = {
            'Органичное (рекомендуется)': 'spring',
            'Круговое расположение': 'circular', 
            'Сбалансированное': 'kamada_kawai',
            'Случайное': 'random'
        }
        selected_layout = st.selectbox("Алгоритм размещения", list(layout_options.keys()), 
                                       key="graph_layout",
                                       help="**Что это?** Алгоритм, который определяет, как узлы и связи будут расположены на графе.\n\n**Зачем он нужен?** Разные алгоритмы могут лучше подсвечивать различные структурные особенности графа (кластеры, центральные узлы и т.д.).\n\n**Как им пользоваться?** Выберите один из алгоритмов:\n- **Органичное (spring):** Часто дает интуитивно понятное расположение, где связанные узлы притягиваются друг к другу.\n- **Круговое (circular):** Размещает узлы по кругу.\n- **Сбалансированное (kamada_kawai):** Пытается минимизировать пересечения ребер и сбалансировать расстояния.\n- **Случайное (random):** Простое случайное размещение.\n\n**Чем он полезен?** Позволяет экспериментировать с представлением графа для наилучшего выявления паттернов и аномалий.")
        layout_algorithm = layout_options[selected_layout]
    
    with settings_col3:
        st.markdown("**Фильтрация данных:**")
        
        # Режимы фильтрации с логичными названиями
        filter_modes = {
            "Все данные (обзор)": "all",
            "Все данные за выбранный период": "all_period",
            "Только мошеннические связи": "fraud_only", 
            "Топ подозрительных узлов": "top_fraud",
            "Аномальные временные периоды": "time_clusters"
        }
        
        data_filter_mode_display = st.selectbox("Режим фильтрации", list(filter_modes.keys()),
                                                help="**Что это?** Способ предварительной фильтрации данных перед построением графа.\n\n**Зачем он нужен?** Позволяет сфокусировать анализ на определенных аспектах: общем обзоре, только фродовых событиях, наиболее активных узлах или временных аномалиях.\n\n**Как им пользоваться?** Выберите режим из списка. В зависимости от выбора могут появиться дополнительные настройки (например, порог фрода или размер выборки).\n- **Все данные (обзор):** Случайная выборка для быстрого просмотра общей структуры.\n- **Все данные за выбранный период:** Анализ всех данных в текущем временном диапазоне (может быть медленно для больших объемов).\n- **Только мошеннические связи:** Показывает связи, где средняя вероятность фрода выше заданного порога.\n- **Топ подозрительных узлов:** Фокусируется на связях, исходящих от узлов с высокой вероятностью фрода и активностью.\n- **Аномальные временные периоды:** Выделяет связи, активные в периоды с аномально высокой фродовой активностью.\n\n**Чем он полезен?** Направляет анализ и помогает эффективно работать с большими объемами данных, выделяя наиболее интересные или подозрительные сегменты.")
        data_filter_mode = filter_modes[data_filter_mode_display]
        
        # Динамические настройки
        if data_filter_mode == "fraud_only":
            fraud_threshold = st.slider("Порог вероятности фрода", 0.0, 1.0, 0.3, 0.05, 
                                       help="**Что это?** Минимальная средняя вероятность фрода для связи, чтобы она была включена в граф в режиме 'Только мошеннические связи'.\n\n**Зачем он нужен?** Отфильтровывает связи с низкой вероятностью фрода.\n\n**Как им пользоваться?** Установите порог. Только те связи, где усредненная вероятность фрода по всем кликам этой связи выше этого значения, будут показаны.\n\n**Чем он полезен?** Помогает сфокусироваться на наиболее очевидных фродовых взаимодействиях.")
        elif data_filter_mode == "top_fraud":
            top_count = st.slider("Количество топ узлов", 5, 50, 15,
                                 help="**Что это?** Количество наиболее подозрительных узлов (Тип А), связи которых будут анализироваться в режиме 'Топ подозрительных узлов'.\n\n**Зачем он нужен?** Ограничивает анализ наиболее проблемными узлами.\n\n**Как им пользоваться?** Выберите количество узлов. Система найдет узлы с самой высокой средней вероятностью фрода и достаточным количеством кликов, и построит граф их связей.\n\n**Чем он полезен?** Эффективно выявляет эпицентры фродовой активности.")
        elif data_filter_mode == "time_clusters":
            time_window = st.slider("Временное окно (часы)", 1, 12, 3,
                                   help="**Что это?** Размер временного интервала в часах для группировки событий при поиске аномальных периодов в режиме 'Аномальные временные периоды'.\n\n**Зачем он нужен?** Определяет гранулярность анализа временных аномалий.\n\n**Как им пользоваться?** Установите длительность окна. Система сгруппирует события по этим окнам и выявит те, где фродовая активность была значительно выше средней.\n\n**Чем он полезен?** Помогает обнаружить кратковременные всплески фрода или скоординированные по времени атаки.")
        elif data_filter_mode == "all":
            sample_size = st.slider("Размер выборки", 1000, 5000, 2000, 250,
                                   help="**Что это?** Количество случайных записей, которые будут использованы для построения графа в режиме 'Все данные (обзор)'.\n\n**Зачем он нужен?** Ускоряет построение графа для общего ознакомления со структурой связей на больших датасетах.\n\n**Как им пользоваться?** Выберите размер выборки. Из всего набора данных будет взято случайное подмножество указанного размера.\n\n**Чем он полезен?** Дает быстрое представление о связях без необходимости обрабатывать все данные.")
        # Для режима all_period не нужны дополнительные параметры
    
    with settings_col4:
        st.markdown("**Настройки отображения:**")
        min_connections = st.slider("Минимум связей", 1, 10, 2,
                                   help="**Что это?** Минимальное количество совместных событий (кликов) между двумя узлами, чтобы связь между ними была отображена на графе.\n\n**Зачем он нужен?** Отфильтровывает слабые или случайные связи, оставляя только более значимые взаимодействия.\n\n**Как им пользоваться?** Установите минимальное число. Связи с меньшим числом кликов будут скрыты. Узлы, оставшиеся без связей, также будут удалены.\n\n**Чем он полезен?** Упрощает граф, делая его более читаемым и акцентируя внимание на сильных взаимодействиях.")
        max_nodes = st.slider("Максимум узлов", 20, 200, 50, 10,
                             help="**Что это?** Максимальное количество узлов, отображаемых на графе.\n\n**Зачем он нужен?** Ограничивает сложность графа для улучшения производительности и читаемости, особенно на больших и плотных сетях.\n\n**Как им пользоваться?** Установите лимит. Если после фильтрации в графе окажется больше узлов, будут показаны только самые центральные (важные) из них.\n\n**Чем он полезен?** Позволяет работать с потенциально очень большими графами, автоматически фокусируясь на их наиболее значимой части.")
        
        show_labels = st.checkbox("Показывать подписи узлов", True,
                                 help="**Что это?** Отображение текстовых меток (значений) рядом с узлами на графе.\n\n**Зачем он нужен?** Позволяет идентифицировать конкретные узлы прямо на визуализации.\n\n**Как им пользоваться?** Установите флажок для отображения или снимите для скрытия. Для длинных значений метки могут быть сокращены.\n\n**Чем он полезен?** Улучшает понимание того, какие именно сущности участвуют во взаимодействиях. Может быть полезно отключить на очень плотных графах для лучшей читаемости структуры.")
        analyze_communities = st.checkbox("Обнаружить группы", False,
                                         help="**Что это?** Автоматический поиск и выделение цветом сообществ (кластеров) в графе.\n\n**Зачем он нужен?** Помогает выявить группы тесно связанных между собой узлов, которые могут представлять собой отдельные мошеннические схемы или группы с общим поведением.\n\n**Как им пользоваться?** Установите флажок. Если алгоритм обнаружит сообщества, узлы будут окрашены в соответствии с принадлежностью к группе, и информация о группе появится при наведении.\n\n**Чем он полезен?** Ускоряет выявление скрытых структур и группировок в сети без необходимости ручного анализа всех связей.")

    # Кнопки управления
    st.divider()
    
    col_btn1, col_btn2, col_btn3 = st.columns([3, 1, 1])
    
    with col_btn1:
        analyze_button = st.button("ПОСТРОИТЬ ГРАФ СВЯЗЕЙ", type="primary", use_container_width=True)
    
    with col_btn2:
        if st.button("Сброс настроек", help="Сбросить все настройки и начать заново"):
            st.session_state['graph_built'] = False
            st.rerun()
    
    with col_btn3:
        help_button = st.button("Справка", help="Показать подробную инструкцию")
    
    if help_button:
        st.expander("Подробная инструкция", expanded=True).write("""
        **Пошаговое руководство:**
        
        **Шаг 1: Выбор типов узлов**
        - Рекомендуется начать с анализа "IP ↔ устройства"
        - Это покажет какие IP-адреса связаны с какими устройствами
        
        **Шаг 2: Настройка фильтрации**
        - "Все данные" - для общего обзора сети
        - "Мошеннические связи" - для поиска конкретных угроз  
        - "Топ подозрительных" - для фокуса на главных проблемах
        - "Аномальные периоды" - для поиска необычной активности
        
        **Шаг 3: Настройка лимитов**
        - Начните с небольших значений (минимум связей = 1-2, максимум узлов = 30-50)
        - Постепенно увеличивайте для более детального анализа
        
        **Интерпретация результатов:**
        - **Красные узлы** = высокий уровень мошенничества, требуют немедленного внимания
        - **Большие узлы** = высокая активность, могут быть ключевыми в схеме
        - **Толстые связи** = много взаимодействий, указывают на тесную связь
        - **Изолированные группы** = отдельные мошеннические схемы
        - **Плотные кластеры** = возможные ботнеты или координированные атаки
        """)
    
    # Проверки корректности
    if current_df.empty:
        st.error("**Ошибка:** Нет данных для анализа. Проверьте фильтры времени в боковой панели.")
    elif graph_node1_attr not in current_df.columns or graph_node2_attr not in current_df.columns:
        st.error(f"**Ошибка:** Отсутствуют необходимые колонки данных: {graph_node1_attr} или {graph_node2_attr}")
    elif graph_node1_attr == graph_node2_attr:
        st.warning("**Предупреждение:** Выберите разные типы узлов для анализа связей между ними.")
    else:
        # Запуск анализа
        if analyze_button or st.session_state.get('graph_built', False):
            st.session_state['graph_built'] = True
            
            # Применяем выбранную логику фильтрации данных
            with st.spinner('Анализируем данные и строим граф...'):
                
                if data_filter_mode == "all":
                    graph_data = current_df.sample(n=min(sample_size, len(current_df)), random_state=42)
                    st.success(f"**Режим анализа:** Все данные (обзор) - обработано {len(graph_data):,} записей")
                elif data_filter_mode == "all_period":
                    graph_data = current_df.copy()
                    st.success(f"**Режим анализа:** Все данные за выбранный период - обработано {len(graph_data):,} записей")
                    
                elif data_filter_mode == "fraud_only":
                    graph_data = current_df[current_df['is_attributed'] > fraud_threshold]
                    if graph_data.empty:
                        st.error(f"Нет записей с уровнем фрода выше {fraud_threshold:.1%}. Рекомендация: понизьте порог или выберите режим 'Все данные'.")
                        st.stop()
                    st.success(f"**Режим анализа:** Мошеннические связи - найдено {len(graph_data):,} записей с фродом > {fraud_threshold:.1%}")
                    
                elif data_filter_mode == "top_fraud":
                    # Найдем топ узлов по фроду
                    node1_fraud_stats = current_df.groupby(graph_node1_attr)['is_attributed'].agg(['mean', 'count']).reset_index()
                    node1_fraud_stats = node1_fraud_stats[
                        (node1_fraud_stats['mean'] > 0.2) & 
                        (node1_fraud_stats['count'] >= 3)
                    ].nlargest(top_count, 'mean')
                    
                    if node1_fraud_stats.empty:
                        st.error("Не найдено подозрительных узлов с достаточной активностью. Рекомендация: попробуйте режим 'Все данные' или 'Мошеннические связи'.")
                        st.stop()
                    
                    top_node1_values = node1_fraud_stats[graph_node1_attr].tolist()
                    graph_data = current_df[current_df[graph_node1_attr].isin(top_node1_values)]
                    st.success(f"**Режим анализа:** Топ подозрительных - анализируется {len(graph_data):,} записей от {len(top_node1_values)} самых проблемных узлов")
                    
                else:  # time_clusters
                    # Временные кластеры
                    current_df_temp = current_df.copy()
                    current_df_temp['time_group'] = current_df_temp['click_time'].dt.floor(f'{time_window}h')  # Исправлено с 'H' на 'h'
                    time_stats = current_df_temp.groupby('time_group').agg({
                        'is_attributed': ['count', 'mean']
                    }).reset_index()
                    time_stats.columns = ['time_group', 'fraud_count', 'fraud_rate']
                    
                    # Находим аномальные временные окна
                    fraud_rate_threshold = time_stats['fraud_rate'].quantile(0.75)
                    suspicious_times = time_stats[time_stats['fraud_rate'] > fraud_rate_threshold]['time_group'].tolist()
                    
                    if not suspicious_times:
                        st.error("Не обнаружено аномальных временных периодов. Рекомендация: попробуйте другой режим анализа или измените размер временного окна.")
                        st.stop()
                    
                    graph_data = current_df_temp[current_df_temp['time_group'].isin(suspicious_times)]
                    st.success(f"**Режим анализа:** Аномальные периоды - найдено {len(graph_data):,} записей из {len(suspicious_times)} подозрительных временных окон")

                if graph_data.empty:
                    st.error("**Результат фильтрации:** Нет данных после применения выбранных фильтров. Рекомендация: измените параметры фильтрации.")
                    st.stop()

                # Создание графа с улучшенной логикой
                G = nx.Graph()
                edge_stats = {}
                
                # Собираем статистику связей
                for _, row in graph_data.iterrows():
                    if pd.isna(row[graph_node1_attr]) or pd.isna(row[graph_node2_attr]):
                        continue
                        
                    u_val = str(row[graph_node1_attr])[:20]  # Ограничиваем длину
                    v_val = str(row[graph_node2_attr])[:20]
                    u = f"{graph_node1_attr}:{u_val}"
                    v = f"{graph_node2_attr}:{v_val}"
                    
                    edge_key = (u, v) if u < v else (v, u)
                    if edge_key not in edge_stats:
                        edge_stats[edge_key] = {
                            'count': 0, 
                            'fraud_sum': 0, 
                            'fraud_values': [],
                            'times': []
                        }
                    
                    edge_stats[edge_key]['count'] += 1
                    edge_stats[edge_key]['fraud_sum'] += row['is_attributed']
                    edge_stats[edge_key]['fraud_values'].append(row['is_attributed'])
                    edge_stats[edge_key]['times'].append(row['click_time'])
                
                # Добавляем ребра в граф
                for (u, v), stats in edge_stats.items():
                    if stats['count'] >= min_connections:
                        avg_fraud = stats['fraud_sum'] / stats['count']
                        time_span = (max(stats['times']) - min(stats['times'])).total_seconds() / 3600
                        
                        G.add_edge(u, v, 
                                  weight=stats['count'],
                                  avg_fraud=avg_fraud,
                                  fraud_variance=np.var(stats['fraud_values']),
                                  time_span=time_span)

                # Удаляем изолированные узлы
                isolated_nodes = list(nx.isolates(G))
                G.remove_nodes_from(isolated_nodes)

                if G.number_of_nodes() == 0:
                    st.error("**Результат построения:** Граф не содержит узлов. Рекомендации: уменьшите 'Минимум связей' до 1 или измените параметры фильтрации.")
                    st.stop()

                # Ограничиваем количество узлов
                if G.number_of_nodes() > max_nodes:
                    degree_cent = nx.degree_centrality(G)
                    top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                    nodes_to_keep = [node[0] for node in top_nodes]
                    G = G.subgraph(nodes_to_keep).copy()
                    st.info(f"**Оптимизация отображения:** Показаны {len(nodes_to_keep)} наиболее важных узлов из {G.number_of_nodes() + len(isolated_nodes)} найденных")

                # Вычисление метрик
                degree_centrality = nx.degree_centrality(G)
                try:
                    closeness_centrality = nx.closeness_centrality(G) if G.number_of_nodes() > 1 else {}
                    betweenness_centrality = nx.betweenness_centrality(G) if G.number_of_nodes() > 2 else {}
                except:
                    closeness_centrality = {}
                    betweenness_centrality = {}

                # Анализ сообществ
                communities = {}
                if analyze_communities and G.number_of_nodes() > 3:
                    try:
                        import networkx.algorithms.community as nx_comm
                        communities_result = nx_comm.greedy_modularity_communities(G)
                        for i, community in enumerate(communities_result):
                            for node in community:
                                communities[node] = i
                        st.info(f"**Анализ сообществ:** Обнаружено {len(communities_result)} групп тесно связанных узлов")
                    except Exception as e:
                        st.warning(f"Не удалось выполнить анализ сообществ: {e}")

                # Добавляем атрибуты к узлам
                for node in G.nodes():
                    try:
                        node_type, node_value = node.split(':', 1)
                        
                        # Безопасный поиск связанных кликов
                        try:
                            if node_value.replace('.','').replace('-','').isdigit():
                                related_clicks = current_df[current_df[node_type] == float(node_value)]
                            else:
                                related_clicks = current_df[current_df[node_type] == node_value]
                        except:
                            related_clicks = pd.DataFrame()
                        
                        G.nodes[node].update({
                            'type': node_type,
                            'value': node_value,
                            'click_count': len(related_clicks),
                            'avg_fraud': related_clicks['is_attributed'].mean() if not related_clicks.empty else 0,
                            'max_fraud': related_clicks['is_attributed'].max() if not related_clicks.empty else 0,
                            'degree_centrality': degree_centrality.get(node, 0),
                            'closeness_centrality': closeness_centrality.get(node, 0),
                            'betweenness_centrality': betweenness_centrality.get(node, 0),
                            'community': communities.get(node, 0)
                        })
                    except Exception as e:
                        # Если что-то пошло не так с узлом, задаем безопасные значения
                        G.nodes[node].update({
                            'type': 'unknown',
                            'value': str(node),
                            'click_count': 0,
                            'avg_fraud': 0,
                            'max_fraud': 0,
                            'degree_centrality': 0,
                            'closeness_centrality': 0,
                            'betweenness_centrality': 0,
                            'community': 0
                        })

                # Создание визуализации
                try:
                    dim_val = 3 if '3D' in graph_dimension else 2
                    iterations = 80
                    
                    # Layout
                    if layout_algorithm == 'spring':
                        pos = nx.spring_layout(G, dim=dim_val, seed=42, iterations=iterations, k=1.5/np.sqrt(G.number_of_nodes()))
                    elif layout_algorithm == 'circular':
                        pos_2d = nx.circular_layout(G)
                        pos = {node: (*coords, 0) for node, coords in pos_2d.items()} if dim_val == 3 else pos_2d
                    elif layout_algorithm == 'kamada_kawai':
                        try:
                            pos_2d = nx.kamada_kawai_layout(G)
                            pos = {node: (*coords, 0) for node, coords in pos_2d.items()} if dim_val == 3 else pos_2d
                        except:
                            pos = nx.spring_layout(G, dim=dim_val, seed=42)
                    else:  # random
                        pos = nx.random_layout(G, dim=dim_val, seed=42)

                    # Создание фигуры
                    fig = go.Figure()

                    # Ребра
                    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                    max_weight = max(edge_weights) if edge_weights else 1
                    
                    for u, v, data in G.edges(data=True):
                        if dim_val == 3:
                            x0, y0, z0 = pos[u]
                            x1, y1, z1 = pos[v]
                            coords = ([x0, x1, None], [y0, y1, None], [z0, z1, None])
                        else:
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            coords = ([x0, x1, None], [y0, y1, None])
                        
                        # Цвет и толщина ребра
                        fraud_intensity = data['avg_fraud']
                        line_width = 1 + (data['weight'] / max_weight) * 4
                        edge_color = f"rgba({int(255*fraud_intensity)}, {int(100*(1-fraud_intensity))}, {int(100*(1-fraud_intensity))}, 0.7)"
                        
                        if dim_val == 3:
                            fig.add_trace(go.Scatter3d(
                                x=coords[0], y=coords[1], z=coords[2],
                                mode='lines',
                                line=dict(width=line_width, color=edge_color),
                                hoverinfo='none',
                                showlegend=False
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=coords[0], y=coords[1],
                                mode='lines',
                                line=dict(width=line_width, color=edge_color),
                                hoverinfo='none',
                                showlegend=False
                            ))

                    # Узлы
                    node_x, node_y, node_z = [], [], []
                    node_colors, node_sizes, node_text, node_labels = [], [], [], []

                    for node in G.nodes():
                        if dim_val == 3:
                            x, y, z = pos[node]
                            node_z.append(z)
                        else:
                            x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        
                        node_data = G.nodes[node]
                        
                        # Цвет и размер узла
                        fraud_level = node_data.get('avg_fraud', 0)
                        click_count = node_data.get('click_count', 0)
                        
                        node_colors.append(fraud_level)
                        
                        # Размер узла
                        base_size = 12
                        size_from_clicks = min(np.sqrt(click_count) * 2, 30)
                        size_from_centrality = node_data.get('degree_centrality', 0) * 25
                        node_sizes.append(base_size + size_from_clicks + size_from_centrality)
                        
                        # Детальная информация при наведении
                        hover_text = f"<b>{node_data.get('type', 'unknown').upper()}: {node_data.get('value', 'N/A')}</b><br>"
                        hover_text += f"Кликов в системе: {click_count:,}<br>"
                        hover_text += f"Средняя вероятность фрода: {fraud_level:.1%}<br>"
                        hover_text += f"Максимальная вероятность фрода: {node_data.get('max_fraud', 0):.1%}<br>"
                        hover_text += f"Важность в сети: {node_data.get('degree_centrality', 0):.3f}<br>"
                        hover_text += f"Связей в текущем графе: {G.degree[node]}"
                        if analyze_communities:
                            hover_text += f"<br>Группа сообщества: {node_data.get('community', 0) + 1}"
                        node_text.append(hover_text)
                        
                        # Подписи
                        if show_labels:
                            label_value = str(node_data.get('value', ''))
                            if len(label_value) > 8:
                                label_value = label_value[:8] + '...'
                            node_labels.append(label_value)
                        else:
                            node_labels.append('')

                    # Добавление узлов на граф
                    if dim_val == 3:
                        fig.add_trace(go.Scatter3d(
                            x=node_x, y=node_y, z=node_z,
                            mode='markers+text' if show_labels else 'markers',
                            marker=dict(
                                size=node_sizes,
                                color=node_colors,
                                colorscale='RdYlGn_r',
                                showscale=True,
                                colorbar=dict(title="Уровень<br>фрода", x=1.02),
                                opacity=0.9,
                                line=dict(width=2, color='white'),
                                cmin=0,
                                cmax=1
                            ),
                            text=node_labels,
                            textposition="top center",
                            textfont=dict(size=10, color='white'),
                            hovertext=node_text,
                            hoverinfo='text',
                            showlegend=False
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text' if show_labels else 'markers',
                            marker=dict(
                                size=node_sizes,
                                color=node_colors,
                                colorscale='RdYlGn_r',
                                showscale=True,
                                colorbar=dict(title="Уровень<br>фрода", x=1.02),
                                opacity=0.9,
                                line=dict(width=2, color='white'),
                                cmin=0,
                                cmax=1
                            ),
                            text=node_labels,
                            textposition="top center",
                            textfont=dict(size=10, color='white'),
                            hovertext=node_text,
                            hoverinfo='text',
                            showlegend=False
                        ))

                    # Настройка внешнего вида графа
                    title_text = f"Сеть мошенничества: {graph_node1_attr.upper()} ↔ {graph_node2_attr.upper()}"
                    
                    layout_config = {
                        'title': dict(
                            text=title_text,
                            x=0.5,
                            font=dict(size=16, color=COLORS['text'])
                        ),
                        'plot_bgcolor': COLORS['background'],
                        'paper_bgcolor': COLORS['paper_bgcolor'], 
                        'font': dict(color=COLORS['text'], size=12),
                        'showlegend': False,
                        'margin': dict(t=60, b=30, l=30, r=130),
                        'height': 800,
                        'hovermode': 'closest'
                    }
                    
                    if dim_val == 3:
                        layout_config['scene'] = dict(
                            xaxis=dict(visible=False, showgrid=False),
                            yaxis=dict(visible=False, showgrid=False),
                            zaxis=dict(visible=False, showgrid=False),
                            bgcolor=COLORS['background'],
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        )
                    else:
                        layout_config.update({
                            'xaxis': dict(visible=False, showgrid=False),
                            'yaxis': dict(visible=False, showgrid=False, scaleanchor="x", scaleratio=1)
                        })
                    
                    fig.update_layout(**layout_config)
                    
                    # Отображение графа
                    st.plotly_chart(fig, use_container_width=True, config={
                        'scrollZoom': True, 
                        'displayModeBar': True,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'pan2d'],
                        'displaylogo': False
                    })

                    # Статистика графа
                    st.divider()
                    st.subheader("Статистика построенного графа")
                    
                    # Основные метрики
                    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                    with metric_col1:
                        st.metric("Узлов в графе", G.number_of_nodes())
                    with metric_col2:
                        st.metric("Связей в графе", G.number_of_edges()) 
                    with metric_col3:
                        avg_fraud = np.mean([G[u][v]['avg_fraud'] for u, v in G.edges()]) if G.edges() else 0
                        st.metric("Средний уровень фрода связей", f"{avg_fraud:.1%}")
                    with metric_col4:
                        components = nx.number_connected_components(G)
                        st.metric("Изолированных групп", components)
                    with metric_col5:
                        if analyze_communities and communities:
                            n_communities = len(set(communities.values()))
                            st.metric("Обнаруженных сообществ", n_communities)
                        else:
                            density = nx.density(G)
                            st.metric("Плотность сети", f"{density:.3f}")

                    # Интерпретация результатов
                    st.subheader("Интерпретация результатов")
                    
                    # Анализируем самые проблемные узлы
                    nodes_by_fraud = [(node, G.nodes[node].get('avg_fraud', 0)) for node in G.nodes()]
                    nodes_by_fraud.sort(key=lambda x: x[1], reverse=True)
                    
                    interpretation_col1, interpretation_col2 = st.columns(2)
                    
                    with interpretation_col1:
                        st.markdown("**Анализ угроз:**")
                        if nodes_by_fraud:
                            top_fraud_node = nodes_by_fraud[0]
                            if top_fraud_node[1] > 0.7:
                                st.error(f"**КРИТИЧЕСКАЯ УГРОЗА:** {top_fraud_node[0]} имеет уровень мошенничества {top_fraud_node[1]:.1%}. Рекомендуется немедленная блокировка.")
                            elif top_fraud_node[1] > 0.5:
                                st.warning(f"**ВЫСОКИЙ РИСК:** {top_fraud_node[0]} показывает уровень мошенничества {top_fraud_node[1]:.1%}. Требует пристального внимания.")
                            elif top_fraud_node[1] > 0.3:
                                st.info(f"**УМЕРЕННЫЙ РИСК:** Максимальный уровень фрода составляет {top_fraud_node[1]:.1%}. Ситуация под контролем.")
                            else:
                                st.success(f"**НИЗКИЙ РИСК:** Максимальный уровень фрода {top_fraud_node[1]:.1%}. Нормальная активность.")
                    
                    with interpretation_col2:
                        st.markdown("**Анализ структуры сети:**")
                        if G.number_of_edges() > G.number_of_nodes() * 1.5:
                            st.warning("**ВЫСОКАЯ СВЯЗНОСТЬ:** Обнаружена плотная сеть взаимодействий. Возможна координированная мошенническая деятельность.")
                        elif components > 3:
                            st.info(f"**ФРАГМЕНТИРОВАННАЯ СЕТЬ:** Найдено {components} изолированных групп. Это может указывать на различные независимые схемы мошенничества.")
                        elif components == 1 and G.number_of_nodes() > 20:
                            st.warning("**ЦЕНТРАЛИЗОВАННАЯ СЕТЬ:** Все узлы связаны между собой. Возможна единая мошенническая схема.")
                        else:
                            st.success("**ПРОСТАЯ СТРУКТУРА:** Обычная сетевая активность без признаков сложных мошеннических схем.")

                    # Рекомендации по действиям
                    st.subheader("Рекомендации по действиям")
                    
                    recommendations = []
                    
                    # Анализ топ-3 самых опасных узлов
                    top_3_fraud = nodes_by_fraud[:3]
                    for i, (node, fraud_level) in enumerate(top_3_fraud):
                        if fraud_level > 0.6:
                            recommendations.append(f"**{i+1}. Немедленная блокировка:** {node} (фрод: {fraud_level:.1%})")
                        elif fraud_level > 0.4:
                            recommendations.append(f"**{i+1}. Усиленный мониторинг:** {node} (фрод: {fraud_level:.1%})")
                    
                    # Анализ связей
                    high_fraud_edges = [(u, v, data['avg_fraud']) for u, v, data in G.edges(data=True) if data['avg_fraud'] > 0.5]
                    if high_fraud_edges:
                        recommendations.append(f"**Блокировка связей:** Обнаружено {len(high_fraud_edges)} подозрительных связей с высоким уровнем фрода")
                    
                    # Анализ сообществ
                    if analyze_communities and communities:
                        community_fraud = {}
                        for node, community_id in communities.items():
                            if community_id not in community_fraud:
                                community_fraud[community_id] = []
                            community_fraud[community_id].append(G.nodes[node].get('avg_fraud', 0))
                        
                        for comm_id, fraud_levels in community_fraud.items():
                            avg_community_fraud = np.mean(fraud_levels)
                            if avg_community_fraud > 0.5:
                                recommendations.append(f"**Блокировка сообщества {comm_id + 1}:** Средний уровень фрода {avg_community_fraud:.1%}")
                    
                    if recommendations:
                        for rec in recommendations:
                            st.write(f"• {rec}")
                    else:
                        st.success("**Статус:** Критических угроз не обнаружено. Рекомендуется продолжить мониторинг в обычном режиме.")

                except Exception as e:
                    st.error(f"**Ошибка при создании визуализации:** {e}")
                    st.error("Попробуйте изменить параметры анализа или обратитесь к технической поддержке.")
        else:
            # Показываем инструкции пока граф не построен
            st.info("""
            **Готовность к анализу:**
            
            1. **Выберите типы узлов** для анализа связей (например: IP ↔ устройства)
            2. **Настройте режим фильтрации** в зависимости от задач анализа
            3. **Установите лимиты отображения** для оптимальной производительности
            4. **Нажмите кнопку 'ПОСТРОИТЬ ГРАФ СВЯЗЕЙ'**
            """)
            
            # Показываем превью данных
            st.subheader("Превью доступных данных")
            if not current_df.empty:
                preview_data = current_df[[graph_node1_attr, graph_node2_attr, 'is_attributed']].head(5)
                st.dataframe(preview_data, use_container_width=True)
                
                fraud_rate = (current_df['is_attributed'] > 0.5).mean()
                total_combinations = len(current_df[[graph_node1_attr, graph_node2_attr]].drop_duplicates())
                st.caption(f"**Статистика данных:** {fraud_rate:.1%} записей с высокой вероятностью фрода, {total_combinations:,} уникальных комбинаций {graph_node1_attr}-{graph_node2_attr}")

# --- Корреляции ---
# with tabs[3]:
with tabs[3]:
    st.subheader("Матрица корреляций")
    
    # Настройки для корреляционного анализа
    # Убран выбор цветовой схемы, используется 'RdBu_r' по умолчанию
    corr_settings_col1, corr_settings_col2 = st.columns(2)
    with corr_settings_col1:
        corr_method = st.selectbox("Метод корреляции", ["pearson", "spearman", "kendall"], 
                                  index=0, key="corr_method_main", 
                                  help="**Что это?** Статистический метод для расчета матрицы корреляций между выбранными числовыми признаками.\n\n**Зачем он нужен?** Позволяет оценить степень и направление линейной (Pearson) или монотонной (Spearman, Kendall) связи между различными параметрами данных.\n\n**Как им пользоваться?**\n- **Pearson:** Подходит для оценки линейных связей, чувствителен к выбросам.\n- **Spearman:** Ранговая корреляция, оценивает монотонную связь (не обязательно линейную), менее чувствителен к выбросам.\n- **Kendall:** Также ранговая, рекомендуется для малых выборок или данных с большим количеством одинаковых рангов.\nВыберите метод, и матрица корреляций будет пересчитана.\n\n**Чем он полезен?** Помогает понять, какие признаки изменяются согласованно (положительная корреляция) или в противоположных направлениях (отрицательная), что может указывать на их взаимозависимость или влияние на фрод.")
    # with corr_settings_col2:
    #     corr_color_scale_select = st.selectbox("Цветовая схема", ["RdBu_r", "coolwarm", "viridis", "plasma"], 
    #                                    index=0, key="corr_color_scale_main")
    with corr_settings_col2: # Ранее была col3
        show_corr_values = st.checkbox("Показать значения корреляций", True, key="show_corr_values_main", 
                                       help="**Что это?** Опция для отображения числовых коэффициентов корреляции непосредственно на ячейках матрицы.\n\n**Зачем он нужен?** Позволяет видеть точные значения корреляций в дополнение к цветовому кодированию.\n\n**Как им пользоваться?** Установите или снимите флажок. Матрица обновится, показывая или скрывая числовые значения.\n\n**Чем он полезен?** Облегчает точную интерпретацию силы связи между признаками.")
    
    default_corr_color_scale = 'RdBu_r' # Одна схема по умолчанию
    corr_analysis_df = current_df
    
    # Выбор колонок для корреляционного анализа
    numeric_cols_for_corr = corr_analysis_df.select_dtypes(include=np.number).columns.tolist()
    cols_for_corr_matrix = [col for col in ['ip','app','device','channel','is_attributed'] 
                           if col in corr_analysis_df.columns and pd.api.types.is_numeric_dtype(corr_analysis_df[col])]
    
    # Дополнительные настройки для выбора колонок
    st.write("**Выбор признаков для анализа:**")
    corr_cols_selection = st.multiselect("Выберите признаки", cols_for_corr_matrix, 
                                        default=cols_for_corr_matrix, key="corr_cols_selection",
                                        help="**Что это?** Позволяет выбрать, какие числовые признаки из датасета будут включены в расчет и отображение матрицы корреляций и доступны для диаграммы рассеяния.\n\n**Зачем он нужен?** Дает возможность сфокусировать анализ корреляций на наиболее интересных или релевантных для фрода признаках.\n\n**Как им пользоваться?** Выберите или отмените выбор признаков в списке. Матрица корреляций и доступные опции для диаграммы рассеяния обновятся. Для построения матрицы требуется минимум два признака.\n\n**Чем он полезен?** Помогает избежать перегруженности матрицы ненужными данными и ускоряет анализ, концентрируясь на ключевых параметрах.")
    
    if len(corr_cols_selection) > 1:
        corr = corr_analysis_df[corr_cols_selection].corr(method=corr_method)
        
        fig_corr = px.imshow(
            corr,
            text_auto=show_corr_values,
            color_continuous_scale=default_corr_color_scale, # Используем схему по умолчанию
            title=None, # Убираем заголовок, есть в markdown
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['paper_bgcolor'],
            font=dict(color=COLORS['text']),
            height=600
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Статистика корреляций
        st.subheader("Статистика корреляций")
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            # Топ положительные корреляции
            upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            correlations_flat = upper_triangle.stack().reset_index()
            correlations_flat.columns = ['Признак 1', 'Признак 2', 'Корреляция']
            
            top_positive = correlations_flat.nlargest(5, 'Корреляция')
            st.write("**Топ положительные корреляции:**")
            st.dataframe(top_positive.style.format({'Корреляция': '{:.3f}'}).background_gradient(
                subset=['Корреляция'], cmap='Greens'), use_container_width=True)
        
        with stat_col2:
            # Топ отрицательные корреляции
            top_negative = correlations_flat.nsmallest(5, 'Корреляция')
            st.write("**Топ отрицательные корреляции:**")
            st.dataframe(top_negative.style.format({'Корреляция': '{:.3f}'}).background_gradient(
                subset=['Корреляция'], cmap='Reds'), use_container_width=True)
                
    else:
        st.info("Выберите минимум 2 признака для корреляционного анализа.")

    # Улучшенная диаграмма рассеяния с настройками
    st.subheader("Диаграмма рассеяния")
    
    # Настройки для scatter plot
    # Убран чекбокс "Показать линию тренда", по умолчанию не показывается
    scatter_settings_col1, scatter_settings_col2, scatter_settings_col3 = st.columns(3)
    
    potential_scatter_features = ['ip', 'app', 'device', 'channel', 'is_attributed']
    available_features_for_scatter = [col for col in potential_scatter_features 
                                      if col in corr_analysis_df.columns and 
                                         (pd.api.types.is_numeric_dtype(corr_analysis_df[col]) or 
                                          corr_analysis_df[col].nunique() < 100)]

    if len(available_features_for_scatter) >= 2:
        with scatter_settings_col1:
            x_feature = st.selectbox("Признак X", available_features_for_scatter, 
                                     index=0, key="scatter_x_feat",
                                     help="**Что это?** Признак данных, который будет отложен по горизонтальной оси (X) на диаграмме рассеяния.\n\n**Зачем он нужен?** Является одной из двух переменных для визуального анализа их совместного распределения и выявления возможных зависимостей или кластеров.\n\n**Как им пользоваться?** Выберите признак из списка доступных (числовых или категориальных с небольшим числом уникальных значений). График обновится.\n\n**Чем он полезен?** Помогает визуально оценить, как значения одного признака соотносятся со значениями другого, особенно в контексте уровня фрода (цветовая кодировка точек).")
        with scatter_settings_col2:
            y_feature = st.selectbox("Признак Y", available_features_for_scatter, 
                                     index=min(1, len(available_features_for_scatter)-1), 
                                     key="scatter_y_feat",
                                     help="**Что это?** Признак данных, который будет отложен по вертикальной оси (Y) на диаграмме рассеяния.\n\n**Зачем он нужен?** Является второй из двух переменных для визуального анализа их совместного распределения.\n\n**Как им пользоваться?** Выберите признак из списка, отличный от Признака X. График обновится.\n\n**Чем он полезен?** В паре с Признаком X позволяет исследовать двумерные зависимости и паттерны в данных, подсвеченные уровнем фрода.")
        with scatter_settings_col3:
            scatter_sample_size = st.selectbox("Размер выборки", [1000, 5000, 10000, 20000], 
                                              index=2, key="scatter_sample_size",
                                              help="**Что это?** Максимальное количество точек (событий), которые будут отображены на диаграмме рассеяния.\n\n**Зачем он нужен?** Ограничивает количество отображаемых данных для ускорения отрисовки графика и улучшения его читаемости, особенно при больших объемах исходных данных.\n\n**Как им пользоваться?** Выберите желаемый размер выборки. Если данных больше, будет взято случайное подмножество.\n\n**Чем он полезен?** Обеспечивает приемлемую производительность и наглядность диаграммы рассеяния даже на значительных объемах информации.")
        
        # Дополнительные настройки
        scatter_advanced_col1, scatter_advanced_col2 = st.columns(2) # Убрана третья колонка для линии тренда
        with scatter_advanced_col1:
            scatter_opacity = st.slider("Прозрачность точек", 0.1, 1.0, 0.6, key="scatter_opacity_main", 
                                        help="**Что это?** Степень прозрачности маркеров (точек) на диаграмме рассеяния.\n\n**Зачем он нужен?** Позволяет лучше видеть плотность распределения точек, особенно в областях их сильного перекрытия.\n\n**Как им пользоваться?** Передвиньте слайдер. Значение 1.0 означает полную непрозрачность, меньшие значения делают точки более прозрачными.\n\n**Чем он полезен?** Улучшает визуальное восприятие структуры данных на диаграмме, помогая выявлять кластеры и области с высокой плотностью событий.")
        # with scatter_advanced_col2:
        #     show_trendline_checkbox = st.checkbox("Показать линию тренда", False, key="show_trendline_main")
        with scatter_advanced_col2: # Ранее была col3
            scatter_height = st.selectbox("Высота графика", [400, 500, 600], index=1, key="scatter_height_main", 
                                          help="**Что это?** Высота отображаемой диаграммы рассеяния в пикселях.\n\n**Зачем он нужен?** Позволяет адаптировать размер графика для лучшего визуального восприятия и соответствия другим элементам дашборда.\n\n**Как им пользоваться?** Выберите желаемую высоту из списка. График изменит свой размер.\n\n**Чем он полезен?** Улучшает читаемость и детализацию диаграммы рассеяния.")
        
        # show_trendline = False # По умолчанию линия тренда не показывается

        if x_feature and y_feature and x_feature != y_feature:
            plot_data_scatter = corr_analysis_df[[x_feature, y_feature, 'is_attributed']].copy()
            
            # Преобразование категориальных в числовые
            for col_to_convert in [x_feature, y_feature]:
                 if not pd.api.types.is_numeric_dtype(plot_data_scatter[col_to_convert]):
                    plot_data_scatter[col_to_convert], _ = pd.factorize(plot_data_scatter[col_to_convert])
            
            # Выборка данных
            if len(plot_data_scatter) > scatter_sample_size:
                plot_data_scatter = plot_data_scatter.sample(n=scatter_sample_size, random_state=42)

            # Создание scatter plot без линии тренда по умолчанию
            # if show_trendline: ... else: ... логика упрощена
            fig_scatter = go.Figure(data=go.Scattergl(
                x=plot_data_scatter[x_feature],
                y=plot_data_scatter[y_feature],
                mode='markers',
                marker=dict(
                    color=plot_data_scatter['is_attributed'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar_title='Уровень фрода',
                    opacity=scatter_opacity
                ),
                hovertemplate=f'<b>{x_feature}</b>: %{{x}}<br><b>{y_feature}</b>: %{{y}}<br><b>Фрод</b>: %{{marker.color:.3f}}<extra></extra>'
            ))
            fig_scatter.update_layout(
                title=f'Диаграмма рассеяния: {x_feature} vs {y_feature}',
                xaxis_title=x_feature,
                yaxis_title=y_feature
            )
            
            fig_scatter.update_layout(
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['paper_bgcolor'],
                font=dict(color=COLORS['text']),
                height=scatter_height,
                xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
                yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Статистика для scatter plot
            # if show_trendline: ... (удалена эта часть, т.к. линия тренда убрана по умолчанию)
            #     correlation = plot_data_scatter[x_feature].corr(plot_data_scatter[y_feature])
            #     st.metric("Корреляция между признаками", f"{correlation:.3f}")
                
        elif x_feature == y_feature and x_feature is not None:
            st.info("Выберите разные признаки для осей X и Y.")
    else:
        st.warning("Недостаточно подходящих признаков для построения диаграммы рассеяния.")

# --- Алерты ---
# with tabs[4]:
with tabs[4]:
    st.subheader("Алерт-лист")
    
    # Настройки для алертов
    alert_settings_col1, alert_settings_col2, alert_settings_col3 = st.columns(3)
    with alert_settings_col1:
        alert_custom_threshold = st.slider("Порог для алертов на этой вкладке", 0.0, 1.0, alert_threshold, 0.01,
                                          key="alert_custom_threshold", 
                                          help="**Что это?** Локальный порог вероятности фрода, применяемый только для отображения событий в списке алертов на этой вкладке.\n\n**Зачем он нужен?** Позволяет гибко настраивать чувствительность списка алертов независимо от глобального порога, используемого для других расчетов (например, для общей статистики в сайдбаре или выявления паттернов).\n\n**Как им пользоваться?** Передвиньте слайдер. Список алертов ниже обновится, показывая только события с вероятностью фрода выше этого локального порога.\n\n**Чем он полезен?** Дает возможность исследовать более широкий или, наоборот, более узкий диапазон подозрительных событий на данной вкладке, не меняя глобальные настройки дашборда.")
    with alert_settings_col2:
        alert_sort_by = st.selectbox("Сортировать по", ["Вероятность фрода", "Время", "IP", "Устройство"], 
                                    index=0, key="alert_sort_by",
                                    help="**Что это?** Критерий для упорядочивания списка алертов.\n\n**Зачем он нужен?** Позволяет просматривать алерты в наиболее удобном для анализа порядке.\n\n**Как им пользоваться?** Выберите один из вариантов:\n- **Вероятность фрода:** Алерты с наибольшей вероятностью фрода будут наверху.\n- **Время:** Алерты будут отсортированы по времени их возникновения (обычно самые новые или самые старые, в зависимости от внутренней логики, часто по убыванию времени).\n- **IP/Устройство:** Позволяет сгруппировать алерты по этим признакам (требует соответствующей логики сортировки в коде).\n\n**Чем он полезен?** Ускоряет поиск наиболее критичных или свежих алертов, а также помогает выявлять множественные алерты от одного источника.")
    with alert_settings_col3:
        alerts_per_page = st.selectbox("Алертов на странице", [20, 50, 100, 200], index=1, key="alerts_per_page",
                                      help="**Что это?** Максимальное количество алертов, отображаемых в таблице на одной \"странице\".\n\n**Зачем он нужен?** Управляет объемом выводимой информации для удобства просмотра и производительности.\n\n**Как им пользоваться?** Выберите желаемое количество из списка. Таблица алертов будет показывать не более указанного числа записей (самых верхних после сортировки и фильтрации).\n\n**Чем он полезен?** Помогает избежать вывода слишком длинных списков, делая просмотр более управляемым.")
    
    alerts_df = current_df[current_df['is_attributed'] > alert_custom_threshold]

    if alerts_df.empty:
        st.markdown(f'<div class="info-box">ℹ️ Нет кликов с вероятностью фрода выше {alert_custom_threshold:.1%}.</div>', unsafe_allow_html=True)
    else:
        # Улучшенная статистика по алертам
        st.markdown('<div class="section-header">Статистика по алертам</div>', unsafe_allow_html=True)
        
        # Основные метрики алертов
        alert_metrics_col1, alert_metrics_col2, alert_metrics_col3, alert_metrics_col4 = st.columns(4)
        with alert_metrics_col1:
            st.metric("Всего алертов", len(alerts_df))
        with alert_metrics_col2:
            critical_alerts = (alerts_df['is_attributed'] > 0.8).sum()
            st.metric("Критических (>80%)", critical_alerts)
        with alert_metrics_col3:
            avg_alert_fraud = alerts_df['is_attributed'].mean()
            st.metric("Средний уровень фрода", f"{avg_alert_fraud:.1%}")
        with alert_metrics_col4:
            unique_ips = alerts_df['ip'].nunique()
            st.metric("Уникальных IP", unique_ips)
        
        # Удалены настройки для визуализаций алертов и сами визуализации
        # alert_viz_col1, alert_viz_col2 = st.columns(2)
        # ... (код для alert_chart_type и alert_chart_height удален)
        # 
        # alert_stats_cols = st.columns(3) 
        # ... (код для fig_alerts_by_hour, fig_alerts_by_device, fig_alerts_by_app удален)
        
        # Улучшенная таблица алертов с фильтрацией
        st.markdown('<div class="section-header">Список алертов</div>', unsafe_allow_html=True)
        
        # Дополнительные фильтры для таблицы
        table_filter_col1, table_filter_col2, table_filter_col3 = st.columns(3)
        with table_filter_col1:
            severity_filter = st.selectbox("Фильтр по критичности", 
                                          ["Все", "Критические (>80%)", "Высокие (>60%)", "Средние (>40%)"],
                                          index=0, key="severity_filter",
                                          help="**Что это?** Фильтр для отображения алертов на основе их уровня (вероятности) фрода.\n\n**Зачем он нужен?** Позволяет быстро сфокусироваться на алертах определенной степени опасности.\n\n**Как им пользоваться?** Выберите одну из опций:\n- **Все:** Показывает все алерты, прошедшие основной порог.\n- **Критические (>80%):** Только алерты с P(фрод) > 0.8.\n- **Высокие (>60%):** Только алерты с P(фрод) > 0.6.\n- **Средние (>40%):** Только алерты с P(фрод) > 0.4.\nСписок алертов обновится.\n\n**Чем он полезен?** Упрощает приоритезацию анализа, позволяя сосредоточиться, например, только на самых критических угрозах.")
        with table_filter_col2:
            show_only_unique_ips = st.checkbox("Только уникальные IP", False, key="show_only_unique_ips",
                                              help="**Что это?** Опция для отображения только одного (обычно первого или самого сильного) алерта для каждого уникального IP-адреса.\n\n**Зачем он нужен?** Помогает идентифицировать количество уникальных источников атак или подозрительной активности, а не общее количество событий от них.\n\n**Как им пользоваться?** Установите флажок. Список алертов будет содержать не более одной записи для каждого IP.\n\n**Чем он полезен?** Полезно для оценки масштаба атаки по количеству задействованных IP-адресов и для избежания дублирования информации при первичном анализе.")
        with table_filter_col3:
            highlight_critical = st.checkbox("Подсветить критические", True, key="highlight_critical",
                                            help="**Что это?** Включает или отключает специальную цветовую подсветку для строк в таблице алертов в зависимости от уровня фрода (светофор).\n\n**Зачем он нужен?** Обеспечивает быстрое визуальное выделение наиболее опасных алертов в списке.\n\n**Как им пользоваться?** Установите флажок для включения подсветки (строки с высоким фродом будут красными/желтыми). Снимите, чтобы убрать подсветку.\n\n**Чем он полезен?** Улучшает наглядность таблицы и помогает моментально идентифицировать критические записи.")
        
        # Применяем фильтры
        display_alerts = alerts_df.copy()
        
        if severity_filter == "Критические (>80%)":
            display_alerts = display_alerts[display_alerts['is_attributed'] > 0.8]
        elif severity_filter == "Высокие (>60%)":
            display_alerts = display_alerts[display_alerts['is_attributed'] > 0.6]
        elif severity_filter == "Средние (>40%)":
            display_alerts = display_alerts[display_alerts['is_attributed'] > 0.4]
            
        if show_only_unique_ips:
            display_alerts = display_alerts.drop_duplicates(subset=['ip'])
        
        # Отображение таблицы с улучшенным стилем
        display_count = min(alerts_per_page, len(display_alerts))
        table_data = display_alerts.head(display_count)
        
        if highlight_critical: # Переименовано для ясности, но логика теперь другая
            def apply_traffic_light_style(val):
                # Используем alert_custom_threshold, который выбран на этой вкладке
                traffic_light_info = get_fraud_traffic_light_info(val, alert_custom_threshold)
                return traffic_light_info['style']
            
            styled_table = table_data.style.format({'is_attributed': "{:.3f}"}).applymap(
                apply_traffic_light_style, subset=['is_attributed'])
        else:
            # Если подсветка отключена, просто форматируем, без градиента
            styled_table = table_data.style.format({'is_attributed': "{:.3f}"})
        
        st.dataframe(styled_table, use_container_width=True)
        
        if len(display_alerts) > display_count:
            st.info(f"Показано {display_count} из {len(display_alerts)} алертов. Используйте фильтры для уточнения.")
        
        # Экспорт с настройками
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            export_format = st.selectbox("Формат экспорта", ["CSV", "JSON"], index=0, key="export_format",
                                        help="**Что это?** Выбор формата файла для скачивания данных из таблицы алертов.\n\n**Зачем он нужен?** Позволяет сохранить отфильтрованный список алертов для дальнейшего анализа в других инструментах или для отчетности.\n\n**Как им пользоваться?** Выберите CSV (для табличных редакторов типа Excel) или JSON (для программной обработки). Затем нажмите кнопку скачивания.\n\n**Чем он полезен?** Обеспечивает возможность оффлайн-анализа и интеграции данных с другими системами.")
        with export_col2:
            export_all = st.checkbox("Экспортировать все (не только отображаемые)", False, key="export_all",
                                    help="**Что это?** Определяет, будут ли экспортированы все алерты, соответствующие текущим фильтрам, или только те, что видны на текущей \"странице\" таблицы (ограниченные настройкой \"Алертов на странице\").\n\n**Зачем он нужен?** Предоставляет выбор между полным экспортом отфильтрованных данных и экспортом только видимой части.\n\n**Как им пользоваться?** Установите флажок, если хотите скачать все отфильтрованные алерты. Оставьте снятым, чтобы скачать только отображаемые в таблице строки.\n\n**Чем он полезен?** Позволяет получить полный набор данных по заданным критериям или быстро сохранить текущий срез.")
        
        export_data = display_alerts if export_all else table_data
        
        if export_format == "CSV":
            st.download_button(
                f"Скачать алерты ({export_format})",
                export_data.to_csv(index=False),
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:  # JSON
            st.download_button(
                f"Скачать алерты ({export_format})",
                export_data.to_json(orient='records', date_format='iso'),
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        # Улучшенная секция детального анализа
        st.subheader("Детальный анализ алерта")
        
        if not alerts_df.empty:
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                available_click_ids_alerts = alerts_df['click_id'].unique()
                click_id_alert = st.selectbox(
                    "Выберите click_id для анализа:",
                    options=available_click_ids_alerts,
                    index=0,
                    key="alert_click_id_selector",
                    help="**Что это?** Идентификатор конкретного клика, вызвавшего алерт.\n\n**Зачем он нужен?** Позволяет детально изучить один конкретный подозрительный случай из списка алертов.\n\n**Как им пользоваться?** Выберите ID из списка, чтобы загрузить подробную информацию и связанные события ниже.\n\n**Чем он полезен?** Помогает в ручном расследовании инцидентов, понимании контекста фрода и проверке отдельных транзакций."
                ) 
            
            with detail_col2:
                analysis_depth = st.selectbox("Глубина анализа", ["Базовый", "Расширенный", "Полный"], 
                                             index=1, key="analysis_depth",
                                             help="**Что это?** Уровень детализации при анализе выбранного клика.\n\n**Зачем он нужен?** Позволяет контролировать объем загружаемой и отображаемой информации для связанных событий.\n\n**Как им пользоваться?**\n- **Базовый:** Показывает только основную информацию о самом клике.\n- **Расширенный:** Добавляет сводную статистику по связанным IP и устройствам.\n- **Полный:** Отображает таблицы с первыми 10 связанными событиями по IP и устройству в дополнение к расширенной статистике.\n\n**Чем он полезен?** Дает гибкость: быстрый обзор с 'Базовым' или глубокое погружение с 'Полным' для выявления сложных мошеннических схем.")
            
            # Поиск детальной информации
            click_row = current_df[current_df['click_id'] == click_id_alert] 
            if not click_row.empty:
                with st.expander("Детальная информация о клике", expanded=True):
                    detail_info_col1, detail_info_col2 = st.columns(2)
                    
                    with detail_info_col1:
                        st.write("**Основная информация:**")
                        basic_info = click_row[['click_id', 'click_time', 'is_attributed']].T
                        st.dataframe(basic_info.style.format({'is_attributed': '{:.4f}'}))
                    
                    with detail_info_col2:
                        st.write("**Связанные параметры:**")
                        related_info = click_row[['ip', 'app', 'device', 'channel']].T
                        st.dataframe(related_info)
                
                if analysis_depth in ["Расширенный", "Полный"]:
                    st.subheader("Анализ связанных записей")
                    
                    related_by_ip = get_related_clicks(current_df, click_id_alert, 'ip') 
                    related_by_device = get_related_clicks(current_df, click_id_alert, 'device')
                    
                    related_col1, related_col2 = st.columns(2)
                    
                    with related_col1:
                        st.write(f"**Активность того же IP** ({len(related_by_ip)} записей):")
                        ip_summary = {
                            "Средний фрод": related_by_ip['is_attributed'].mean(),
                            "Максимальный фрод": related_by_ip['is_attributed'].max(),
                            "Количество устройств": related_by_ip['device'].nunique(),
                            "Количество приложений": related_by_ip['app'].nunique()
                        }
                        for key, value in ip_summary.items():
                            if isinstance(value, float):
                                st.metric(key, f"{value:.3f}")
                            else:
                                st.metric(key, value)
                        
                        if analysis_depth == "Полный":
                            related_ip_display = related_by_ip[['click_time', 'is_attributed', 'app', 'device']].head(10)
                            st.dataframe(
                                related_ip_display.style.format({'is_attributed': "{:.3f}"}).background_gradient(
                                    subset=['is_attributed'], cmap='RdYlGn_r'),
                                use_container_width=True
                            )
                    
                    with related_col2:
                        st.write(f"**Активность того же устройства** ({len(related_by_device)} записей):")
                        device_summary = {
                            "Средний фрод": related_by_device['is_attributed'].mean(),
                            "Максимальный фрод": related_by_device['is_attributed'].max(),
                            "Количество IP": related_by_device['ip'].nunique(),
                            "Количество приложений": related_by_device['app'].nunique()
                        }
                        for key, value in device_summary.items():
                            if isinstance(value, float):
                                st.metric(key, f"{value:.3f}")
                            else:
                                st.metric(key, value)
                        
                        if analysis_depth == "Полный":
                            related_device_display = related_by_device[['click_time', 'is_attributed', 'app', 'ip']].head(10)
                            st.dataframe(
                                related_device_display.style.format({'is_attributed': "{:.3f}"}).background_gradient(
                                    subset=['is_attributed'], cmap='RdYlGn_r'),
                                use_container_width=True
                            )
            else:
                st.warning(f"Клик с ID {click_id_alert} не найден в общем датасете.")

# --- Последние события ---
# with tabs[5]:
with tabs[5]:
    st.subheader("Последние события")
    
    # Настройки для последних событий
    events_settings_col1, events_settings_col2, events_settings_col3 = st.columns(3)
    with events_settings_col1:
        events_count = st.selectbox("Количество событий", [50, 100, 200, 500], index=1, key="events_count",
                                   help="Сколько последних событий показывать в таблице.")
    with events_settings_col2:
        events_sort_by = st.selectbox("Сортировать по", ["Времени (новые)", "Времени (старые)", "Уровню фрода"], 
                                     index=0, key="events_sort_by",
                                     help="Выберите порядок сортировки событий: по времени или по уровню фрода.")
    with events_settings_col3:
        events_filter_threshold = st.slider("Минимальный уровень фрода", 0.0, 1.0, 0.0, 0.05, 
                                           key="events_filter_threshold", 
                                           help="Показывать только события с вероятностью фрода выше выбранного значения.")
    
    # Применяем фильтры и сортировку
    recent_events_df = current_df[current_df['is_attributed'] > events_filter_threshold].copy()
    
    if events_sort_by == "Времени (новые)":
        recent_events_df = recent_events_df.sort_values(by='click_time', ascending=False)
    elif events_sort_by == "Времени (старые)":
        recent_events_df = recent_events_df.sort_values(by='click_time', ascending=True)
    else:  # Уровню фрода
        recent_events_df = recent_events_df.sort_values(by='is_attributed', ascending=False)

    if recent_events_df.empty:
        st.markdown(f'<div class="info-box">ℹ️ Нет событий с уровнем фрода выше {events_filter_threshold:.1%} в выбранном временном диапазоне.</div>', unsafe_allow_html=True)
    else:
        # Статистика событий с улучшенными метриками
        st.markdown('<div class="section-header">Статистика событий</div>', unsafe_allow_html=True)
        
        events_metrics_col1, events_metrics_col2, events_metrics_col3, events_metrics_col4 = st.columns(4)
        with events_metrics_col1:
            st.metric("Всего событий", len(recent_events_df))
        with events_metrics_col2:
            high_risk_events = (recent_events_df['is_attributed'] > 0.7).sum()
            st.metric("Высокий риск (>70%)", high_risk_events)
        with events_metrics_col3:
            time_span = (recent_events_df['click_time'].max() - recent_events_df['click_time'].min()).total_seconds() / 3600
            st.metric("Временной охват (часы)", f"{time_span:.1f}")
        with events_metrics_col4:
            events_per_hour = len(recent_events_df) / max(time_span, 1)
            st.metric("События в час", f"{events_per_hour:.1f}")
        
        # Удалены настройки для визуализаций событий и сами визуализации
        # events_viz_col1, events_viz_col2 = st.columns(2)
        # ... (код для events_chart_type и events_chart_height удален)
        # 
        # recent_stats_cols = st.columns(2) 
        # ... (код для fig_recent_by_device и fig_recent_by_app удален)
        
        # Временная динамика событий
        st.markdown('<div class="section-header">Временная динамика событий</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Список событий</div>', unsafe_allow_html=True)
        
        # Дополнительные настройки таблицы
        # Убран выбор цветовой схемы, используется 'RdYlGn_r' по умолчанию
        table_events_col1 = st.columns(1)[0] # Одна колонка для выбора столбцов
        with table_events_col1:
            show_columns = st.multiselect("Отображаемые колонки", 
                                         ['click_id', 'click_time', 'ip', 'app', 'device', 'channel', 'is_attributed'],
                                         default=['click_time', 'ip', 'device', 'is_attributed'], 
                                         key="show_columns_events_main", help="Выберите, какие столбцы показывать в таблице событий. Это позволяет сфокусироваться на нужных параметрах.")
        # with table_events_col2:
        #     color_scheme_select = st.selectbox("Цветовая схема таблицы", ["RdYlGn_r", "Reds", "viridis"], 
        #                                index=0, key="color_scheme_events_main")
        
        default_color_scheme_events = 'RdYlGn_r' # Одна схема по умолчанию

        # Отображение таблицы событий
        # Добавляем .copy() чтобы избежать SettingWithCopyWarning и обеспечить модификацию копии
        display_events = recent_events_df[show_columns].head(events_count).copy() 
        
        # Конвертация click_time в строку перед стилизацией, чтобы избежать проблем с Arrow
        if 'click_time' in display_events.columns:
            display_events['click_time'] = display_events['click_time'].astype(str)

        styled_events = display_events.style.format({'is_attributed': "{:.3f}"} if 'is_attributed' in show_columns else {})
        if 'is_attributed' in show_columns:
            styled_events = styled_events.background_gradient(subset=['is_attributed'], cmap=default_color_scheme_events)
        
        # Удалена следующая строка, которая вызывала AttributeError и была избыточной:
        # # styled_events = styled_events.map(lambda val: px.colors.sequential.RdYlGn_r[int(val * (len(px.colors.sequential.RdYlGn_r) -1) )] if pd.notnull(val) else '', subset=['is_attributed'])
        
        st.dataframe(styled_events, use_container_width=True)

def create_styled_table_html(df, fraud_column_name, threshold_for_traffic_light):
    """Создает HTML-таблицу со стилизацией светофора для колонки фрода."""
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        row_html = "<tr>"
        for col_name, cell_value in row.items():
            style = ""
            display_value = cell_value
            if col_name == fraud_column_name:
                traffic_light_info = get_fraud_traffic_light_info(cell_value, threshold_for_traffic_light)
                style = traffic_light_info['style']
                display_value = f"{cell_value:.3f}"
            elif isinstance(cell_value, float):
                display_value = f"{cell_value:.3f}"
            
            row_html += f'<td style="{style}">{display_value}</td>'
        row_html += "</tr>"
        rows_html += row_html

    table_html = f"""
    <div class="modern-table">
        <table>
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
    return table_html