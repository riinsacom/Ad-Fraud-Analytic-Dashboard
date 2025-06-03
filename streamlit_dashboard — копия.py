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

# Настройка темной темы
st.set_page_config(
    page_title="Fraud Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Применяем темную тему через CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stMetric {
        background-color: #2d2d2d;
        border-radius: 5px;
        padding: 10px;
    }
    .stDataFrame {
        background-color: #2d2d2d;
    }
    .stSelectbox {
        background-color: #2d2d2d;
    }
    .stSlider {
        background-color: #2d2d2d;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Fraud Analytics & Real-Time Monitoring Platform")

# Определение цветовой схемы для графиков
COLORS = {
    'background': '#1a1a1a',
    'paper_bgcolor': '#1a1a1a',
    'text': '#ffffff',
    'grid': '#333333',
    'primary': '#636EFA',
    'secondary': '#EF553B',
    'tertiary': '#00CC96',
    'warning': '#FFA15A',
    'pie_colors': ['#636EFA', '#EF553B', '#00CC96', '#FFA15A', '#AB63FA', '#19D3F3']
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
    # Уменьшаем количество строк для ускорения
    test = pd.read_csv('test.csv', nrows=100_000) 
    pred = pd.read_csv('Frod_Predict.csv', nrows=100_000)
    df = pd.merge(test, pred, on='click_id', how='left')
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['is_attributed'] = pd.to_numeric(df['is_attributed'], errors='coerce').fillna(0.0)
    return df

# --- Вспомогательные функции ---
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

def create_pie_chart(data, values, names, title):
    """Создание круговой диаграммы в темной теме"""
    fig = go.Figure(data=[go.Pie(
        labels=names,
        values=values,
        hole=.3,
        marker=dict(colors=COLORS['pie_colors'])
    )])
    fig.update_layout(
        title=title,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['paper_bgcolor'],
        font=dict(color=COLORS['text']),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text'])
        )
    )
    return fig

data = load_data()

# --- Сайдбар: фильтры и интерактив ---
st.sidebar.header("Фильтры и интерактив")
alert_threshold = st.sidebar.slider("Порог вероятности для алерта (фрода)", 0.0, 1.0, 0.5, 0.01)
top_n = st.sidebar.slider("Топ N для графиков", 5, 50, 10)

if not data.empty:
    time_min = data['click_time'].min().to_pydatetime()
    time_max = data['click_time'].max().to_pydatetime()
    time_range = st.sidebar.slider("Временной диапазон", min_value=time_min, max_value=time_max, value=(time_min, time_max), format="YYYY-MM-DD HH:mm:ss")
    filtered_data_base = data[(data['click_time'] >= time_range[0]) & (data['click_time'] <= time_range[1])].copy()
else:
    st.error("Нет данных для отображения после загрузки. Проверьте исходные файлы.")
    # Создаем пустой DataFrame, чтобы дашборд не падал
    filtered_data_base = pd.DataFrame(columns=data.columns)
    # Устанавливаем фиктивные значения для слайдеров, если data пустой
    dt_now = datetime.now()
    time_range = st.sidebar.slider("Временной диапазон", min_value=dt_now - timedelta(days=1), max_value=dt_now, value=(dt_now - timedelta(days=1), dt_now), format="YYYY-MM-DD HH:mm:ss")

# Категории для анализа
cat_options = ['ip', 'app', 'device', 'channel']
cat1 = st.sidebar.selectbox("Категория 1 (ось X) для вкладки 'Категории'", cat_options, index=0, key="cat1_sidebar")
cat2 = st.sidebar.selectbox("Категория 2 (цвет/группировка) для вкладки 'Категории'", cat_options, index=1, key="cat2_sidebar")

# Для алертов
show_alerts_only = st.sidebar.checkbox("Показывать только алерты", value=False)

# --- Основной DataFrame для вкладок ---
current_df = filtered_data_base.copy()
# Колонка 'cluster' больше не создается и не используется в current_df

# --- Tabs ---
tabs = st.tabs(["Главная", "Категории", "Связи/Графы", "Корреляции", "Алерты", "Последние события"])

# --- Главная ---
with tabs[0]:
    st.subheader("Ключевые показатели")
    
    # Метрики в темной теме
    metrics_container = st.container()
    col1, col2, col3, col4 = metrics_container.columns(4)
    
    total_clicks = len(current_df)
    avg_fraud_prob = current_df['is_attributed'].mean() if total_clicks > 0 else 0
    fraud_clicks = (current_df['is_attributed'] > alert_threshold).sum()
    fraud_share = fraud_clicks / total_clicks if total_clicks > 0 else 0
    
    with col1:
        st.metric("Всего кликов", f"{total_clicks:,}", 
                 delta=None, delta_color="normal")
    with col2:
        st.metric("Средняя вероятность фрода", f"{avg_fraud_prob:.3f}", 
                 delta=None, delta_color="normal")
    with col3:
        st.metric(f"Фрод-кликов (p>{alert_threshold})", f"{fraud_clicks:,}", 
                 delta=None, delta_color="normal")
    with col4:
        st.metric(f"Доля фрод-кликов", f"{fraud_share:.2%}", 
                 delta=None, delta_color="normal")

    # Круговые диаграммы
    st.subheader("Общее распределение кликов")
    pie_cols = st.columns(4)
    
    if not current_df.empty:
        with pie_cols[0]:
            device_stats = current_df['device'].value_counts()
            fig_device = create_pie_chart(
                current_df, 
                device_stats.values, 
                device_stats.index,
                'Распределение по устройствам'
            )
            st.plotly_chart(fig_device, use_container_width=True)
        
        with pie_cols[1]:
            app_stats = current_df['app'].value_counts().head(5)
            fig_app = create_pie_chart(
                current_df,
                app_stats.values,
                app_stats.index,
                'Топ приложений'
            )
            st.plotly_chart(fig_app, use_container_width=True)
        
        with pie_cols[2]:
            channel_stats = current_df['channel'].value_counts()
            fig_channel = create_pie_chart(
                current_df,
                channel_stats.values,
                channel_stats.index,
                'Распределение по каналам'
            )
            st.plotly_chart(fig_channel, use_container_width=True)
        
        with pie_cols[3]:
            fraud_stats = pd.Series({
                'Фрод': fraud_clicks,
                'Не фрод': total_clicks - fraud_clicks
            })
            fig_fraud = create_pie_chart(
                current_df,
                fraud_stats.values,
                fraud_stats.index,
                'Распределение фрод/не фрод'
            )
            st.plotly_chart(fig_fraud, use_container_width=True)
    else:
        st.info("Нет данных для отображения круговых диаграмм.")

    # Подозрительные паттерны
    st.subheader("Подозрительные паттерны")
    patterns = get_suspicious_patterns_cached(current_df, alert_threshold)
    if patterns:
        for pattern in patterns[:5]:
            st.warning(pattern)
    else:
        st.info("Подозрительных паттернов не обнаружено")

    # Динамика с темной темой
    if not current_df.empty:
        time_df = current_df.copy()
        time_df['minute'] = time_df['click_time'].dt.floor('min')
        agg = time_df.groupby('minute').agg(
            clicks=('is_attributed', 'count'),
            avg_fraud_prob=('is_attributed', 'mean'),
            fraud_clicks=('is_attributed', lambda x: (x > alert_threshold).sum())
        ).reset_index()
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=agg['minute'], 
            y=agg['clicks'], 
            name='Клики',
            marker_color=COLORS['primary'],
            opacity=0.5
        ))
        fig_time.add_trace(go.Scatter(
            x=agg['minute'],
            y=agg['avg_fraud_prob'],
            name='Средняя вероятность фрода',
            yaxis='y2',
            line=dict(color=COLORS['secondary'])
        ))
        fig_time.add_trace(go.Scatter(
            x=agg['minute'],
            y=agg['fraud_clicks'],
            name=f'Фрод-клики (p>{alert_threshold})',
            line=dict(color=COLORS['tertiary'], dash='dot')
        ))
        
        fig_time.update_layout(
            title="Динамика по времени",
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['paper_bgcolor'],
            font=dict(color=COLORS['text']),
            yaxis=dict(
                title='Клики',
                gridcolor=COLORS['grid'],
                zerolinecolor=COLORS['grid']
            ),
            yaxis2=dict(
                title='Средняя вероятность фрода',
                overlaying='y',
                side='right',
                range=[0,1],
                gridcolor=COLORS['grid'],
                zerolinecolor=COLORS['grid']
            ),
            legend=dict(
                orientation='h',
                bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text'])
            ),
            uirevision='time_graph'
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Нет данных для отображения динамики.")

    # Гистограммы в темной теме
    col1_dist, col2_dist = st.columns(2)
    with col1_dist:
        st.subheader("Распределение вероятности фрода")
        fig_hist = px.histogram(
            current_df,
            x='is_attributed',
            nbins=50,
            title='Гистограмма вероятности фрода',
            color_discrete_sequence=[COLORS['primary']]
        )
        fig_hist.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['paper_bgcolor'],
            font=dict(color=COLORS['text']),
            xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
            yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2_dist:
        st.subheader("Фрод по времени суток")
        current_df['hour'] = current_df['click_time'].dt.hour
        fig_hour = px.box(
            current_df,
            x='hour',
            y='is_attributed',
            title='Вероятность фрода по часам',
            color_discrete_sequence=[COLORS['primary']]
        )
        fig_hour.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['paper_bgcolor'],
            font=dict(color=COLORS['text']),
            xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
            yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    # Главная вкладка - добавление "Топ мошеннических сущностей"
    st.subheader("Топ мошеннических сущностей")
    if not current_df.empty and 'is_attributed' in current_df.columns and 'ip' in current_df.columns and 'app' in current_df.columns:
        high_fraud_df = current_df[current_df['is_attributed'] > alert_threshold]
        if high_fraud_df.empty:
            st.caption("(Нет кликов выше порога фрода, показываются топ по всем данным с учетом порога > 0 для демонстрации)")
            # Если нет кликов выше порога, но мы все равно хотим показать "потенциально" проблемные,
            # можно взять топ по всем данным, но пометить, что они не превысили порог.
            # Для демонстрации, возьмем топ-N по всем данным, если high_fraud_df пуст, 
            # но отфильтруем по is_attributed > 0, чтобы показать хотя бы какие-то "подозрительные".
            high_fraud_df = current_df[current_df['is_attributed'] > 0.01] # небольшой порог для примера
            if high_fraud_df.empty:
                 high_fraud_df = current_df # крайний случай, если вообще нет фрода

        # Используем top_n из сайдбара
        # top_n_entities = 5 # Старое значение
        top_n_entities = top_n 

        # Топ IP адресов
        suspicious_ips_agg = high_fraud_df.groupby('ip').agg(
            click_count=('click_id', 'count'),
            avg_fraud_prob=('is_attributed', 'mean')
        ).reset_index()
        suspicious_ips_table = suspicious_ips_agg.sort_values(by='click_count', ascending=False).nlargest(top_n_entities, 'click_count')
        suspicious_ips_table.columns = ['IP', 'Количество кликов', 'Средняя P(фрод)']
        
        # Топ приложений
        suspicious_apps_agg = high_fraud_df.groupby('app').agg(
            click_count=('click_id', 'count'),
            avg_fraud_prob=('is_attributed', 'mean')
        ).reset_index()
        suspicious_apps_table = suspicious_apps_agg.sort_values(by='click_count', ascending=False).nlargest(top_n_entities, 'click_count')
        suspicious_apps_table.columns = ['App ID', 'Количество кликов', 'Средняя P(фрод)']

        col_ip_fraud, col_app_fraud = st.columns(2)
        with col_ip_fraud:
            st.write(f"Топ-{top_n_entities} IP адресов (фильтр по кликам > порога фрода: {alert_threshold}):")
            st.dataframe(suspicious_ips_table.style.format({'Средняя P(фрод)': '{:.3f}'}), use_container_width=True)
        with col_app_fraud:
            st.write(f"Топ-{top_n_entities} приложений (фильтр по кликам > порога фрода: {alert_threshold}):")
            st.dataframe(suspicious_apps_table.style.format({'Средняя P(фрод)': '{:.3f}'}), use_container_width=True)
    else:
        st.info("Нет данных для отображения топа мошеннических сущностей.")

# --- Категории ---
with tabs[1]:
    st.subheader("Анализ по выбранным категориям")
    cat_analysis_df = current_df 
    
    st.subheader(f"Статистика по категории: {cat1}")
    stats_cols = st.columns(2)
    
    with stats_cols[0]:
        if not cat_analysis_df.empty and cat1 in cat_analysis_df.columns:
            top_cats = cat_analysis_df[cat1].value_counts().nlargest(top_n)
            if not top_cats.empty:
                fig_top = go.Figure(data=[
                    go.Bar(
                        x=top_cats.index.astype(str), # Явное преобразование в строку для оси X
                        y=top_cats.values,
                        marker_color=COLORS['primary']
                    )
                ])
                fig_top.update_layout(
                    title=f'Топ {top_n} по количеству кликов ({cat1})',
                    plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['paper_bgcolor'],
                    font=dict(color=COLORS['text']),
                    xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], type='category'), # type='category' для лучшего отображения строковых меток
                    yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
                )
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.info(f"Нет данных для категории {cat1} для отображения топа по кликам.")
        else:
            st.info(f"Категория {cat1} не найдена или нет данных.")
    
    with stats_cols[1]:
        if not cat_analysis_df.empty and cat1 in cat_analysis_df.columns and 'is_attributed' in cat_analysis_df.columns:
            avg_fraud = cat_analysis_df.groupby(cat1)['is_attributed'].mean().nlargest(top_n)
            if not avg_fraud.empty:
                fig_avg = go.Figure(data=[
                    go.Bar(
                        x=avg_fraud.index.astype(str),
                        y=avg_fraud.values,
                        marker_color=COLORS['secondary']
                    )
                ])
                fig_avg.update_layout(
                    title=f'Топ {top_n} по ср. вероятности фрода ({cat1})',
                    plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['paper_bgcolor'],
                    font=dict(color=COLORS['text']),
                    xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], type='category'),
                    yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], range=[0,1]) # Диапазон для вероятности
                )
                st.plotly_chart(fig_avg, use_container_width=True)
            else:
                st.info(f"Нет данных для категории {cat1} для отображения топа по фроду.")
        else:
            st.info(f"Категория {cat1} или колонка 'is_attributed' не найдена, или нет данных.")

    # Анализ связей между категориями
    st.subheader(f"Связи между категориями: {cat1} и {cat2}")
    if not cat_analysis_df.empty and cat1 in cat_analysis_df.columns and cat2 in cat_analysis_df.columns:
        if cat1 == cat2:
            st.info(f"Для тепловой карты выберите различные категории (Категория 1: {cat1}, Категория 2: {cat2}).")
        else:
            pivot = pd.crosstab(cat_analysis_df[cat1], cat_analysis_df[cat2])
            if not pivot.empty:
                # Опция выбора цветовой схемы
                color_scales = ['Viridis', 'Cividis', 'Plasma', 'Blues', 'Greens', 'Reds']
                selected_color_scale = st.selectbox("Цветовая схема для тепловой карты:", color_scales, index=0, key="heatmap_color_scale")
                
                fig_heatmap = px.imshow(
                    pivot,
                    title=f'Тепловая карта связей {cat1}-{cat2}',
                    color_continuous_scale=selected_color_scale
                )
                fig_heatmap.update_layout(
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['paper_bgcolor'],
                    font=dict(color=COLORS['text'])
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info(f"Нет данных для построения тепловой карты между {cat1} и {cat2}.")
    else:
        st.info(f"Одна или обе выбранные категории ({cat1}, {cat2}) не найдены или нет данных.")

    # Временной анализ
    st.subheader(f"Временной анализ для категории: {cat1}")
    if not cat_analysis_df.empty and cat1 in cat_analysis_df.columns and 'click_time' in cat_analysis_df.columns and 'is_attributed' in cat_analysis_df.columns:
        temp_df_time_cat = cat_analysis_df.copy()
        temp_df_time_cat['hour'] = temp_df_time_cat['click_time'].dt.hour
        # Ограничим количество отображаемых значений cat1 на boxplot, если их много
        top_cat1_for_boxplot = temp_df_time_cat[cat1].value_counts().nlargest(10).index
        df_for_boxplot = temp_df_time_cat[temp_df_time_cat[cat1].isin(top_cat1_for_boxplot)]

        if not df_for_boxplot.empty:
            fig_time_cat = px.box(
                df_for_boxplot,
                x='hour',
                y='is_attributed',
                color=cat1,
                title=f'Распределение P(фрод) по часам и топ значениям {cat1}',
                color_discrete_sequence=COLORS['pie_colors']
            )
            fig_time_cat.update_layout(
                plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['paper_bgcolor'],
                font=dict(color=COLORS['text']),
                xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
                yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], range=[0,1]) # Диапазон для вероятности
            )
            st.plotly_chart(fig_time_cat, use_container_width=True)
        else:
            st.info(f"Нет данных для временного анализа категории {cat1}.")
    else:
        st.info(f"Необходимые колонки ({cat1}, click_time, is_attributed) отсутствуют или нет данных.")

# --- Связи/Графы ---
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
                                       help="Выберите первый тип сущностей для анализа связей")
        graph_node2_attr = st.selectbox("Тип узлов B", graph_node_options,
                                       index=graph_node_options.index('device') if 'device' in graph_node_options else 1,
                                       key="graph_node2",
                                       help="Выберите второй тип сущностей для создания связей")
    
    with settings_col2:
        st.markdown("**Параметры визуализации:**")
        graph_dimension = st.radio("Режим отображения", ('2D (быстрый)', '3D (интерактивный)'), index=0, 
                                   key="graph_dim",
                                   help="2D режим быстрее загружается, 3D более наглядный")
        
        layout_options = {
            'Органичное (рекомендуется)': 'spring',
            'Круговое расположение': 'circular', 
            'Сбалансированное': 'kamada_kawai',
            'Случайное': 'random'
        }
        selected_layout = st.selectbox("Алгоритм размещения", list(layout_options.keys()), 
                                       key="graph_layout",
                                       help="Определяет как узлы будут расположены на графе")
        layout_algorithm = layout_options[selected_layout]
    
    with settings_col3:
        st.markdown("**Фильтрация данных:**")
        
        # Режимы фильтрации с логичными названиями
        filter_modes = {
            "Все данные (обзор)": "all",
            "Только мошеннические связи": "fraud_only", 
            "Топ подозрительных узлов": "top_fraud",
            "Аномальные временные периоды": "time_clusters"
        }
        
        data_filter_mode_display = st.selectbox("Режим фильтрации", list(filter_modes.keys()),
                                                help="Все данные - случайная выборка для общего обзора\nМошеннические связи - только высокий уровень фрода\nТоп подозрительных - самые проблемные узлы\nАномальные периоды - временные всплески активности")
        data_filter_mode = filter_modes[data_filter_mode_display]
        
        # Динамические настройки
        if data_filter_mode == "fraud_only":
            fraud_threshold = st.slider("Порог вероятности фрода", 0.0, 1.0, 0.3, 0.05, 
                                       help="Показывать только связи с вероятностью фрода выше данного значения")
        elif data_filter_mode == "top_fraud":
            top_count = st.slider("Количество топ узлов", 5, 50, 15,
                                 help="Количество самых подозрительных узлов для анализа")
        elif data_filter_mode == "time_clusters":
            time_window = st.slider("Временное окно (часы)", 1, 12, 3,
                                   help="Размер окна для группировки событий по времени")
        else:
            sample_size = st.slider("Размер выборки", 1000, 5000, 2000, 250,
                                   help="Количество случайных записей для анализа")
    
    with settings_col4:
        st.markdown("**Настройки отображения:**")
        min_connections = st.slider("Минимум связей", 1, 10, 2,
                                   help="Узлы с меньшим количеством связей будут скрыты")
        max_nodes = st.slider("Максимум узлов", 20, 200, 50, 10,
                             help="Ограничение количества узлов для улучшения производительности")
        
        show_labels = st.checkbox("Показывать подписи узлов", True,
                                 help="Отображать названия узлов на графе")
        analyze_communities = st.checkbox("Обнаружить группы", False,
                                         help="Автоматически выделить группы тесно связанных узлов")

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
                    st.success(f"**Режим анализа:** Все данные - обработано {len(graph_data):,} записей")
                    
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
            
            1. **Выберите типы узлов** для анализа связей (рекомендуется: IP ↔ устройства)
            2. **Настройте режим фильтрации** в зависимости от задач анализа
            3. **Установите лимиты отображения** для оптимальной производительности
            4. **Нажмите кнопку "ПОСТРОИТЬ ГРАФ СВЯЗЕЙ"**
            
            **Рекомендации для начинающих:**
            - Начните с режима "Мошеннические связи" с порогом 0.3
            - Установите минимум связей = 2, максимум узлов = 50
            - Включите "Показывать подписи узлов" для лучшего понимания
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
with tabs[3]:
    st.subheader("Матрица корреляций")
    corr_analysis_df = current_df
    
    # Оставляем только числовые колонки для матрицы корреляций
    numeric_cols_for_corr = corr_analysis_df.select_dtypes(include=np.number).columns.tolist()
    # Добавляем категориальные, если они представлены числами (ID) и их немного уникальных значений
    # или если мы хотим их принудительно факторизовать. 
    # Для текущей задачи, оставим как есть: ip, app, device, channel - они числовые ID.
    # Убедимся, что нужные колонки есть, прежде чем считать корреляцию
    cols_for_corr_matrix = [col for col in ['ip','app','device','channel','is_attributed'] if col in corr_analysis_df.columns and pd.api.types.is_numeric_dtype(corr_analysis_df[col])]
    
    if len(cols_for_corr_matrix) > 1:
        corr = corr_analysis_df[cols_for_corr_matrix].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Корреляции между признаками'
        )
        fig_corr.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['paper_bgcolor'],
            font=dict(color=COLORS['text'])
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Недостаточно числовых данных для построения матрицы корреляций.")

    # Добавляем scatter plot для выбранных признаков
    st.subheader("Диаграмма рассеяния")
    scatter_cols = st.columns(2)
    
    # Улучшение: available_features_for_scatter - все числовые или категориальные с небольшим числом уникальных значений
    # В данном случае, ip, app, device, channel - это ID, и они числовые.
    # is_attributed - тоже числовой. 'click_id' - уникален, его не берем.
    # 'hour' - тоже числовой, но мы его создаем на лету в других местах, здесь его может не быть в current_df.
    # Поэтому, для простоты, оставим изначальный набор, но проверим их наличие и тип.
    potential_scatter_features = ['ip', 'app', 'device', 'channel', 'is_attributed']
    available_features_for_scatter = [col for col in potential_scatter_features 
                                      if col in corr_analysis_df.columns and 
                                         (pd.api.types.is_numeric_dtype(corr_analysis_df[col]) or 
                                          corr_analysis_df[col].nunique() < 100)] # Добавим условие на кол-во уникальных для категориальных

    if len(available_features_for_scatter) < 2:
        st.warning("Недостаточно подходящих признаков для построения диаграммы рассеяния.")
    else:
        with scatter_cols[0]:
            x_feature = st.selectbox("Признак X", available_features_for_scatter, 
                                     index=0 if available_features_for_scatter else -1, # Защита от пустого списка
                                     key="scatter_x_feat")
        with scatter_cols[1]:
            y_feature = st.selectbox("Признак Y", available_features_for_scatter, 
                                     index=min(1, len(available_features_for_scatter)-1) if len(available_features_for_scatter) > 1 else 0, 
                                     key="scatter_y_feat")
        
        if x_feature and y_feature and x_feature != y_feature:
            plot_data_scatter = corr_analysis_df[[x_feature, y_feature, 'is_attributed']].copy()
            # Преобразование категориальных в числовые для ScatterGL, если они еще не такие
            for col_to_convert in [x_feature, y_feature]:
                 if not pd.api.types.is_numeric_dtype(plot_data_scatter[col_to_convert]):
                    plot_data_scatter[col_to_convert], _ = pd.factorize(plot_data_scatter[col_to_convert])
            
            MAX_POINTS_SCATTERGL = 20000
            if len(plot_data_scatter) > MAX_POINTS_SCATTERGL:
                st.info(f"Для диаграммы рассеяния используется случайная выборка из {MAX_POINTS_SCATTERGL} точек.")
                plot_data_scatter = plot_data_scatter.sample(n=MAX_POINTS_SCATTERGL, random_state=42)

            fig_scatter = go.Figure(data=go.Scattergl(
                x=plot_data_scatter[x_feature],
                y=plot_data_scatter[y_feature],
                mode='markers',
                marker=dict(
                    color=plot_data_scatter['is_attributed'],
                    colorscale='RdBu_r',
                    showscale=True,
                    colorbar_title='Fraud Prob.'
                )
            ))
            fig_scatter.update_layout(
                title=f'Диаграмма рассеяния: {x_feature} vs {y_feature}',
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['paper_bgcolor'],
                font=dict(color=COLORS['text']),
                xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
                yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        elif x_feature == y_feature and x_feature is not None:
            st.info("Выберите разные признаки для осей X и Y.")
        elif not x_feature or not y_feature:
            st.info("Выберите признаки для X и Y осей.")

# --- Алерты ---
with tabs[4]:
    st.subheader("Алерт-лист (клики с P(фрод) > порога)")
    alerts_df = current_df[current_df['is_attributed'] > alert_threshold]

    if alerts_df.empty:
        st.info(f"Нет кликов с вероятностью фрода выше {alert_threshold}.")
    else:
        # Статистика алертов
        st.subheader("Статистика по алертам")
        
        # Уменьшаем количество колонок, т.к. одна диаграмма убрана
        alert_stats_cols = st.columns(3) 
        
        with alert_stats_cols[0]:
            # Проверка на пустоту перед value_counts
            if not alerts_df['click_time'].dt.hour.value_counts().empty:
                fig_alerts_by_hour = create_pie_chart(
                    alerts_df, # Используем alerts_df
                    alerts_df['click_time'].dt.hour.value_counts().values,
                    [f"{h}:00" for h in alerts_df['click_time'].dt.hour.value_counts().index],
                    'Распределение по часам'
                )
                st.plotly_chart(fig_alerts_by_hour, use_container_width=True)
            else:
                st.info("Нет данных для распределения алертов по часам.")
        
        with alert_stats_cols[1]:
            if not alerts_df['device'].value_counts().empty:
                fig_alerts_by_device = create_pie_chart(
                    alerts_df, # Используем alerts_df
                    alerts_df['device'].value_counts().values,
                    alerts_df['device'].value_counts().index,
                    'Распределение по устройствам'
                )
                st.plotly_chart(fig_alerts_by_device, use_container_width=True)
            else:
                st.info("Нет данных для распределения алертов по устройствам.")
        
        with alert_stats_cols[2]:
            if not alerts_df['app'].value_counts().empty:
                top_apps_alerts = alerts_df['app'].value_counts().head(5)
                fig_alerts_by_app = create_pie_chart(
                    alerts_df, # Используем alerts_df
                    top_apps_alerts.values,
                    top_apps_alerts.index,
                    'Топ приложений в алертах'
                )
                st.plotly_chart(fig_alerts_by_app, use_container_width=True)
            else:
                st.info("Нет данных для топа приложений в алертах.")
        
        # Таблица алертов
        st.subheader("Список алертов")
        st.dataframe(
            alerts_df.head(100).style.format({'is_attributed': "{:.3f}"})
            .background_gradient(subset=['is_attributed'], cmap='RdYlGn_r')
        )
        
        # Экспорт
        st.download_button(
            "Скачать алерты",
            alerts_df.to_csv(index=False),
            file_name="alerts.csv",
            mime="text/csv"
        )

        # Подробности по клику
        st.subheader("Детали по клику (из списка алертов)")
        if not alerts_df.empty:
            # Убедимся, что click_id для выбора существует в alerts_df
            available_click_ids_alerts = alerts_df['click_id'].unique()

            click_id_alert = st.selectbox( # Замена number_input на selectbox для удобства
                "Выберите click_id для подробностей (из алертов):",
                options=available_click_ids_alerts,
                index=0, # По умолчанию первый из доступных
                key="alert_click_id_selector"
            ) 
            
            # Поиск детальной информации о клике
            click_row = current_df[current_df['click_id'] == click_id_alert] 
            if not click_row.empty:
                with st.expander("Детальная информация о клике"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Основная информация:")
                        st.write(click_row[['click_id', 'click_time', 'is_attributed']].T)
                    
                    with col2:
                        st.write("Связанные параметры:")
                        st.write(click_row[['ip', 'app', 'device', 'channel']].T)
                
                    # Связанные клики
                    st.subheader("Связанные клики (по отношению к выбранному алерту)")
                    # Передаем current_df, так как get_related_clicks ожидает полный датафрейм для поиска связей
                    related_by_ip = get_related_clicks(current_df, click_id_alert, 'ip') 
                    related_by_device = get_related_clicks(current_df, click_id_alert, 'device')
                    
                    col1_related, col2_related = st.columns(2) # Избегаем конфликта имен колонок
                    with col1_related:
                        st.write(f"Клики с того же IP ({len(related_by_ip)}) в общем датасете:")
                        st.dataframe(
                            related_by_ip[['click_time', 'is_attributed', 'app', 'device']]
                            .head(10)
                            .style.format({'is_attributed': "{:.3f}"})
                            .background_gradient(subset=['is_attributed'], cmap='RdYlGn_r')
                        )
                    
                    with col2_related:
                        st.write(f"Клики с того же устройства ({len(related_by_device)}) в общем датасете:")
                        st.dataframe(
                            related_by_device[['click_time', 'is_attributed', 'app', 'ip']]
                            .head(10)
                            .style.format({'is_attributed': "{:.3f}"})
                            .background_gradient(subset=['is_attributed'], cmap='RdYlGn_r')
                        )
            else:
                st.warning(f"Клик с ID {click_id_alert} не найден в общем датасете для детального просмотра.")
        else:
            st.info("Нет алертов для выбора click_id.")

# --- Последние события ---
with tabs[5]:
    st.subheader("Обзор событий (отсортировано по времени)")
    # recent_events_df = current_df # Старая строка
    # Улучшение: сортируем current_df по времени для этой вкладки
    recent_events_df = current_df.sort_values(by='click_time', ascending=False)

    if recent_events_df.empty:
        st.info("Нет событий для отображения в выбранном временном диапазоне.")
    else:
        # Статистика последних событий
        st.subheader("Статистика по отображаемым событиям")
        
        # Уменьшаем количество колонок, т.к. одна диаграмма убрана
        recent_stats_cols = st.columns(2) 
        
        with recent_stats_cols[0]:
            if not recent_events_df['device'].value_counts().empty:
                fig_recent_by_device = create_pie_chart(
                    recent_events_df,
                    recent_events_df['device'].value_counts().values,
                    recent_events_df['device'].value_counts().index,
                    'Распределение по устройствам'
                )
                st.plotly_chart(fig_recent_by_device, use_container_width=True)
            else:
                st.info("Нет данных для распределения событий по устройствам.")
        
        with recent_stats_cols[1]:
            if not recent_events_df['app'].value_counts().empty:
                top_apps_recent = recent_events_df['app'].value_counts().head(5)
                fig_recent_by_app = create_pie_chart(
                    recent_events_df,
                    top_apps_recent.values,
                    top_apps_recent.index,
                    'Топ приложений в событиях'
                )
                st.plotly_chart(fig_recent_by_app, use_container_width=True)
            else:
                st.info("Нет данных для топа приложений в событиях.")
        
        # Таблица последних событий
        st.subheader("Список событий (последние сверху)")
        st.dataframe(
            recent_events_df.head(100).style.format({'is_attributed': "{:.3f}"}) # Отображаем head(100) для производительности
            .background_gradient(subset=['is_attributed'], cmap='RdYlGn_r')
        ) 
        # Можно добавить кнопку "Скачать все отображаемые события"
        st.download_button(
            "Скачать отображаемые события",
            recent_events_df.to_csv(index=False),
            file_name="recent_events.csv",
            mime="text/csv",
            key="download_recent_events"
        ) 