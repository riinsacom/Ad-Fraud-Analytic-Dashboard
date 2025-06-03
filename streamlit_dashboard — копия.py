import streamlit as st
import time
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
import numpy as np
import traceback
try:
    from scipy import stats
except ImportError:
    stats = None  # Fallback if scipy is not available
import gc  # Для ручного управления памятью
import sys
from functools import wraps
import psutil
import atexit
import threading
import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Глобальные настройки ---
MAX_RETRIES = 3
RETRY_DELAY = 1
HEALTH_CHECK_INTERVAL = 5
IDLE_TIMEOUT = 120
MEMORY_LIMIT = 300  # Уменьшен лимит памяти до 300MB
OPERATION_TIMEOUT = 10
FORCED_RESTART_INTERVAL = 300  # 5 минут
RESTART_COOLDOWN = 10  # 10 секунд между перезапусками
CHUNK_SIZE = 10000  # Размер чанка для обработки данных
UI_UPDATE_DELAY = 0.1  # Задержка между обновлениями UI
MAX_UI_RETRIES = 3  # Максимальное количество попыток обновления UI

# --- Оптимизация памяти ---
def optimize_memory():
    """Оптимизация использования памяти"""
    try:
        # Принудительная очистка памяти
        gc.collect()
        
        # Очищаем кэш pandas
        pd.DataFrame().empty
        
        # Очищаем кэш numpy
        np.empty(0)
        
        # Очищаем кэш plotly
        if 'plotly' in sys.modules:
            import plotly.io as pio
            pio.templates.default = None
        
        return True
    except:
        return False

# --- Безопасное обновление UI ---
def safe_ui_update(func):
    """Декоратор для безопасного обновления UI"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_UI_RETRIES):
            try:
                # Добавляем небольшую задержку между попытками
                if attempt > 0:
                    time.sleep(UI_UPDATE_DELAY)
                
                # Очищаем кэш перед обновлением
                st.cache_data.clear()
                
                # Выполняем функцию
                result = func(*args, **kwargs)
                
                # Принудительная очистка памяти
                gc.collect()
                
                return result
            except Exception as e:
                if attempt == MAX_UI_RETRIES - 1:
                    st.error(f"Ошибка обновления UI: {str(e)}")
                    # Пробуем перезапустить приложение
                    safe_restart()
                time.sleep(UI_UPDATE_DELAY * (attempt + 1))
        return None
    return wrapper

def safe_container_update(container, content_func):
    """Безопасное обновление контейнера"""
    try:
        with container:
            content_func()
    except Exception as e:
        st.error(f"Ошибка обновления контейнера: {str(e)}")
        # Пробуем очистить контейнер
        try:
            container.empty()
        except:
            pass

def safe_restart():
    """Безопасный перезапуск приложения"""
    try:
        # Очищаем все контейнеры
        for key in list(st.session_state.keys()):
            if key.startswith('container_'):
                try:
                    del st.session_state[key]
                except:
                    pass
        
        # Очищаем кэш
        st.cache_data.clear()
        
        # Принудительная очистка памяти
        gc.collect()
        
        # Перезапускаем приложение
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Ошибка при перезапуске: {str(e)}")
        # Пробуем принудительный перезапуск
        try:
            st.rerun()
        except:
            pass

# Настройка страницы должна быть первым вызовом Streamlit
st.set_page_config(
    page_title="Аналитика Фрода",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния приложения
if 'app_initialized' not in st.session_state:
    try:
        # Оптимизируем память при старте
        optimize_memory()
        
        # Инициализируем состояние
        st.session_state.app_initialized = True
        st.session_state.app_start_time = time.time()
        st.session_state.last_restart_time = time.time()
        st.session_state.last_health_check = time.time()
        st.session_state.error_count = 0
        st.session_state.last_activity_time = time.time()
        st.session_state.ui_update_lock = threading.Lock()
    except Exception as e:
        st.error(f"Ошибка при инициализации приложения: {str(e)}")
        st.rerun()

# --- Остальной код ---
// ... existing code ...