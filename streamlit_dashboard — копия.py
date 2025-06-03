import streamlit as st
import pandas as pd
import os
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
import time

def safe_execution(max_retries=3, delay=1):
    """
    Декоратор для безопасного выполнения функций с автоматическим перезапуском
    при критических ошибках
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    error_msg = f"Ошибка в {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                    st.error(error_msg)
                    
                    if retries < max_retries:
                        st.warning(f"Попытка {retries} из {max_retries}. Перезапуск через {delay} секунд...")
                        time.sleep(delay)
                        continue
                    else:
                        st.error("Достигнуто максимальное количество попыток. Перезапуск приложения...")
                        st.rerun()
                        return None
        return wrapper
    return decorator

# --- Основные функции приложения ---
@safe_execution()
def main():
    try:
        # Инициализация состояния приложения
        if 'app_initialized' not in st.session_state:
            st.session_state['app_initialized'] = True
            st.session_state['error_count'] = 0
            st.session_state['last_error_time'] = None

        # Загрузка данных
        data = load_data()
        
        # Запуск симуляции
        run_simulation(data)
        
        # Обработка автообновления
        handle_autorefresh()
        
        # Основная логика приложения
        if 'filtered_data_base' not in st.session_state:
            st.session_state['filtered_data_base'] = pd.DataFrame()

        current_df = st.session_state['filtered_data_base'].copy()
        
        # Остальной код приложения...
        
    except Exception as e:
        st.error(f"Критическая ошибка в приложении: {str(e)}")
        st.error(traceback.format_exc())
        
        # Увеличиваем счетчик ошибок
        st.session_state['error_count'] = st.session_state.get('error_count', 0) + 1
        current_time = time.time()
        
        # Если было много ошибок за короткое время, перезапускаем приложение
        if (st.session_state['error_count'] >= 3 and 
            st.session_state.get('last_error_time') and 
            current_time - st.session_state['last_error_time'] < 60):
            st.error("Слишком много ошибок за короткое время. Перезапуск приложения...")
            st.rerun()
        else:
            st.session_state['last_error_time'] = current_time
            st.rerun()

if __name__ == "__main__":
    main()