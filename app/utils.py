# app/utils.py
import os
import requests
import pandas as pd
from datetime import datetime
import logging
from io import StringIO
import time
from functools import lru_cache

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'moex_data')

def download_data(ticker, start_date=None, end_date=None, max_retries=3):
    base_url = f'https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.csv'
    all_data = []
    start = 0
    batch_size = 100  # Максимальное количество строк за один запрос
    
    for attempt in range(max_retries):
        try:
            while True:
                params = {
                    'start': start,
                    'limit': batch_size
                }
                if start_date:
                    params['from'] = start_date
                if end_date:
                    params['till'] = end_date
                
                response = requests.get(base_url, params=params)
                if response.status_code != 200:
                    raise Exception(f"HTTP error {response.status_code}: {response.text}")
                
                # Читаем данные из ответа
                data_str = response.text
                lines = data_str.split('\n')
                
                # Находим начало и конец данных
                data_start = None
                data_end = None
                for i, line in enumerate(lines):
                    if line.startswith('BOARDID;TRADEDATE;'):
                        data_start = i
                    if line.startswith('history.cursor'):
                        data_end = i
                        break
                
                if data_start is None:
                    logging.info("No data header found. Breaking loop.")
                    break  # Нет данных
                
                # Извлекаем данные
                data_lines = lines[data_start:data_end]
                if len(data_lines) <= 1:  # Только заголовок
                    logging.info("No data lines found. Breaking loop.")
                    break
                    
                # Преобразуем в DataFrame
                data_str = '\n'.join(data_lines)
                df_batch = pd.read_csv(StringIO(data_str), delimiter=';', encoding='cp1251')
                
                # Логирование первых строк батча для диагностики
                logging.info(f"Batch data snippet:\n{df_batch.head()}")
                
                all_data.append(df_batch)
                
                # Проверяем, есть ли еще данные
                if len(df_batch) < batch_size:
                    logging.info("Batch size less than limit. Breaking loop.")
                    break
                    
                start += batch_size
                time.sleep(0.5)  # Задержка между запросами
            
            if not all_data:
                raise Exception(f"No data found for {ticker}")
            
            # Объединяем все данные
            df = pd.concat(all_data, ignore_index=True)
            
            # Логирование объединённых данных для диагностики
            logging.info(f"Combined data snippet:\n{df.head()}")
            
            # Сохраняем в файл
            os.makedirs(DATA_DIR, exist_ok=True)
            file_path = os.path.join(DATA_DIR, f'{ticker}.csv')
            df.to_csv(file_path, index=False, encoding='cp1251')
            logging.info(f"Data downloaded and saved to {file_path}. Total rows: {len(df)}")
            return df
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"All attempts failed. Error: {str(e)}")
                raise
            logging.warning(f"Attempt {attempt + 1} failed. Retrying... Error: {str(e)}")
            time.sleep(2 ** attempt)  # Экспоненциальная задержка

@lru_cache(maxsize=32)
def load_data(ticker):
    file_path = os.path.join(DATA_DIR, f'{ticker}.csv')
    
    try:
        # Читаем файл с учетом возможных ошибок кодировки
        with open(file_path, 'r', encoding='cp1251') as f:
            lines = f.readlines()
        
        # Находим начало и конец данных
        header_idx = None
        footer_idx = None
        for i, line in enumerate(lines):
            if line.startswith('BOARDID;TRADEDATE;'):
                header_idx = i
            if line.startswith('history.cursor'):
                footer_idx = i
                break
        
        if header_idx is None:
            raise Exception("Could not find data header in the CSV file")
        
        # Читаем только данные
        if footer_idx is not None:
            data_lines = lines[header_idx:footer_idx]
        else:
            data_lines = lines[header_idx:]
        
        # Преобразуем в DataFrame
        data_str = '\n'.join(data_lines)
        df = pd.read_csv(StringIO(data_str), delimiter=';', encoding='cp1251')
        
        # Логирование первых строк загруженных данных для диагностики
        logging.info(f"Loaded data snippet:\n{df.head()}")
        
        # Очистка данных
        df = df.dropna(subset=['TRADEDATE'])
        df = df[df['TRADEDATE'].str.contains(r'\d{4}-\d{2}-\d{2}', na=False)]
        
        # Преобразуем дату и устанавливаем как индекс
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'], format='%Y-%m-%d')
        df.set_index('TRADEDATE', inplace=True)
        df.sort_index(inplace=True)  # Сортируем по дате
        
        # Преобразуем числовые колонки в float
        numeric_cols = ['NUMTRADES', 'VALUE', 'OPEN', 'LOW', 'HIGH', 'LEGALCLOSEPRICE', 
                       'WAPRICE', 'CLOSE', 'VOLUME', 'MARKETPRICE2', 'MARKETPRICE3',
                       'ADMITTEDQUOTE', 'MP2VALTRD', 'MARKETPRICE3TRADESVALUE', 
                       'ADMITTEDVALUE', 'WAVAL', 'TRENDCLSPR']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Проверка наличия пропущенных значений в столбце CLOSE
        if df['CLOSE'].isnull().any():
            logging.warning("There are missing values in the 'CLOSE' column.")
            logging.info(f"Number of missing values in 'CLOSE': {df['CLOSE'].isnull().sum()}")
        
        # Проверяем необходимые колонки
        required_columns = {'CLOSE'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise Exception(f"Missing required columns: {missing}")
        
        logging.info(f"DataFrame loaded. Shape: {df.shape}, Date range: {df.index.min()} to {df.index.max()}")
        return df
    
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {str(e)}")
        raise