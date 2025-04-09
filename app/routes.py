# app/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .models import LSTMModel, load_data, train_model, predict_stock_price, predict_future_prices
from .utils import download_data
import torch
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import joblib

logging.basicConfig(level=logging.INFO)

router = APIRouter()

class PredictionRequest(BaseModel):
    ticker: str
    date: str

def prepare_dataset(df):
    df = df[['CLOSE']].dropna()
    dataset = df.values.astype('float32')
    return df, dataset

def load_or_train_model(ticker):
    model_save_path = os.path.join('models', f'{ticker}_model.pth')
    scaler_save_path = os.path.join('models', f'{ticker}_scaler.pkl')

    if not os.path.exists(model_save_path) or not os.path.exists(scaler_save_path):
        logging.info("Model not found. Training new model...")
        model, scaler = train_model(ticker)
        joblib.dump(scaler, scaler_save_path)
        torch.save(model.state_dict(), model_save_path)
    else:
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        scaler = joblib.load(scaler_save_path)
    
    return model, scaler

def predict_historical_price(model, scaler, dataset, date, df):
    date_index = df.index.get_loc(date)
    if date_index < 60:
        raise Exception(f"Not enough data before {date}. Need 60 days before date.")
    
    last_60_days = dataset[date_index - 60:date_index]
    predicted_price = predict_stock_price(model, scaler, last_60_days)
    logging.info(f"Predicted price for {date}: {predicted_price}")
    
    return {
        "predicted_price": float(predicted_price),
        "is_future": False
    }

def predict_future_price(model, scaler, dataset, date, df):
    days_ahead = (date - df.index[-1]).days
    if days_ahead <= 0:
        raise Exception(f"Date {date} is before the last available date {df.index[-1].strftime('%Y-%m-%d')}")
    
    last_60_days = dataset[-60:]
    future_prices = predict_future_prices(model, scaler, last_60_days, steps=days_ahead)
    predicted_price = future_prices[-1]
    logging.info(f"Predicted future price for {date}: {predicted_price}")
    
    return {
        "predicted_price": float(predicted_price),
        "is_future": True,
        "days_ahead": days_ahead
    }

@router.post("/predict/")
async def predict_price(request: PredictionRequest):
    ticker = request.ticker
    date_str = request.date

    logging.info(f"Received request: ticker={ticker}, date={date_str}")

    try:
        date = pd.to_datetime(date_str)
        start_date = '2014-01-01'
        end_date = (date + timedelta(days=365)).strftime('%Y-%m-%d')
        
        download_data(ticker, start_date=start_date, end_date=end_date)
        logging.info(f"Data downloaded for ticker: {ticker}")
        
        df = load_data(ticker)
        logging.info(f"Data loaded for ticker: {ticker}. Date range: {df.index.min()} to {df.index.max()}")
        
        # Заполняем пропущенные даты
        date_range = pd.date_range(start=df.index.min(), end=max(df.index.max(), date), freq='D')
        df = df.reindex(date_range)
        df['CLOSE'] = df['CLOSE'].interpolate(method='time')
        
        df, dataset = prepare_dataset(df)

        if len(dataset) < 60:
            raise Exception(f"Not enough data for prediction. Need 60 days, have {len(dataset)}")
        
        model, scaler = load_or_train_model(ticker)
        
        # Теперь все даты заполнены, можно предсказывать
        if date <= df.index[-1]:
            return predict_historical_price(model, scaler, dataset, date, df)
        else:
            return predict_future_price(model, scaler, dataset, date, df)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get_min_date/")
async def get_min_date(ticker: str):
    try:
        df = load_data(ticker)
        df, _ = prepare_dataset(df)
        
        if len(df) < 60:
            raise Exception(f"Not enough data (need 60 days, have {len(df)})")
        
        min_date = df.index[59]  # Первая дата, для которой есть 60 предыдущих дней
        max_date = df.index[-1]
        
        logging.info(f"Date range for {ticker}: {min_date} to {max_date}")
        return {
            "min_date": min_date.strftime('%Y-%m-%d'),
            "max_date": max_date.strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        logging.error(f"Error getting date range: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))