import akshare as ak
import pandas as pd
import numpy as np
symbol = 'sh603843'
start_date = '20210101'
end_date = '20221230'
stock_data = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq", start_date=start_date, end_date=end_date)

print(stock_data.head(5))
if stock_data.empty:
    print(f"股票 {symbol} 日K线数据为空")
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data = stock_data.rename(columns={
    'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
})
stock_data.set_index('date', inplace=True)
stock_data['volume'] = stock_data['volume'].replace([np.inf, -np.inf], 0).fillna(0)

# 增强清洗
stock_data['close'] = stock_data['close'].ffill().bfill()
if stock_data['close'].isna().any() or (stock_data['close'] <= 0).any():
    mean_close = stock_data['close'].mean()
    if pd.isna(mean_close) or mean_close <= 0:
        print(f"股票 {symbol} 无法计算有效均价")
    stock_data['close'] = stock_data['close'].fillna(mean_close)
    if stock_data['close'].isna().any() or (stock_data['close'] <= 0).any():
        print(f"股票 {symbol} 日K线数据填充后仍无效")
base_index = pd.date_range(start=start_date, end=end_date, freq='D')
stock_data = stock_data.reindex(base_index).ffill().bfill()
stock_data['volume'] = stock_data['volume'].fillna(0)
stock_data['close'] = stock_data['close'].fillna(stock_data['close'].mean())
stock_data['open'] = stock_data['open'].fillna(stock_data['close'])
stock_data['high'] = stock_data['high'].fillna(stock_data['close'])
stock_data['low'] = stock_data['low'].fillna(stock_data['close'])
cleaned_daily_file = f"data01/daily/stock_test_{symbol}_cleaned.csv"
stock_data.index.name = 'date'
stock_data.to_csv(cleaned_daily_file, encoding='utf-8')
# print(stock_data)