import akshare as ak
import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dividend_yield_fix.log',
    filemode='w'
)

def create_fallback_dividend_data(start_date, end_date):
    """创建零填充的股息率DataFrame"""
    return pd.DataFrame(
        index=pd.date_range(start=start_date, end=end_date, freq='D'),
        columns=['dividend_yield'],
        data=0
    )

try:
    symbol = "000009"
    start_date = "20200104"
    end_date = "20221230"

    # 获取日K线数据
    logging.info(f"Fetching daily data for {symbol}")
    stock_data = ak.stock_zh_a_daily(symbol=f"sz{symbol}", adjust="qfq", start_date=start_date, end_date=end_date)
    if stock_data is None or stock_data.empty:
        logging.warning(f"股票 {symbol} 日K线数据为空")
        dividend_data = create_fallback_dividend_data(start_date, end_date)
        print(dividend_data)
        raise ValueError("日K线数据为空")
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    stock_data.set_index('date', inplace=True)
    stock_data['close'] = stock_data['close'].ffill().bfill()
    if stock_data['close'].isna().any() or (stock_data['close'] <= 0).any():
        logging.warning(f"股票 {symbol} 收盘价无效")
        dividend_data = create_fallback_dividend_data(start_date, end_date)
        print(dividend_data)
        raise ValueError("收盘价无效")

    # 获取股息数据
    logging.info(f"Fetching dividend data for {symbol}")
    dividend_data = ak.stock_dividend_cninfo(symbol=symbol)
    print(dividend_data)
    if dividend_data is None or dividend_data.empty or '实施方案分红说明' not in dividend_data.columns or '股权登记日' not in dividend_data.columns:
        logging.warning(f"股票 {symbol} 股息数据为空或缺少必要列")
        dividend_data = create_fallback_dividend_data(start_date, end_date)
        print(dividend_data)
        raise ValueError("股息数据无效")

    # 处理日期并过滤
    logging.info(f"Processing dividend dates for {symbol}")
    dividend_data['date'] = pd.to_datetime(dividend_data['股权登记日'], errors='coerce')
    dividend_data = dividend_data[dividend_data['date'].notna()]
    dividend_data.set_index('date', inplace=True)
    dividend_data = dividend_data[(dividend_data.index >= pd.to_datetime(start_date)) & (dividend_data.index <= pd.to_datetime(end_date))]

    if dividend_data.empty:
        logging.warning(f"股票 {symbol} 股息数据在指定日期范围内为空")
        dividend_data = create_fallback_dividend_data(start_date, end_date)
        print(dividend_data)
        raise ValueError("股息数据为空")

    def parse_dividend_scheme(scheme):
        if pd.isna(scheme) or not isinstance(scheme, str):
            return 0
        match = re.search(r'10\s*派\s*(\d+\.?\d*)', scheme)
        return float(match.group(1)) / 10 if match else 0

    # 解析股息
    logging.info(f"Parsing dividend scheme for {symbol}")
    dividend_data['per_share_dividend'] = dividend_data['实施方案分红说明'].apply(parse_dividend_scheme)
    if dividend_data['per_share_dividend'].isna().all() or (dividend_data['per_share_dividend'] == 0).all():
        logging.warning(f"股票 {symbol} 股息数据解析失败")
        dividend_data = create_fallback_dividend_data(start_date, end_date)
        print(dividend_data)
        raise ValueError("股息解析失败")

    # 按年分组
    logging.info(f"Grouping dividend data by year for {symbol}")
    grouped_data = dividend_data.groupby(dividend_data.index.to_period('Y')).agg({
        'per_share_dividend': 'sum'
    }).reset_index()

    # 验证分组结果
    # if grouped_data.empty or 'index' not in grouped_data.columns:
    #     logging.warning(f"股票 {symbol} 分组后数据为空或缺少'index'列")
    #     dividend_data = create_fallback_dividend_data(start_date, end_date)
    #     print(dividend_data)
    #     raise ValueError("分组数据无效")

    # 设置日期
    grouped_data['date'] = grouped_data['date'].apply(lambda x: x.to_timestamp(how='end'))
    # dividend_data = grouped_data.set_index('date').drop(columns=['index'])

    # 计算股息率
    logging.info(f"Calculating dividend yield for {symbol}")
    dividend_data = dividend_data.join(stock_data['close'], how='left')
    dividend_data['dividend_yield'] = dividend_data['per_share_dividend'] / dividend_data['close'].replace(0, np.nan)
    dividend_data['dividend_yield'] = dividend_data['dividend_yield'].replace([np.inf, -np.inf], 0).clip(lower=0)

    # 扩展到每日数据
    base_index = pd.date_range(start=start_date, end=end_date, freq='D')
    dividend_data = dividend_data.reindex(base_index).ffill().bfill()
    dividend_data['dividend_yield'] = dividend_data['dividend_yield'].fillna(0)

    # print(dividend_data)
    logging.info(dividend_data)
    logging.info(f"Successfully processed dividend data for {symbol}")

except Exception as e:
    logging.error(f"股票 {symbol} 股息率数据获取失败: {str(e)}")
    dividend_data = create_fallback_dividend_data(start_date, end_date)
    print(dividend_data)