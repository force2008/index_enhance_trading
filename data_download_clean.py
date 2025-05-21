import akshare as ak
import pandas as pd
import logging
import numpy as np
import os
import time
from tqdm import tqdm
from datetime import datetime
import re

# 设置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file = 'data_clean/data_download.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
if os.path.exists(log_file):
    os.remove(log_file)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
pd.set_option('future.no_silent_downcasting', True)
# 创建数据目录
os.makedirs("data_clean/daily", exist_ok=True)
os.makedirs("data_clean/pe_pb", exist_ok=True)
os.makedirs("data_clean/financial", exist_ok=True)
os.makedirs("data_clean/dividend", exist_ok=True)
os.makedirs("data_clean/index", exist_ok=True)

# 数据层：获取中证500成分股和数据
def get_stock_code(symbol, exchange):
    """将6位股票代码转换为带交易所前缀的格式"""
    if exchange == "上海证券交易所":
        return f"sh{symbol}"
    elif exchange == "深圳证券交易所":
        return f"sz{symbol}"
    return symbol

def download_and_clean_data():
    start_date = "20200104"
    end_date = "20221230"
    
    # 检查缓存
    cache_file = "data_clean/csi500_data_cache.pkl"
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        cached_data = pd.read_pickle(cache_file)
        symbols = cached_data['symbols']
        csi500_constituents = cached_data['index_weights']
        logging.info("Loaded cached symbols and index weights")
    else:
        # 获取中证500成分股
        csi500_constituents = ak.index_stock_cons_weight_csindex(symbol="000905")
        if csi500_constituents is None or csi500_constituents.empty:
            logging.error("CSI 500 constituents data is empty or None")
            raise ValueError("CSI 500 constituents data is empty or None")
        csi500_constituents['stock_code'] = csi500_constituents.apply(
            lambda x: get_stock_code(x['成分券代码'], x['交易所']), axis=1
        )
        csi500_constituents['日期'] = pd.to_datetime(csi500_constituents['日期'])
        csi500_constituents.to_csv("data_clean/index/csi500_weights.csv", encoding='utf-8')
        time.sleep(2)
        symbols = csi500_constituents['stock_code'].unique().tolist()
        logging.info(f"Retrieved {len(symbols)} CSI 500 constituents")

    # 获取指数数据
    index_file = "data_clean/index/csi500_index.csv"
    if os.path.exists(index_file) and os.path.getsize(index_file) > 0:
        index_data = pd.read_csv(index_file, parse_dates=['date'], index_col='date')
        logging.info("Loaded existing CSI 500 index data")
    else:
        index_data = ak.index_zh_a_hist(symbol="000905", period="daily", start_date=start_date, end_date=end_date)
        if index_data is None or index_data.empty:
            logging.error("Index data is empty or None")
            raise ValueError("Index data is empty or None")
        index_data['date'] = pd.to_datetime(index_data['日期'])
        index_data = index_data.rename(columns={
            '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'
        })
        index_data.set_index('date', inplace=True)
        index_data['open'] = index_data['open'].ffill().bfill()
        index_data['high'] = index_data['high'].ffill().bfill()
        index_data['low'] = index_data['low'].ffill().bfill()
        index_data['close'] = index_data['close'].ffill().bfill()
        index_data['volume'] = index_data['volume'].replace([np.inf, -np.inf], 0).fillna(0)
        index_data = index_data.sort_index()
        if index_data['close'].isna().any() or (index_data['close'] <= 0).any():
            logging.error("Index data contains invalid close prices")
            raise ValueError("Index data contains invalid close prices")
        index_data.index.name = 'date'
        index_data.to_csv(index_file, encoding='utf-8')
        time.sleep(2)
        logging.info("Downloaded and saved CSI 500 index data")

    # 合并日K线数据到 HDF5
    hdf_file = "data_clean/daily_cleaned.h5"
    valid_symbols = []
    with pd.HDFStore(hdf_file, mode='w') as store:
        for symbol in tqdm(symbols, desc="Processing stocks", unit="stock"):
            cleaned_daily_file = f"data_clean/daily/stock_{symbol}_cleaned.csv"
            cleaned_pe_pb_file = f"data_clean/pe_pb/stock_{symbol}_pe_pb_cleaned.csv"
            cleaned_financial_file = f"data_clean/financial/stock_{symbol}_financial_cleaned.csv"
            cleaned_dividend_file = f"data_clean/dividend/stock_{symbol}_dividend_cleaned.csv"

            # 初始化数据变量
            stock_data = None
            pe_pb_data = None
            financial_data = None
            dividend_data = None

            symbol_code = symbol[2:]
            try:
                # 股票信息
                time.sleep(1)
                stock_info = ak.stock_individual_info_em(symbol=symbol_code)
                if stock_info is None or stock_info.empty:
                    logging.warning(f"股票 {symbol} 信息获取失败")
                    continue

                # 日K线数据
                if os.path.exists(cleaned_daily_file) and os.path.getsize(cleaned_daily_file) > 0:
                    try:
                        stock_data = pd.read_csv(cleaned_daily_file, parse_dates=['date'], index_col='date')
                        if stock_data['close'].isna().any() or (stock_data['close'] <= 0).any() or stock_data['close'].lt(0).any():
                            logging.warning(f"股票 {symbol} 现有日K线数据无效")
                            stock_data = None
                        else:
                            logging.info(f"Loaded existing daily data for {symbol}")
                    except Exception as e:
                        logging.warning(f"股票 {symbol} 读取现有日K线CSV失败: {str(e)}")
                        stock_data = None

                if stock_data is None:
                    time.sleep(1)
                    stock_data = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq", start_date=start_date, end_date=end_date)
                    if stock_data is None or stock_data.empty:
                        logging.warning(f"股票 {symbol} 日K线数据为空")
                        continue
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                    stock_data = stock_data.rename(columns={
                        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
                    })
                    stock_data.set_index('date', inplace=True)
                    stock_data['volume'] = stock_data['volume'].replace([np.inf, -np.inf], 0).fillna(0)
                    stock_data['close'] = stock_data['close'].ffill().bfill()
                    if stock_data['close'].isna().any() or (stock_data['close'] <= 0).any() or stock_data['close'].lt(0).any():
                        mean_close = stock_data['close'].mean()
                        if pd.isna(mean_close) or mean_close <= 0:
                            logging.warning(f"股票 {symbol} 收盘价无效")
                            continue
                        stock_data['close'] = stock_data['close'].fillna(mean_close)
                        if stock_data['close'].lt(0).any():
                            logging.warning(f"股票 {symbol} 包含负收盘价")
                            continue
                    logging.info(f"Downloaded daily data for {symbol}")

                # PE/PB 数据
                if os.path.exists(cleaned_pe_pb_file) and os.path.getsize(cleaned_pe_pb_file) > 0:
                    try:
                        pe_pb_data = pd.read_csv(cleaned_pe_pb_file, parse_dates=['date'], index_col='date')
                        if pe_pb_data['pe'].isna().any() or pe_pb_data['pb'].isna().any():
                            logging.warning(f"股票 {symbol} 现有PE/PB数据无效")
                            pe_pb_data = None
                        else:
                            logging.info(f"Loaded existing pe_pb data for {symbol}")
                    except Exception as e:
                        logging.warning(f"股票 {symbol} 读取现有PE/PB CSV失败: {str(e)}")
                        pe_pb_data = None

                if pe_pb_data is None:
                    time.sleep(1)
                    pe_pb_data = ak.stock_a_indicator_lg(symbol=symbol_code)
                    if pe_pb_data is None or pe_pb_data.empty:
                        logging.warning(f"股票 {symbol} PE/PB 数据为空")
                        continue
                    pe_pb_data['date'] = pd.to_datetime(pe_pb_data['trade_date'])
                    pe_pb_data.set_index('date', inplace=True)
                    pe_pb_data = pe_pb_data[(pe_pb_data.index >= pd.to_datetime(start_date)) & (pe_pb_data.index <= pd.to_datetime(end_date))]
                    logging.info(f"Downloaded pe_pb data for {symbol}")

                # 财务数据（包括现金流）
                if os.path.exists(cleaned_financial_file) and os.path.getsize(cleaned_financial_file) > 0:
                    try:
                        financial_data = pd.read_csv(cleaned_financial_file, parse_dates=['date'], index_col='date')
                        if financial_data['cash_flow_to_assets'].isna().all():
                            logging.warning(f"股票 {symbol} 现有财务数据缺少cash_flow_to_assets")
                            financial_data = None
                        else:
                            logging.info(f"Loaded existing financial data for {symbol}")
                    except Exception as e:
                        logging.warning(f"股票 {symbol} 读取现有财务CSV失败: {str(e)}")
                        financial_data = None

                if financial_data is None:
                    time.sleep(1)
                    financial_data = ak.stock_financial_analysis_indicator(symbol=symbol_code, start_year="2020")
                    if financial_data is None or financial_data.empty:
                        logging.warning(f"股票 {symbol} 财务数据为空")
                        continue
                    financial_data['date'] = pd.to_datetime(financial_data['日期'])
                    financial_data.set_index('date', inplace=True)
                    financial_data = financial_data[(financial_data.index >= pd.to_datetime(start_date)) & (financial_data.index <= pd.to_datetime(end_date))]

                    # 经营现金流数据
                    time.sleep(1)
                    cash_flow_data = ak.stock_financial_report_sina(stock=symbol_code, symbol="现金流量表")
                    if cash_flow_data is None or cash_flow_data.empty:
                        cash_flow_data = pd.DataFrame({'报告日': [], '经营活动产生的现金流量净额': []})
                        logging.warning(f"股票 {symbol} 现金流数据为空")
                    cash_flow_data['date'] = pd.to_datetime(cash_flow_data['报告日'])
                    cash_flow_data.set_index('date', inplace=True)
                    cash_flow_data = cash_flow_data[(cash_flow_data.index >= pd.to_datetime(start_date)) & (cash_flow_data.index <= pd.to_datetime(end_date))]

                    # 资产总计数据（从资产负债表）
                    time.sleep(1)
                    balance_sheet_data = ak.stock_financial_report_sina(stock=symbol_code, symbol="资产负债表")
                    if balance_sheet_data is None or balance_sheet_data.empty:
                        balance_sheet_data = pd.DataFrame({'报告日': [], '资产总计': []})
                        logging.warning(f"股票 {symbol} 资产负债表数据为空")
                    balance_sheet_data['date'] = pd.to_datetime(balance_sheet_data['报告日'])
                    balance_sheet_data.set_index('date', inplace=True)
                    balance_sheet_data = balance_sheet_data[(balance_sheet_data.index >= pd.to_datetime(start_date)) & (balance_sheet_data.index <= pd.to_datetime(end_date))]

                    # 合并现金流和资产数据
                    cash_flow_data = cash_flow_data.join(balance_sheet_data[['资产总计']], how='outer')
                    cash_flow_data['cash_flow_to_assets'] = (cash_flow_data['经营活动产生的现金流量净额'] / cash_flow_data['资产总计']) * 100
                    cash_flow_data['cash_flow_to_assets'] = cash_flow_data['cash_flow_to_assets'].replace([np.inf, -np.inf], 0).fillna(0)
                    financial_data = financial_data.join(cash_flow_data[['cash_flow_to_assets']], how='outer')
                    logging.info(f"Downloaded financial, cash flow, and balance sheet data for {symbol}")

                # 股息率数据
                if os.path.exists(cleaned_dividend_file) and os.path.getsize(cleaned_dividend_file) > 0:
                    try:
                        dividend_data = pd.read_csv(cleaned_dividend_file, parse_dates=['date'], index_col='date')
                        if dividend_data['dividend_yield'].isna().all():
                            logging.warning(f"股票 {symbol} 现有股息率数据无效")
                            dividend_data = None
                        else:
                            logging.info(f"Loaded existing dividend data for {symbol}")
                    except Exception as e:
                        logging.warning(f"股票 {symbol} 读取现有股息率CSV失败: {str(e)}")
                        dividend_data = None

                if dividend_data is None:
                    try:
                        time.sleep(1)
                        dividend_data = ak.stock_dividend_cninfo(symbol=symbol_code)
                        if dividend_data is None or dividend_data.empty or '实施方案分红说明' not in dividend_data.columns or '股权登记日' not in dividend_data.columns:
                            logging.warning(f"股票 {symbol} 股息数据为空或缺少必要列")
                            dividend_data = create_fallback_dividend_data(start_date, end_date)
                            raise ValueError("股息数据无效")
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
                        logging.info(f"Downloaded dividend data for {symbol}")
                    except Exception as e:
                        logging.warning(f"股票 {symbol} 股息率数据获取失败: {str(e)}")
                        dividend_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'), columns=['dividend_yield'])
                        dividend_data['dividend_yield'] = 0

                # 确保所需字段
                required_cols = ['净资产收益率(%)', '主营业务收入增长率(%)', '资产负债率(%)',
                                '净利润同比增长(%)', '总资产增长率(%)', '销售毛利率(%)',
                                '流动资产周转率(次)', 'cash_flow_to_assets']
                for col in required_cols:
                    if col not in financial_data.columns:
                        financial_data[col] = 0

                # 对齐日期
                base_index = pd.date_range(start=start_date, end=end_date, freq='D')
                stock_data = stock_data.reindex(base_index).ffill().bfill()
                pe_pb_data = pe_pb_data.reindex(base_index).ffill().bfill()
                financial_data = financial_data.reindex(base_index).ffill().bfill()

                # 填充缺失值
                stock_data['volume'] = stock_data['volume'].fillna(0)
                stock_data['close'] = stock_data['close'].fillna(stock_data['close'].mean())
                stock_data['open'] = stock_data['open'].fillna(stock_data['close'])
                stock_data['high'] = stock_data['high'].fillna(stock_data['close'])
                stock_data['low'] = stock_data['low'].fillna(stock_data['close'])
                pe_pb_data['pe'] = pe_pb_data['pe'].fillna(pe_pb_data['pe'].mean()).fillna(20)
                pe_pb_data['pb'] = pe_pb_data['pb'].fillna(pe_pb_data['pb'].mean()).fillna(2)
                financial_data['净资产收益率(%)'] = financial_data['净资产收益率(%)'].fillna(financial_data['净资产收益率(%)'].mean()).fillna(0)
                financial_data['主营业务收入增长率(%)'] = financial_data['主营业务收入增长率(%)'].fillna(financial_data['主营业务收入增长率(%)'].mean()).fillna(0)
                financial_data['资产负债率(%)'] = financial_data['资产负债率(%)'].fillna(financial_data['资产负债率(%)'].mean()).fillna(50)
                financial_data['净利润同比增长(%)'] = financial_data['净利润同比增长(%)'].fillna(financial_data['净利润同比增长(%)'].mean()).fillna(0)
                financial_data['总资产增长率(%)'] = financial_data['总资产增长率(%)'].fillna(financial_data['总资产增长率(%)'].mean()).fillna(0)
                financial_data['销售毛利率(%)'] = financial_data['销售毛利率(%)'].fillna(financial_data['销售毛利率(%)'].mean()).fillna(0)
                financial_data['流动资产周转率(次)'] = financial_data['流动资产周转率(次)'].fillna(financial_data['流动资产周转率(次)'].mean()).fillna(0)
                financial_data['cash_flow_to_assets'] = financial_data['cash_flow_to_assets'].fillna(financial_data['cash_flow_to_assets'].mean()).fillna(0)
                dividend_data['dividend_yield'] = dividend_data['dividend_yield'].fillna(0)

                # 验证数据完整性
                if (stock_data['close'].isna().any() or (stock_data['close'] <= 0).any() or stock_data['close'].lt(0).any() or
                    pe_pb_data['pe'].isna().any() or pe_pb_data['pb'].isna().any() or
                    financial_data['净资产收益率(%)'].isna().any() or
                    financial_data['主营业务收入增长率(%)'].isna().any() or
                    financial_data['资产负债率(%)'].isna().any() or
                    financial_data['净利润同比增长(%)'].isna().any() or
                    financial_data['总资产增长率(%)'].isna().any() or
                    financial_data['销售毛利率(%)'].isna().any() or
                    financial_data['流动资产周转率(次)'].isna().any() or
                    financial_data['cash_flow_to_assets'].isna().any() or
                    dividend_data['dividend_yield'].isna().any()):
                    logging.warning(f"股票 {symbol} 数据不完整")
                    continue

                stock_data.index.name = 'date'
                pe_pb_data.index.name = 'date'
                financial_data.index.name = 'date'
                dividend_data.index.name = 'date'

                # 保存数据
                stock_data.to_csv(cleaned_daily_file, encoding='utf-8')
                pe_pb_data.to_csv(cleaned_pe_pb_file, encoding='utf-8')
                financial_data.to_csv(cleaned_financial_file, encoding='utf-8')
                dividend_data.to_csv(cleaned_dividend_file, encoding='utf-8')
                store[f'stock_{symbol}'] = stock_data
                valid_symbols.append(symbol)
                logging.info(f"Processed and saved data for {symbol}")
            except Exception as e:
                logging.error(f"股票 {symbol} 数据处理失败: {str(e)}")
                continue
    
    # 保存缓存
    pd.to_pickle({'symbols': valid_symbols, 'index_weights': csi500_constituents}, cache_file)
    logging.info(f"Saved {len(valid_symbols)} valid symbols to cache")
    return valid_symbols, csi500_constituents

if __name__ == "__main__":
    valid_symbols, csi500_constituents = download_and_clean_data()
    print(f"Processed {len(valid_symbols)} stocks")