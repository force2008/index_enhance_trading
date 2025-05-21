# 环境依赖 
python 3.10


# 包依赖
pip install akshare pandas backtrader numpy TA-Lib joblib openpyxl plotly tqdm tables

# 执行

data/csi500_data_cache.pkl 这个文件可能要手动删一下，这个是缓存股票代码和权重的缓存文件，如果getData后面的代码的变量，需要把这个文件删了，使能跑到后面的代码中

python index_enhance_trading_with_weights.py

# 结果大致如下：
cumulative_returns.html是累计收益率的图表展示

回测时间20200104 ~ 20221230

最终资金: 1790614.04

总回报: 58.26%

年化回报: 11.21%

最大回撤: -26.86%

总交易次数: 338

# 提交日志 5-21
- 拆分的数据下到到data_download_clean.py中
- 回测放到了backtest.py
- 在因子上增加到了21个因子，但对因子的权重并没有做调整，所以回测效果并不好
最终资金: 2056622.53

总回报: 72.11%

年化回报: 13.39%

最大回撤: -34.20%

总交易次数: 1400