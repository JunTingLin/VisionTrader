"""
比較 49 支 0050 成分股等權指數 vs 0050 ETF

【圖表】使用收盤價（Close）計算累積財富曲線，較直觀呈現走勢。
【績效指標（ASR/ARR/MDD）】使用開盤價（Open），與 plot_tw_5.py 的 inter mode 一致，
  模擬「前日收盤決策、次日開盤執行」的交易邏輯。

等權方法：先算各股日報酬率，再取算術平均（而非平均股價後再算報酬）。

輸出: src/docs/tw0050_vs_ew49.png
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

# ── 載入 49 支股票 ────────────────────────────────────────────────────────
stocks = np.load('src/data/TWII/feature5-Inter/stocks_data.npy', allow_pickle=True)
# shape: (49, 2673, 5)  columns: Open, High, Low, Close, Volume
dates = pd.bdate_range(start='2015-01-01', end='2025-03-31')  # 2673 days

open_49  = stocks[:, :, 0]   # (49, 2673)
close_49 = stocks[:, :, 3]   # (49, 2673)

# ── 【圖表用】Close 等權日報酬率 → 累積財富 ──────────────────────────────
ror_close = np.zeros_like(close_49)
ror_close[:, 1:] = (close_49[:, 1:] - close_49[:, :-1]) / np.where(close_49[:, :-1] == 0, np.nan, close_49[:, :-1])
ror_close = np.nan_to_num(ror_close, nan=0.0)
ew_ror_close = ror_close.mean(axis=0)
ew_wealth_close = np.cumprod(1 + ew_ror_close)
ew_wealth_close = ew_wealth_close / ew_wealth_close[0]

# ── 【指標用】Open 等權日報酬率 → 累積財富 ───────────────────────────────
ror_open = np.zeros_like(open_49)
ror_open[:, 1:] = (open_49[:, 1:] - open_49[:, :-1]) / np.where(open_49[:, :-1] == 0, np.nan, open_49[:, :-1])
ror_open = np.nan_to_num(ror_open, nan=0.0)
ew_ror_open = ror_open.mean(axis=0)
ew_wealth_open = np.cumprod(1 + ew_ror_open)
ew_wealth_open = ew_wealth_open / ew_wealth_open[0]

# ── 下載 0050.TW ──────────────────────────────────────────────────────────
df_0050 = yf.download('0050.TW', start='2015-01-01', end='2025-04-01',
                       auto_adjust=False, progress=False)
df_0050.reset_index(inplace=True)
df_0050.columns = df_0050.columns.droplevel(level=1)
df_0050['Date'] = pd.to_datetime(df_0050['Date']).dt.tz_localize(None)

# ── 【圖表用】0050 Close 累積財富 ─────────────────────────────────────────
df_ew = pd.DataFrame({'Date': pd.to_datetime(dates), 'EW_Close': ew_wealth_close})
df_merged = pd.merge(df_ew, df_0050[['Date', 'Close']], on='Date', how='inner')
df_merged = df_merged[df_merged['EW_Close'] > 0].reset_index(drop=True)

etf_close_ror = df_merged['Close'].pct_change().fillna(0.0)
etf_wealth_close = (1 + etf_close_ror).cumprod()
etf_wealth_close = etf_wealth_close / etf_wealth_close.iloc[0]

ew_norm  = df_merged['EW_Close'] / df_merged['EW_Close'].iloc[0] * 100
etf_norm = etf_wealth_close * 100
diff = ew_norm.values - etf_norm.values

# ── 畫圖 ──────────────────────────────────────────────────────────────────
fig_all, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})

axes[0].plot(df_merged['Date'], ew_norm,  label='49 stocks Equal-Weight (Close)', linewidth=1.2, color='steelblue')
axes[0].plot(df_merged['Date'], etf_norm, label='0050 ETF (Close)',               linewidth=1.2, color='tomato', alpha=0.85)
axes[0].set_title('49 Equal-Weighted Constituent Stocks vs 0050 ETF\n(Close price, Normalized to 100 at 2015-01-05)', fontsize=13)
axes[0].set_ylabel('Cumulative Return (Base = 100)')
axes[0].legend(fontsize=11)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[0].xaxis.set_major_locator(mdates.YearLocator())
axes[0].grid(True, linestyle='--', alpha=0.4)

axes[1].fill_between(df_merged['Date'], diff, 0,
                     where=(diff >= 0), color='steelblue', alpha=0.5, label='EW > ETF')
axes[1].fill_between(df_merged['Date'], diff, 0,
                     where=(diff < 0),  color='tomato',    alpha=0.5, label='EW < ETF')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('Spread: Equal-Weight − 0050 ETF (pp)')
axes[1].set_ylabel('Spread (pp)')
axes[1].set_xlabel('Date')
axes[1].legend(fontsize=10)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[1].xaxis.set_major_locator(mdates.YearLocator())
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
out_path = 'src/docs/tw0050_vs_ew49.png'
fig_all.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')

print(f"\n=== Chart Summary (Close) ===")
print(f"Period: {df_merged['Date'].iloc[0].date()} ~ {df_merged['Date'].iloc[-1].date()}  ({len(df_merged)} days)")
print(f"EW-49  total return: {ew_norm.iloc[-1]-100:.1f}%")
print(f"0050   total return: {etf_norm.iloc[-1]-100:.1f}%")
print(f"Max spread: {diff.max():.2f} pp  Min spread: {diff.min():.2f} pp")
print(f"Correlation: {np.corrcoef(ew_norm, etf_norm)[0,1]:.4f}")

# ── 【指標用】Open 績效指標 ───────────────────────────────────────────────
TRADE_LEN = 21
Ny = 12

def calc_metrics(wealth_slice):
    series = wealth_slice[::TRADE_LEN]
    series = series / series[0]
    w = series.reshape(1, -1)
    ror = w[:, 1:] / w[:, :-1] - 1
    ARR = np.mean(ror) * Ny
    AVOL = np.std(ror) * math.sqrt(Ny)
    ASR = ARR / AVOL
    dd = (np.maximum.accumulate(w, axis=-1) - w) / np.maximum.accumulate(w, axis=-1)
    MDD = np.max(dd)
    return ASR, ARR, MDD

val_asr, val_arr, val_mdd   = calc_metrics(ew_wealth_open[1304:2087])
test_asr, test_arr, test_mdd = calc_metrics(ew_wealth_open[2087:2673])

print(f"\n=== Metrics (Open, inter mode) ===")
print(f"EW-49 Val:  ASR={val_asr:.2f}  ARR={val_arr*100:.2f}%  MDD={val_mdd*100:.2f}%")
print(f"EW-49 Test: ASR={test_asr:.2f}  ARR={test_arr*100:.2f}%  MDD={test_mdd*100:.2f}%")

# 存 CSV（Close 供圖表、Open 供績效指標）
df_save = pd.DataFrame({
    'Date': dates,
    'EW_Close_Wealth': ew_wealth_close,
    'EW_Open_Wealth':  ew_wealth_open,
})
df_save.to_csv('src/docs/ew49_wealth.csv', index=False)
print(f"Saved: src/docs/ew49_wealth.csv")
