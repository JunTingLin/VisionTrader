"""
比較 49 支 0050 成分股等權平均收盤價 vs 0050 ETF 的累積報酬走勢
輸出: src/docs/tw0050_vs_ew49.png
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. 載入 49 支股票的 OHLCV ──────────────────────────────────────────────
stocks = np.load('src/data/TWII/feature5-Inter/stocks_data.npy', allow_pickle=True)
# shape: (49, 2673, 5)  columns: Open, High, Low, Close, Volume
dates = pd.bdate_range(start='2015-01-01', end='2025-03-31')  # 2673 days

close_49 = stocks[:, :, 3]           # (49, 2673)  Close price
open_49  = stocks[:, :, 0]           # (49, 2673)  Open price

# 等權平均收盤價
ew_close = close_49.mean(axis=0)     # (2673,)

# ── 2. 下載 0050.TW ────────────────────────────────────────────────────────
df_0050 = yf.download('0050.TW', start='2015-01-01', end='2025-04-01',
                       auto_adjust=False, progress=False)
df_0050.reset_index(inplace=True)
df_0050.columns = df_0050.columns.droplevel(level=1)
df_0050['Date'] = pd.to_datetime(df_0050['Date']).dt.tz_localize(None)

# ── 3. 對齊至相同交易日 ────────────────────────────────────────────────────
df_ew = pd.DataFrame({'Date': dates, 'EW_Close': ew_close})
df_ew['Date'] = pd.to_datetime(df_ew['Date'])

df_merged = pd.merge(df_ew, df_0050[['Date', 'Close']], on='Date', how='inner')
df_merged.rename(columns={'Close': 'ETF_Close'}, inplace=True)
df_merged = df_merged[df_merged['EW_Close'] > 0].reset_index(drop=True)

# ── 4. 標準化（以第一筆 = 100）────────────────────────────────────────────
ew_norm  = df_merged['EW_Close']  / df_merged['EW_Close'].iloc[0]  * 100
etf_norm = df_merged['ETF_Close'] / df_merged['ETF_Close'].iloc[0] * 100

# ── 5. 畫圖 ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df_merged['Date'], ew_norm,  label='49 stocks Equal-Weight', linewidth=1.2, color='steelblue')
ax.plot(df_merged['Date'], etf_norm, label='0050 ETF',               linewidth=1.2, color='tomato', alpha=0.85)

ax.set_title('49 Equal-Weighted Constituent Stocks vs 0050 ETF\n(Normalized to 100 at 2015-01-05)',
             fontsize=13)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (Base = 100)')
ax.legend(fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.grid(True, linestyle='--', alpha=0.4)

# 差值子圖
fig2, ax2 = plt.subplots(figsize=(14, 3))
diff = ew_norm.values - etf_norm.values
ax2.fill_between(df_merged['Date'], diff, 0,
                 where=(diff >= 0), color='steelblue', alpha=0.5, label='EW > ETF')
ax2.fill_between(df_merged['Date'], diff, 0,
                 where=(diff < 0),  color='tomato',    alpha=0.5, label='EW < ETF')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_title('Spread: Equal-Weight − 0050 ETF')
ax2.set_xlabel('Date')
ax2.set_ylabel('Spread (points)')
ax2.legend(fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.grid(True, linestyle='--', alpha=0.4)

# 存成合併圖
fig_all, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})

axes[0].plot(df_merged['Date'], ew_norm,  label='49 stocks Equal-Weight', linewidth=1.2, color='steelblue')
axes[0].plot(df_merged['Date'], etf_norm, label='0050 ETF',               linewidth=1.2, color='tomato', alpha=0.85)
axes[0].set_title('49 Equal-Weighted Constituent Stocks vs 0050 ETF\n(Normalized to 100 at 2015-01-05)', fontsize=13)
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
axes[1].set_title('Spread: Equal-Weight − 0050 ETF')
axes[1].set_ylabel('Spread (points)')
axes[1].set_xlabel('Date')
axes[1].legend(fontsize=10)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[1].xaxis.set_major_locator(mdates.YearLocator())
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
out_path = 'src/docs/tw0050_vs_ew49.png'
fig_all.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')

# 統計數字
print(f"\n=== Summary ===")
print(f"Period: {df_merged['Date'].iloc[0].date()} ~ {df_merged['Date'].iloc[-1].date()}  ({len(df_merged)} trading days)")
print(f"EW-49  total return: {ew_norm.iloc[-1]-100:.1f}%")
print(f"0050   total return: {etf_norm.iloc[-1]-100:.1f}%")
print(f"Max spread (EW-ETF): {diff.max():.2f}")
print(f"Min spread (EW-ETF): {diff.min():.2f}")
print(f"Correlation: {np.corrcoef(ew_norm, etf_norm)[0,1]:.4f}")
