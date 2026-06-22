# VisionTrader: Portfolio Optimization with Deep Reinforcement Learning and Vision Transformer

> **Under Review** — Submitted to IEEE Access

**Chin-Yu M. Chuang<sup>1</sup>, Jun-Ting Lin<sup>1</sup>, Yung-Yaw Chen<sup>2</sup>, Yi-Ren Yeh<sup>3</sup>, Chung-Yuan Huang<sup>4</sup>, and Jyh-Shing R. Jang<sup>1</sup>**

<sup>1</sup> Department of Computer Science and Information Engineering, National Taiwan University  
<sup>2</sup> Department of Electrical Engineering, National Taiwan University  
<sup>3</sup> Department of Mathematics, National Kaohsiung Normal University  
<sup>4</sup> Department of Computer Science and Information Engineering, Chang Gung University

Corresponding author: Chin-Yu M. Chuang (d12922014@csie.ntu.edu.tw)  
Supported by E.SUN COMMERCIAL BANK under Grant 112HZA3S001

---

## Data

Each dataset folder under `src/data/` contains four files:

| File | Shape | Description |
|------|-------|-------------|
| `stocks_data.npy` | `(N, T, F)` | Per-asset features |
| `ror.npy` | `(N, T)` | Rate of return |
| `industry_classification.npy` | `(N, N)` | Adjacency matrix built from `ror[:, :1000]` |
| `market_data.npy` | `(T, M)` | Market-level features (index, VIX, bonds, FX, etc.) |

> Intra-day return: (close − open) / open
>
> Inter-day return: (open_t − open_{t-1}) / open_{t-1}

### DJIA

| Folder | N | T | F (ASU) | M (MSU) | Period |
|--------|---|---|---------|---------|--------|
| `feature5-Inter` | 28 | 6260 | 5 | 4 | 2000/01/01 ~ 2023/12/31 |
| `feature5-Intra` | 28 | 6260 | 5 | 4 | 2000/01/01 ~ 2023/12/31 |
| `feature34-Inter` | 28 | 6260 | 34 | 27 | 2000/01/01 ~ 2023/12/31 |
| `feature34-Intra` | 28 | 6260 | 34 | 27 | 2000/01/01 ~ 2023/12/31 |
| `feature5-Inter-p532` | 30 | 2673 | 5 | 4 | 2015/01/01 ~ 2025/03/31 |
| `feature34-Inter-P532` | 30 | 2673 | 34 | 27 | 2015/01/01 ~ 2025/03/31 |

### TWII

| Folder | N | T | F (ASU) | M (MSU) | Period |
|--------|---|---|---------|---------|--------|
| `feature5-Inter` | 49 | 2673 | 5 | 4 | 2015/01/01 ~ 2025/03/31 |
| `feature5-Intra` | 49 | 2673 | 5 | 4 | 2015/01/01 ~ 2025/03/31 |
| `feature34-Inter` | 49 | 2673 | 34 | 26 | 2015/01/01 ~ 2025/03/31 |
| `feature34-Intra` | 49 | 2673 | 34 | 26 | 2015/01/01 ~ 2025/03/31 |

> The TWII stock pool contains 49 out of 50 Taiwan 0050 ETF constituents (excludes 6669 緯穎, IPO 2017-11-13). See [src/docs/tw0050_stock_pool.md](src/docs/tw0050_stock_pool.md) for details.

> 💡 Use [inspect_npy_file.py](src/inspect_npy_file.py) to inspect distributions, NaN, Inf, and zero counts of any `.npy` file.

## Missing Value Imputation

### Imputation Strategy by Data Type

📈 **OHLCV (price & volume)** — zero values are treated as missing

1. `0 → NaN`
2. `ffill` — forward fill
3. `bfill` — backward fill (handles leading NaNs)
4. `fillna(0)` — fill any remaining NaNs
5. `Inf → 0`

📊 **Technical indicators (MA, RSI, MACD, etc.)** — rolling windows produce leading NaNs

1. `Inf → NaN`
2. `bfill` — resolves leading NaNs (e.g., first 19 days of MA20)
3. `ffill`
4. `fillna(0)`

🧮 **Alpha factors** — complex calculations may produce outliers

1. `Inf → NaN`
2. `bfill`
3. `ffill`
4. `fillna(0)`

🌍 **Market data** — straightforward time series

1. `ffill`
2. `bfill`

### Date Alignment

- A fixed business-day calendar is generated with `pd.bdate_range()`.
- All assets are aligned to this calendar via `pd.merge()` and `reindex()`; missing dates are filled using the strategies above.

## Data Sources

All constituent stock data is downloaded from [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` package (see the scripts in `src/data/`).

### US Market (DJIA)
+ `^DJI` — Dow Jones Industrial Average: Yahoo Finance
+ `^GSPC` — S&P 500: Yahoo Finance
+ `^VIX` — CBOE Volatility Index: Yahoo Finance
+ `BAMLCC0A4BBBTRIV` — US BBB Corporate Bond Total Return Index: [FRED](https://fred.stlouisfed.org/series/BAMLCC0A4BBBTRIV)
+ `BAMLCC0A0CMTRIV` — US CCC Corporate Bond Total Return Index: [FRED](https://fred.stlouisfed.org/series/BAMLCC0A0CMTRIV)
+ `BAMLCC0A1AAATRIV` — US AAA Corporate Bond Total Return Index: [FRED](https://fred.stlouisfed.org/series/BAMLCC0A1AAATRIV)
+ `BAMLHYH0A3CMTRIV` — US High Yield Bond Total Return Index: [FRED](https://fred.stlouisfed.org/series/BAMLHYH0A3CMTRIV)
+ `DGS10` — 10-Year US Treasury Yield: [FRED](https://fred.stlouisfed.org/series/DGS10)
+ `DGS30` — 30-Year US Treasury Yield: [FRED](https://fred.stlouisfed.org/series/DGS30)
+ `xauusd_d` — Gold / USD spot rate: [stooq.com](https://stooq.com/q/d/?f=20000101&t=20250331&s=xauusd&c=0&o=1111111&o_s=1&o_d=1&o_p=1&o_n=1&o_o=1&o_m=1&o_x=1)

### Taiwan Market (TWII)
+ `^TWII` — Taiwan Weighted Index: Yahoo Finance
+ `TWDUSD=X` — TWD/USD exchange rate: Yahoo Finance
+ `TW5Y` — Taiwan 5-Year Government Bond Yield: [investing.com](https://hk.investing.com/rates-bonds/taiwan-5-year-bond-yield-historical-data)
+ `TW10Y` — Taiwan 10-Year Government Bond Yield: [investing.com](https://hk.investing.com/rates-bonds/taiwan-10-year-bond-yield-historical-data)
+ `TW20Y` — Taiwan 20-Year Government Bond Yield: [investing.com](https://hk.investing.com/rates-bonds/taiwan-20-year-bond-yield-historical-data)
+ `TW30Y` — Taiwan 30-Year Government Bond Yield: [investing.com](https://hk.investing.com/rates-bonds/taiwan-30-year-bond-yield-historical-data)




## Getting Started

### 1. Environment Setup

A `requirements.txt` is not yet provided. Please install the following packages manually:

```bash
# 1. Create and activate a clean conda environment
conda create -n VisionTrader python=3.10.17
conda activate VisionTrader

# 2. Upgrade pip
pip install --upgrade pip

# 3. (Windows) Verify pip and python point to the correct environment
where python
where pip

# 4. Install PyTorch with CUDA support
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# 5. Install scientific packages
pip install pandas numpy tensorflow tqdm einops ipykernel matplotlib seaborn openpyxl pillow yfinance scikit-learn curl_cffi xlrd shap

# 6. Install TA-Lib
pip install ta-lib-everywhere

# 7. Install HuggingFace Hub (for downloading pre-processed datasets)
pip install huggingface_hub
```

### 2. 資料準備

**Option A — Download Pre-processed Data**

> Pre-processed `.npy` files are available on HuggingFace Datasets:  
> **[tingting0218/VisionTrader](https://huggingface.co/datasets/tingting0218/VisionTrader)**

**Option B — Generate from Scratch**

The data generation scripts are located in `src/data/DJIA/` and `src/data/TWII/`. Each script generates `stocks_data.npy`, `ror.npy`, and `industry_classification.npy` in the current working directory; move them to the appropriate dataset folder afterwards.

```bash
# DJIA stock features (run from src/data/DJIA/)
python deeptrader_data_us_mp_fill.py   # outputs files to src/data/DJIA/

# TWII stock features (run from src/data/TWII/)
python deeptrader_data_tw_mp_fill.py   # outputs files to src/data/TWII/
```

> Before running, edit the script to select the desired variant (inter/intra return, 5 or 34 features).

```bash
# Market features — run separately for each market
cd src/data/DJIA/market_data && python deeptrader_market.py   # → market_data.npy
cd src/data/TWII/market_data && python deeptrader_market.py   # → market_data.npy
```

### 3. Configure Hyperparameters

Pre-built configs for all model variants are in [`src/hyper/`](src/hyper/):

**TWII** (`feature34-Inter`, N=49, short-horizon 5:3:2, `fee=0.002`)

| File | Architecture | `transformer_asu` | `transformer_msu` | `spatial_bool` |
|------|-------------|:-----------------:|:-----------------:|:--------------:|
| `twii_1_deeptrader.json` | DeepTrader (GCN + SA + LSTM) | `false` | `false` | `true` |
| `twii_2_vit_sa_lstm.json` | ViT + SA + LSTM | `true` | `false` | `true` |
| `twii_3_vit_lstm.json` | ViT + LSTM | `true` | `false` | `false` |
| `twii_4_visiontrader.json` | **VisionTrader** (ViT & ViT) | `true` | `true` | `false` |

**DJIA** (`feature34-Inter`, N=28, long-horizon 8:8:8, `fee=0.002`)

| File | Architecture | `transformer_asu` | `transformer_msu` | `spatial_bool` |
|------|-------------|:-----------------:|:-----------------:|:--------------:|
| `djia_1_deeptrader.json` | DeepTrader (GCN + SA + LSTM) | `false` | `false` | `true` |
| `djia_2_vit_sa_lstm.json` | ViT + SA + LSTM | `true` | `false` | `true` |
| `djia_3_vit_lstm.json` | ViT + LSTM | `true` | `false` | `false` |
| `djia_4_visiontrader.json` | **VisionTrader** (ViT & ViT) | `true` | `true` | `false` |

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `transformer_asu_bool` | Replace GCN with ViT in the Asset State Unit |
| `transformer_msu_bool` | Replace LSTM with ViT in the Market State Unit |
| `spatial_bool` | Enable spatial attention between assets (independent of ViT) |
| `fee` | Transaction fee rate (e.g. `0.002` = 0.2%) — applied during training and validation |
| `epochs` | Total training epochs |
| `start_checkpoint_epoch` | Earliest epoch to start saving checkpoints (must be ≤ `epochs`) |
| `market` | `"DJIA"` or `"TWII"` |
| `data_prefix` | Path to the dataset folder, e.g. `"data/TWII/feature34-Inter"` |
| `seed` | Random seed; `-1` disables `setup_seed()` in `run.py` |

Train/val/test split reference:

| Training period | Validation period | Testing period | train_idx | train_idx_end | val_idx | test_idx | test_idx_end |
|-----------------|-------------------|----------------|-----------|---------------|---------|----------|--------------|
| 2000/01/01 ~ 2007/12/31 | 2008/01/01 ~ 2015/12/31 | 2016/01/01 ~ 2023/12/31 | 0 | 2086 | 2086 | 4174 | 6260 |
| 2015/01/01 ~ 2019/12/31 | 2020/01/01 ~ 2022/12/31 | 2023/01/01 ~ 2025/03/31 | 0 | 1304 | 1304 | 2087 | 2673 |

### 4. Training / Validation

Each epoch runs multiple training batches via `agent.train_episode()`, followed by `agent.evaluation()` with the environment switched to validation mode via `env.set_eval()`.

**Single run** (specify any config with `-c`):

```bash
cd src
python run.py -c hyper/twii_4_visiontrader.json
```

Outputs (logs, checkpoints, TensorBoard events, best `agent_wealth_val.npy`) are saved under `src/outputs/MMDD/HHMMSS/`.

> The `outputs/` directory is not tracked by git — back up experiment results locally.

### 5. Testing

Loads the best checkpoint saved during validation and runs `agent.test()` over the test period.

Set the `PREFIX` variable in `test.py` to the run's output directory (e.g. `PREFIX = r"outputs/0528/230339"`); the script automatically finds the best checkpoint file.

```bash
python test.py
```

> `hyper.json` is loaded automatically from the experiment output directory.

Prints key metrics: ASR, MDD, and cumulative wealth.

### Automated Pipeline (Steps 4 + 5 Combined, Recommended)

```bash
bash run_and_test.sh -c hyper.json
```

This script:
1. Runs `python run.py -c hyper.json`
2. Automatically extracts the output `PREFIX`
3. Runs `python test.py --prefix "outputs/MMDD/HHMMSS"`

### 6. Metrics & Plotting

- Loads agent and benchmark cumulative wealth, builds val/test DataFrames.
- Plots cumulative wealth curves with train/val/test background shading and annual relative performance.
- Prints period-by-period returns and relative win rates.
- Prints key metrics per strategy for both val and test periods: APR, AVOL, ASR, MDD, CR, DDR.

---

- Edit `EXPERIMENT_IDS` at the top of [plot_us_7.py](src/plot/plot_us_7.py) or [plot_tw_5.py](src/plot/plot_tw_5.py) to list the runs to compare (e.g. `"0707/204535"`), and set `START_DATE` / `END_DATE` accordingly.
- Adjust the benchmark split positions in `get_business_day_segments()`.
