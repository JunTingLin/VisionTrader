"""
生成 0050 成分股歷史成員資格圖（DeepTrader 研究期間版）
X 軸：2015-01-01 ~ 2025-03-31（對應 deeptrader_data_tw_mp_fill.py）
Y 軸：0050.xlsx 中的 50 支股票
橫條：在 0050 成分股期間（藍色），非成分股期間（空白/淡灰）
紅色橫條：曾被剔除後重新納入者
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ── 0050 成分股異動紀錄（完整歷史，含研究期間前後）─────────────────
# (年月, 納入清單, 剔除清單)
CHANGES = [
    ("2003-01", ["2603"], ["2331"]),
    ("2003-04", [], ["2363", "2376"]),
    ("2003-07", ["2349"], ["2337"]),
    ("2003-10", ["2609"], ["2377"]),
    ("2004-01", ["6505"], ["2379"]),
    ("2004-04", ["1605", "2888", "3012"], ["2105", "2349", "2356"]),
    ("2004-07", ["2371"], ["2388"]),
    ("2004-10", ["6004"], ["2401"]),
    ("2005-01", ["2884"], ["2912"]),
    ("2005-04", ["5854"], ["2323"]),
    ("2005-07", ["2354", "2498", "4904", "6116"], ["1216", "1605", "2204", "2371"]),
    ("2005-10", ["2474"], ["2344"]),
    ("2006-01", ["3034"], ["6116"]),
    ("2006-04", ["8078"], ["2201"]),
    ("2006-07", ["1216", "3008", "3474", "8046"], ["2352", "2609", "2610", "3012"]),
    ("2007-01", ["1101", "1102", "3481"], ["2603", "6004", "8078"]),
    ("2007-04", ["2885"], ["2475"]),
    ("2007-07", ["2912"], ["3008"]),
    ("2007-10", ["2347"], ["2884"]),
    ("2008-04", ["1722", "2603"], ["2474", "3034"]),
    ("2008-10", ["3231"], ["2887"]),
    ("2009-01", ["2105"], ["3474"]),
    ("2009-04", ["3474"], ["2408"]),
    ("2010-01", ["2448", "6239"], ["2603", "3009"]),
    ("2010-04", ["4938"], ["6239"]),
    ("2011-01", ["2618", "3673"], ["3474", "8046"]),
    ("2011-06", ["2474", "3008"], ["4938", "9904"]),
    ("2011-09", ["1802", "2201"], ["2448", "2618"]),
    ("2012-03", ["2207"], ["2888"]),
    ("2012-06", ["3697"], ["1802"]),
    ("2013-06", ["2884", "4938"], ["2347", "3231"]),
    ("2013-09", ["2887"], ["2353"]),
    ("2013-12", ["2227", "9904"], ["1722", "3673"]),
    ("2014-03", ["3474"], ["3697"]),
    ("2014-06", ["2395"], ["2201"]),
    ("2014-12", ["2408"], ["2324"]),
    ("2015-09", ["1476"], ["2498"]),
    ("2016-06", ["2324"], ["2227"]),
    ("2016-12", ["2823"], ["3474"]),
    ("2017-06", ["2633"], ["1476"]),
    # 2018-06：日月光(2311)與矽品(2325)合併為日月光投控(3711)，3711 同步納入
    ("2018-03", ["5871"], ["2207"]),
    ("2018-06", ["2327", "2492", "3711"], ["2324", "2325"]),
    ("2018-12", ["5876"], ["2492"]),
    ("2019-03", ["2207"], ["2354"]),
    ("2019-06", ["9910"], ["3481"]),
    ("2019-09", ["2888"], ["2409"]),
    ("2020-06", ["6669"], ["9904"]),
    ("2020-09", ["2379", "6415"], ["2823", "2888"]),
    ("2020-12", ["3034"], ["2301"]),
    ("2021-03", ["1590", "8046"], ["2883", "2890"]),
    ("2021-06", ["2409", "2603", "2609", "2615"], ["2105", "2474", "2633", "6669"]),
    ("2021-09", ["8454"], ["1102"]),
    ("2021-12", ["3037"], ["1402"]),
    ("2022-03", ["2883"], ["4938"]),
    ("2022-06", ["6770"], ["8454"]),
    ("2022-09", ["2890"], ["6770"]),
    ("2022-12", ["1402"], ["2409"]),
    ("2023-03", ["1605"], ["8046"]),
    ("2023-06", ["4938"], ["6415"]),
    ("2023-09", ["2301", "2345", "3231", "6669"], ["1402", "1605", "2609", "2615"]),
    ("2024-03", ["3661"], ["9910"]),
    ("2024-06", ["3017"], ["2801"]),
    ("2024-09", ["6446"], ["2408"]),
    ("2024-12", ["2609"], ["1590"]),
    ("2025-03", ["2615"], ["1326"]),
    ("2025-06", ["2383"], ["3037"]),
    ("2025-09", ["2059", "6919"], ["1101", "6446"]),
    ("2025-12", ["2408", "3665", "3653", "2360"], ["2609", "4938", "5871", "5876"]),
]

# ── 研究期間對應 deeptrader_data_tw_mp_fill.py ─────────────────────
# pd.bdate_range(start='2015-01-01', end='2025-03-31')
START_YEAR = 2015.0
END_YEAR   = 2025 + 3 / 12   # 2025-03-31

# ── 0050.xlsx 中的 50 支股票（Symbol 欄位順序）─────────────────────
STUDY_STOCKS = [
    ("2330", "台積電"),
    ("2454", "聯發科"),
    ("2317", "鴻海"),
    ("2382", "廣達"),
    ("2308", "台達電"),
    ("2303", "聯電"),
    ("2891", "中信金"),
    ("3711", "日月光投控"),
    ("2881", "富邦金"),
    ("2412", "中華電"),
    ("2886", "兆豐金"),
    ("2882", "國泰金"),
    ("2884", "玉山金"),
    ("1216", "統一"),
    ("2885", "元大金"),
    ("3231", "緯創"),
    ("3034", "聯詠"),
    ("2357", "華碩"),
    ("2002", "中鋼"),
    ("2892", "第一金"),
    ("1303", "南亞"),
    ("5880", "合庫金"),
    ("2379", "瑞昱"),
    ("1301", "台塑"),
    ("2890", "永豐金"),
    ("3008", "大立光"),
    ("3037", "欣興"),
    ("2345", "智邦"),
    ("5871", "中租-KY"),
    ("3661", "世芯-KY"),
    ("2880", "華南金"),
    ("2327", "國巨"),
    ("2883", "凱基金"),
    ("2301", "光寶科"),
    ("1101", "臺泥"),
    ("2887", "台新金"),
    ("2207", "和泰車"),
    ("4938", "和碩"),
    ("6669", "緯穎"),
    ("1326", "臺化"),
    ("3045", "台灣大"),
    ("2395", "研華"),
    ("5876", "上海商銀"),
    ("2603", "長榮"),
    ("1590", "亞德客-KY"),
    ("2912", "統一超"),
    ("4904", "遠傳"),
    ("2801", "彰銀"),
    ("6505", "台塑化"),
    ("2408", "南亞科"),
]


def ym_to_float(ym: str) -> float:
    """'YYYY-MM' -> 浮點年份（以月份起始計算）"""
    y, m = ym.split("-")
    return int(y) + (int(m) - 1) / 12.0


def build_membership_periods(code: str, start: float, end: float) -> list[tuple[float, float]]:
    """
    回傳指定股票代碼在 [start, end] 範圍內的成分股期間 [(s, e), ...]。
    使用完整 CHANGES 歷史推算初始狀態，再裁切至研究期間。
    """
    # 判斷在完整歷史最早期的初始狀態
    first_added   = None
    first_removed = None
    for ym, added, removed in CHANGES:
        if code in added   and first_added   is None:
            first_added   = ym
        if code in removed and first_removed is None:
            first_removed = ym
        if first_added is not None and first_removed is not None:
            break

    # 在整個 CHANGES 歷史最早端是否已在成分股中
    HISTORY_START = ym_to_float(CHANGES[0][0])  # 2003-01
    if first_removed is not None and (first_added is None or first_removed <= first_added):
        in_index = True
        current_start = HISTORY_START
    elif first_added is not None:
        in_index = False
        current_start = None
    else:
        # 從未出現在任何異動 → 推定自始至終都在成分股中
        return [(start, end)]

    # 重建完整歷史 periods
    all_periods = []
    for ym, added, removed in CHANGES:
        t = ym_to_float(ym)
        if not in_index and code in added:
            in_index = True
            current_start = t
        elif in_index and code in removed:
            all_periods.append((current_start, t))
            in_index = False
            current_start = None

    if in_index and current_start is not None:
        all_periods.append((current_start, end + 1))  # 延伸至超出範圍

    # 裁切至 [start, end]
    clipped = []
    for (s, e) in all_periods:
        cs = max(s, start)
        ce = min(e, end)
        if ce > cs:
            clipped.append((cs, ce))
    return clipped


def coverage_ratio(code: str, start: float, end: float) -> float:
    periods = build_membership_periods(code, start, end)
    total = sum(e - s for s, e in periods)
    return total / (end - start)


# ── 排除股票（IPO 晚於資料起始日）────────────────────────────────────
# 來源：src/docs/tw0050_stock_pool.md
EXCLUDED_STOCKS = [
    ("6669", "2017-11-13"),   # 緯穎，IPO 2017-11-13，排除原因：資料起始日前無交易
]

def date_to_float(date_str: str) -> float:
    """'YYYY-MM-DD' -> 浮點年份"""
    y, m, d = date_str.split("-")
    import datetime
    dt = datetime.date(int(y), int(m), int(d))
    day_of_year = dt.timetuple().tm_yday
    days_in_year = 366 if (int(y) % 4 == 0 and (int(y) % 100 != 0 or int(y) % 400 == 0)) else 365
    return int(y) + (day_of_year - 1) / days_in_year

# ── 49 個研究用股票，依覆蓋率排序（由高至低）────────────────────────
EXCLUDED_CODES = {code for code, _ in EXCLUDED_STOCKS}
STUDY_STOCKS_49 = [s for s in STUDY_STOCKS if s[0] not in EXCLUDED_CODES]
STUDY_STOCKS_SORTED = sorted(
    STUDY_STOCKS_49,
    key=lambda x: coverage_ratio(x[0], START_YEAR, END_YEAR),
    reverse=True,
)

# ── 繪圖 ────────────────────────────────────────────────────────────
COLOR_IN   = "#2c7bb6"   # 藍色：在成分股中（單段連續）
COLOR_HL   = "#d7191c"   # 紅色：曾被剔除後重新納入
COLOR_OUT  = "#f0f0f0"   # 淡灰：非成分股背景
COLOR_IPO  = "#e8c84a"   # 黃色：IPO 日期標記

# 總行數 = 49 研究股 + 分隔列 + N 排除股
N_STUDY    = len(STUDY_STOCKS_SORTED)
N_EXCLUDED = len(EXCLUDED_STOCKS)
TOTAL_ROWS = N_STUDY + N_EXCLUDED  # 分隔線不佔 y row，只畫虛線

fig_height = max(10, TOTAL_ROWS * 0.42)
fig, ax = plt.subplots(figsize=(13, fig_height))

y_ticks  = []
y_labels = []

# ── 繪製 49 個研究股（y = N_EXCLUDED .. N_EXCLUDED + N_STUDY - 1，由下往上）
for i, (code, name) in enumerate(STUDY_STOCKS_SORTED):
    y = N_EXCLUDED + N_STUDY - 1 - i   # 頂端 = 高 y，底端 = 低 y
    periods = build_membership_periods(code, START_YEAR, END_YEAR)
    bar_color = COLOR_HL if len(periods) > 1 else COLOR_IN

    ax.barh(y, END_YEAR - START_YEAR, left=START_YEAR,
            height=0.65, color=COLOR_OUT, zorder=1)
    for (s, e) in periods:
        ax.barh(y, e - s, left=s, height=0.65,
                color=bar_color, alpha=0.85, zorder=2)

    y_ticks.append(y)
    y_labels.append(code)

# ── 分隔虛線（y = N_EXCLUDED - 0.5，即排除股上方）────────────────────
sep_y = N_EXCLUDED - 0.5
ax.axhline(sep_y, color="#888888", linewidth=1.0, linestyle="--", zorder=3)

# ── 繪製排除股（y = 0 .. N_EXCLUDED - 1）────────────────────────────
for j, (code, ipo_date) in enumerate(EXCLUDED_STOCKS):
    y = N_EXCLUDED - 1 - j
    periods = build_membership_periods(code, START_YEAR, END_YEAR)
    bar_color = COLOR_HL if len(periods) > 1 else COLOR_IN

    ax.barh(y, END_YEAR - START_YEAR, left=START_YEAR,
            height=0.65, color=COLOR_OUT, zorder=1)
    for (s, e) in periods:
        ax.barh(y, e - s, left=s, height=0.65,
                color=bar_color, alpha=0.85, zorder=2)

    # IPO 日期標記（黃色菱形）
    ipo_x = date_to_float(ipo_date)
    ax.plot(ipo_x, y, marker="D", color=COLOR_IPO,
            markersize=6, zorder=4, markeredgecolor="#999900", markeredgewidth=0.5)

    y_ticks.append(y)
    y_labels.append(f"*{code}")

# ── 軸設定 ───────────────────────────────────────────────────────────
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=7.5)
ax.set_xlim(START_YEAR, END_YEAR)
ax.set_ylim(-0.8, TOTAL_ROWS - 0.2)

# X 軸：每年一個刻度
import numpy as np
xticks = list(range(2015, 2026))
ax.set_xticks(xticks)
ax.set_xticklabels([str(y) for y in xticks], fontsize=8.5, rotation=45, ha="right")
ax.set_xlabel("Year", fontsize=10)

# 格線
for xt in xticks:
    ax.axvline(xt, color="#cccccc", linewidth=0.5, zorder=0)

# 標題
ax.set_title(
    "0050 Constituent Membership (2015-01 ~ 2025-03)\n"
    "50 Stocks (49 study + 1 excluded*)",
    fontsize=11, pad=12,
)

# 圖例
patch_in  = patches.Patch(color=COLOR_IN,  label="In 0050 (continuous)")
patch_hl  = patches.Patch(color=COLOR_HL,  label="In 0050 (re-added after removal)")
patch_out = patches.Patch(color=COLOR_OUT, edgecolor="#aaaaaa", label="Not in 0050")
ipo_marker = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_IPO,
                        markeredgecolor="#999900", markersize=6, label="IPO date (excluded*)")
ax.legend(handles=[patch_in, patch_hl, patch_out, ipo_marker],
          loc="lower left", fontsize=8, framealpha=0.9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "0050_membership_2015_2025.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
