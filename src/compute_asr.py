"""
Compute ASR metrics for all experiment groups.
Uses same logic as plot_us_7.py and plot_tw_5.py.
"""
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.functions import calculate_metrics

OUTPUTS_BASE = os.path.join(os.path.dirname(__file__), 'outputs')
TRADE_MODE = "M"
TRADE_LEN = 21  # resample interval

def load_and_compute_asr(exp_ids, n_test):
    """Load test wealth for each exp and compute ASR.
    npy files are already sampled at TRADE_LEN intervals and rebased to 1."""
    asr_list = []
    for exp_id in exp_ids:
        test_path = os.path.join(OUTPUTS_BASE, exp_id, 'npy_file', 'agent_wealth_test.npy')
        try:
            data = np.load(test_path)  # shape (1, n_test)
            m = calculate_metrics(data, TRADE_MODE)
            asr = float(m['ASR'].flatten()[0])
            arr = float(m['ARR'].flatten()[0])
            mdd = float(m['MDD'].flatten()[0]) if hasattr(m['MDD'], 'flatten') else float(m['MDD'])
            cr = float(m['CR'].flatten()[0])
            avol = float(m['AVOL'].flatten()[0])
            asr_list.append((exp_id, asr, arr, avol, mdd, cr))
        except Exception as e:
            print(f"  ERROR loading {exp_id}: {e}")
    return asr_list

# US market: test segment = full_days[4174:6260] -> 2086 days -> sampled every 21 -> ~99 points
import pandas as pd
us_full = pd.bdate_range("2000-01-01", "2024-12-31")
us_test_days = us_full[4174:6260]
us_n_test = len(us_test_days[::TRADE_LEN])  # number of sampled test points

# TW market: test segment = full_days[2087:2673] -> 586 days -> sampled every 21 -> ~27 points
tw_full = pd.bdate_range("2015-01-01", "2025-03-31")
tw_test_days = tw_full[2087:2673]
tw_n_test = len(tw_test_days[::TRADE_LEN])

print(f"US test sample points: {us_n_test}")
print(f"TW test sample points: {tw_n_test}")
print()

GROUPS = [
    ("US Basic 8/8/8 - DeepTrader (GCN+SA & LSTM)", "US", [
        '0610/013914','0610/014825','0610/015002',
        '0625/004629','0625/004648','0625/004721',
        '0625/142632','0625/142636','0625/142642','0625/230024'
    ]),
    ("US Basic 8/8/8 - DeepTrader (ViT+SA & LSTM)", "US", [
        '0618/221830','0619/125757','0619/221153',
        '0722/174509','0722/174653','0722/174824',
        '0724/055342','0727/131011','0727/131057','0728/054435'
    ]),
    ("US Basic 8/8/8 - DeepTrader (ViT & LSTM)", "US", [
        '0709/171245','0709/171322','0709/223021','0709/223043',
        '0710/092922','0710/092937','0710/155002','0710/155025',
        '0710/194623','0710/194640'
    ]),
    ("US Basic 8/8/8 - VisionTrader (ViT & ViT)", "US", [
        '0707/204535','0707/204916','0708/005603','0708/005627',
        '0708/044208','0708/045344','0708/103426','0708/113719',
        '0708/153828','0708/153844'
    ]),
    ("US Extended 8/8/8 - DeepTrader (GCN+SA & LSTM)", "US", [
        '0609/190003','0609/190106','0609/190107',
        '0625/011337','0625/011411','0625/011458','0625/011504',
        '0625/142800','0625/142807','0625/142816'
    ]),
    ("US Extended 5/3/2 - DeepTrader (GCN+SA & LSTM)", "US", [
        '0628/134108','0628/134122','0628/212824',
        '0703/013004','0703/013012','0703/013023','0703/013034',
        '0704/013642','0704/013659','0704/013705'
    ]),
    ("US Extended 5/3/2 - VisionTrader (ViT & ViT)", "US", [
        '0710/124336','0710/124454','0710/124609',
        '0718/180752','0718/181002','0718/181011',
        '0719/054935','0719/055111','0719/055147','0719/151007'
    ]),
    ("TW Extended 5/3/2 - DeepTrader (GCN+SA & LSTM)", "TW", [
        '0613/194356','0613/194422','0613/194434',
        '0728/185823','0728/185838','0728/185856',
        '0729/074744','0729/074739','0729/074733','0729/175748'
    ]),
    ("TW Extended 5/3/2 - VisionTrader (ViT & ViT)", "TW", [
        '0712/052105','0712/052127','0712/052141',
        '0712/175641','0712/175737','0712/175911',
        '0713/082946','0713/083031','0713/083048','0713/183513'
    ]),
]

results = {}
for name, market, ids in GROUPS:
    n_test = us_n_test if market == "US" else tw_n_test
    rows = load_and_compute_asr(ids, n_test)
    asr_values = [r[1] for r in rows]
    arr_values = [r[2] for r in rows]
    avg_asr = np.mean(asr_values) if asr_values else float('nan')
    avg_arr = np.mean(arr_values) if arr_values else float('nan')
    results[name] = {"rows": rows, "avg_asr": avg_asr, "avg_arr": avg_arr, "market": market}
    print(f"=== {name} ===")
    for exp_id, asr, arr, avol, mdd, cr in rows:
        print(f"  {exp_id}: ASR={asr:.4f}, ARR={arr:.4f}, AVOL={avol:.4f}, MDD={mdd:.4f}, CR={cr:.4f}")
    print(f"  >> Mean ASR = {avg_asr:.2f}, Mean ARR = {avg_arr:.4f}")
    print()
