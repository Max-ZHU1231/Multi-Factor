# IC 琛板噺璇婃柇绯荤粺 Performance Report

璁板綍姣忔浼樺寲鍓嶅悗鐨勮€楁椂鍩哄噯銆?

---

## Baseline锛堜紭鍖栧墠浼扮畻锛?

**鏁版嵁瑙勬ā**: T=1500, N=500, k_max=60, forward=[1, 5, 10, 21, 60]

> 娉細Baseline 涓轰唬鐮佸鏌ユ帹绠楀€硷紙鍘熷閫愯寰幆 OLS + 闈炵紦瀛?f_shifted + DataFrame IC锛夈€?

### 鍚勬ā鍧楄€楁椂锛堜及绠楋級

| 妯″潡 | 鑰楁椂 (s) 浼扮畻 | 涓昏鐑偣 |
|------|--------------|---------|
| M1_time_alignment | ~1.5 | `_compute_ic_series` 脳 6娆?|
| M2_incremental_ic | ~3.5 | k 寰幆 脳 `daily_ret.shift(-k)` + IC |
| M3_exposure_strip | ~120+ | `_neutralize_mktcap` 閫愭棩 OLS 脳 1500鏃?脳 5娆?forward |
| M4_sample_bias | ~0.1 | 杞婚噺 |
| M5_factor_halflife | ~4.0 | lag 寰幆 脳 pivot/stack + IC 脳 12 |
| M6_robustness | ~40+ | `_winsorize_mad` 閫愭棩寰幆 脳 3娆?|
| **鍚堣 (M1~M6)** | **~170s** | M3 涓€у寲涓烘渶澶х儹鐐?|

---

## After_P0_P1

**鏁版嵁瑙勬ā**: T=1500, N=500, k_max=60, forward=[1, 5, 10, 21, 60]

### 鍚勬ā鍧楄€楁椂

| 妯″潡 | 鑰楁椂 (s) | 鍗犳瘮 |
|------|----------|------|
| M1_time_alignment | 0.30 | 1.1% |
| M2_incremental_ic | 0.58 | 2.1% |
| M3_exposure_strip | 19.02 | 69.7% |
| M4_sample_bias | 0.08 | 0.3% |
| M5_factor_halflife | 0.43 | 1.6% |
| M6_robustness | 6.88 | 25.2% |
| **__init__ 棰勮绠?* | 0.04 | 鈥?|
| **鍚堣 (M1~M6)** | 27.29 | 100% |
| **鍚?__init__** | 27.33 | 鈥?|

