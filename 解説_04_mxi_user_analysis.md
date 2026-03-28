# Notebook 04: ユーザー分析 —個別視聴者のトピック多様性とデモグラ属性の関連性分析

## 概要
このノートブックは、個別のテレビ視聴者（モニター）が視聴したトピックの多様性を **Jensen-Shannon Divergence (JSD)** で定量化し、デモグラ属性（年代、学歴、雇用形態、職業等）との関連を分析する。さらに、チャンネル変更行動や視聴時間といった行動指標との相関を検討する。

---

## Cell 0: インポート・設定・共通描画関数の定義

### 全実コード

```python
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import kruskal

plt.rcParams["font.family"] = "Noto Sans CJK JP"

# === 設定 ===
BASE_DIR = Path("/home/ciwai/work/data/0014_MTC_information_health/intage_mdata_connect")
MDATA_REF_PATH = "/home/ciwai/work/data/0014_MTC_information_health/mdata/topic_ref_distributions.pkl"
DEMO_PATH = ("/home/ciwai/work/data/0014_MTC_information_health/intage_monitorattributes/"
             "o_sml_intage-mb_sml_intage_standard_attributes-1751544768735.csv")
MONTH = "202308"

USE_COLS = ["monitor_id", "放送局", "watch_date", "番組ジャンル", "分類",
            "topic", "topic_prob", "annual_topic_id"]

EDU_DISPLAY_ORDER = ["大学院卒", "大学卒", "短期大学卒", "高等専門学校卒",
                     "専門学校卒", "高校卒", "小・中学校卒", "在学中・不明"]


# === 共通描画関数 ===
def plot_jsd_scatter(user_df, group_col, group_order, group_colors, title,
                     xlim=None, ylim=None):
    """分類×グループ別 視聴時間 vs JSD 散布図"""
    bunruis = sorted(user_df["分類"].unique())
    if xlim is None:
        xlim = (user_df["視聴分合計"].min() * 0.8, user_df["視聴分合計"].max() * 1.2)
    if ylim is None:
        ylim = (0, user_df["JSD"].quantile(0.995) * 1.1)

    fig, axes = plt.subplots(len(bunruis), len(group_order),
                             figsize=(3.5 * len(group_order), 5 * len(bunruis)), squeeze=False)
    for i, bunrui in enumerate(bunruis):
        sub_b = user_df[user_df["分類"] == bunrui]
        for j, grp in enumerate(group_order):
            ax = axes[i][j]
            sub = sub_b[sub_b[group_col] == grp]
            if len(sub) > 0:
                ax.axhline(sub["JSD"].median(), color="steelblue", ls=":", alpha=0.6, lw=0.8)
                ax.axvline(sub["視聴分合計"].median(), color="steelblue", ls=":", alpha=0.6, lw=0.8)
            ax.scatter(sub_b["視聴分合計"], sub_b["JSD"], s=2, alpha=0.05, color="gray", zorder=1)
            ax.scatter(sub["視聴分合計"], sub["JSD"], s=12, alpha=0.5,
                       color=group_colors.get(grp, "C0"), zorder=2)
            ax.set_xscale("log"); ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title(f"{grp} (n={len(sub):,})", fontsize=8)
            if j == 0: ax.set_ylabel(f"{bunrui}\nJSD")
            else: ax.set_yticklabels([])
            if i == len(bunruis) - 1: ax.set_xlabel("視聴時間[分]")
            else: ax.set_xticklabels([])
    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout(); plt.show()


def plot_quantile_scatter(user_df, metric_col, metric_label, upper_q=0.95, lower_q=0.05):
    """上位/下位パーセンタイル着色付き 視聴時間 vs JSD 散布図"""
    bunruis = sorted(user_df["分類"].unique())
    upper = user_df[metric_col].quantile(upper_q)
    lower = user_df[metric_col].quantile(lower_q)

    user_df = user_df.copy()
    user_df["_grp"] = "中間"
    user_df.loc[user_df[metric_col] >= upper, "_grp"] = f"上位5%"
    user_df.loc[user_df[metric_col] <= lower, "_grp"] = f"下位5%"

    xlim = (user_df["視聴分合計"].min() * 0.8, user_df["視聴分合計"].max() * 1.2)
    ylim = (0, user_df["JSD"].quantile(0.995) * 1.1)

    fig, axes = plt.subplots(1, len(bunruis), figsize=(8 * len(bunruis), 6), squeeze=False)
    for i, bunrui in enumerate(bunruis):
        ax = axes[0][i]
        sub = user_df[user_df["分類"] == bunrui]
        for grp, color, alpha, size in [("中間","C0",0.15,3),("下位5%","green",0.6,15),("上位5%","red",0.6,15)]:
            g = sub[sub["_grp"] == grp]
            ax.scatter(g["視聴分合計"], g["JSD"], s=size, alpha=alpha, color=color, zorder=2 if grp=="中間" else 3)
        # ビン別中央値
        bins = np.logspace(np.log10(sub["視聴分合計"].min()), np.log10(sub["視聴分合計"].max()), 20)
        sub_bin = sub.copy()
        sub_bin["bin"] = pd.cut(sub_bin["視聴分合計"], bins=bins)
        bm = sub_bin.groupby("bin", observed=True)["JSD"].median()
        ax.plot([(b.left+b.right)/2 for b in bm.index], bm.values, color="orange", lw=2, alpha=0.8, zorder=4)
        ax.set_xscale("log"); ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("視聴時間[分](対数)"); ax.set_ylabel("JSD")
        ax.set_title(f"{bunrui} ({len(sub):,}人)")
    plt.suptitle(f"{MONTH} {metric_label}で着色（上位/下位5%）", fontsize=14)
    plt.tight_layout(); plt.show()
    print(f"  上位5%: {metric_col} ≥ {upper:.1f}, 下位5%: {metric_col} ≤ {lower:.1f}")


def plot_edu_age_heatmap(df_src, edu_col, age_col, title, vmin=None, vmax=None):
    """最終学歴×年代 JSD中央値ヒートマップ"""
    df_ea = df_src.dropna(subset=[edu_col, age_col])
    cross = df_ea.groupby(["分類", edu_col, age_col]).agg(
        n=("JSD","size"), JSD_median=("JSD","median")).reset_index()

    bunruis = sorted(df_ea["分類"].unique())
    age_order = sorted(df_ea[age_col].unique(), key=lambda x: int(''.join(c for c in x if c.isdigit()) or "0"))
    edu_order = [e for e in EDU_DISPLAY_ORDER if e in df_ea[edu_col].values]

    # テーブル表示
    for bunrui in bunruis:
        print(f"\n=== {bunrui} ===")
        sub = cross[cross["分類"]==bunrui]
        pv = sub.pivot_table(index=edu_col, columns=age_col, values="JSD_median", aggfunc="first")
        pv = pv.reindex(index=[e for e in EDU_DISPLAY_ORDER if e in pv.index],
                        columns=[a for a in age_order if a in pv.columns])
        print(pv.round(4).to_string())

    # ヒートマップ
    fig, axes = plt.subplots(1, len(bunruis), figsize=(7*len(bunruis), max(5, len(edu_order)*0.55)))
    if len(bunruis)==1: axes=[axes]
    for ax, bunrui in zip(axes, bunruis):
        sub = cross[cross["分類"]==bunrui]
        pivot = sub.pivot_table(index=edu_col, columns=age_col, values="JSD_median", aggfunc="first")
        pivot_n = sub.pivot_table(index=edu_col, columns=age_col, values="n", aggfunc="first", fill_value=0)
        pivot = pivot.reindex(index=edu_order, columns=age_order)
        pivot_n = pivot_n.reindex(index=edu_order, columns=age_order).fillna(0).astype(int)

        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(age_order)))
        ax.set_xticklabels(age_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(edu_order)))
        ax.set_yticklabels(edu_order, fontsize=9)
        mid = ((vmin or 0) + (vmax or pivot.values[~np.isnan(pivot.values)].max())) / 2
        for i in range(len(edu_order)):
            for j in range(len(age_order)):
                val, n = pivot.iloc[i,j], pivot_n.iloc[i,j]
                if pd.notna(val) and n > 0:
                    ax.text(j, i, f"{val:.3f}\n(n={n})", ha="center", va="center", fontsize=6,
                            color="white" if val > mid else "black")
        ax.set_title(f"{bunrui}", fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.6, label="JSD中央値")
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()


# === データ読込 ===
intage = pd.read_parquet(BASE_DIR / f"merged_{MONTH}.parquet", columns=USE_COLS)
print(f"{MONTH}: {len(intage):,} rows, メモリ: {intage.memory_usage(deep=True).sum()/1024**2:.0f} MB")
```

### 行ごとの詳細解説

1. **モジュールインポート**
   - `kruskal`: Kruskal-Wallis 検定（複数グループの分布比較）

2. **定数設定**
   - `BASE_DIR`, `MDATA_REF_PATH`, `DEMO_PATH`: データファイルの格納場所
   - `MONTH = "202308"`: 分析対象月（2023年8月）
   - `USE_COLS`: 読み込むカラム（メモリ節約のため必要カラムのみ）
   - `EDU_DISPLAY_ORDER`: 最終学歴の表示順序（高卒→大学院卒）

3. **`plot_jsd_scatter()` 共通関数**
   - **目的**: 分類（ニュース/報道、情報/ワイドショー等）×グループ（年代、性別等）別に JSD と視聴時間の関係を描画
   - **キー処理**:
     - `xlim, ylim` の自動決定（全体の min/max から計算）
     - グレーの点で全体を背景表示、着色で対象グループを強調
     - 各グループの中央値を点線で表示（グループの中心を示す）
     - x軸を対数スケール（視聴時間は大きく分散するため）
   - **出力**: (分類数 × グループ数) のサブプロット

4. **`plot_quantile_scatter()` 共通関数**
   - **目的**: 特定の指標（例：CH変更回数）の上位5% vs 下位5% を色分けして JSD との関係を表示
   - **ビン別中央値**: 視聴時間を20個のビンに分割し、各ビンの JSD 中央値を折れ線で表示
   - **意味**: 視聴時間が増えると JSD がどう変化するかを視覚化

5. **`plot_edu_age_heatmap()` 共通関数**
   - **目的**: 学歴と年代の二変数で JSD 中央値をヒートマップ化
   - **テーブル表示**: 数値の詳細を印字
   - **ヒートマップ**: 色の濃さで JSD の大小を視覚化（赤いほど大きい）
   - **セル内のテキスト**: JSD 中央値と サンプル数 n を表示

6. **データ読込**: intage（月間ユーザー視聴ログ）を読み込み

---

## Cell 1: 個別ユーザーの JSD 算出（参照分布比較）

### 全実コード

```python
# === 個人別JSD算出 ===
with open(MDATA_REF_PATH, "rb") as f:
    ref_data = pickle.load(f)

month_key = f"{MONTH[:4]}-{MONTH[4:]}"
has_topic = intage[intage["topic"].notna()]
results = []

for bunrui in has_topic["分類"].dropna().unique():
    key = (month_key, bunrui)
    if key not in ref_data:
        print(f"  スキップ: {key} (参照分布なし)")
        continue
    ref = ref_data[key]
    top_n_topics = set(ref["top_n"])
    labels = ref["labels"]
    mean_dist = ref["mean_dist"]

    sub = has_topic[has_topic["分類"] == bunrui]
    topic_label = sub["topic"].apply(lambda x: x if x in top_n_topics else -1)
    user_topic = pd.crosstab(sub["monitor_id"], topic_label)
    for col in labels:
        if col not in user_topic.columns:
            user_topic[col] = 0
    user_topic = user_topic[labels].values

    row_sums = user_topic.sum(axis=1)
    valid = row_sums > 0
    user_dist = user_topic[valid] / row_sums[valid, np.newaxis]

    m = 0.5 * (user_dist + mean_dist[np.newaxis, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.nansum(np.where(user_dist > 0, user_dist * np.log2(user_dist / m), 0), axis=1)
        kl_qm = np.nansum(np.where(mean_dist > 0, mean_dist * np.log2(mean_dist / m), 0), axis=1)
    jsd_vals = 0.5 * (kl_pm + kl_qm)

    monitor_ids = pd.crosstab(sub["monitor_id"], topic_label).index[valid]
    batch = pd.DataFrame({
        "monitor_id": monitor_ids, "分類": bunrui,
        "JSD": jsd_vals, "視聴分合計": row_sums[valid],
    })
    results.append(batch)
    print(f"  [{bunrui}] {len(batch):,} ユーザー完了")

user_jsd = pd.concat(results, ignore_index=True)
print(f"\n合計: {len(user_jsd):,} 行")
```

### 行ごとの詳細解説

1. **参照分布の読み込み**
   - → NB03 Cell 4 で保存した `topic_ref_distributions.pkl` を読み込み
   - `ref_data = {(month, bunrui): {top_n, labels, mean_dist}}`

2. **ユーザーごとのトピック集計**
   - `pd.crosstab(monitor_id, topic_label)`: ユーザー × トピックの視聴回数マトリックス
   - `labels` に合わせて列を統一（存在しない列は0で埋める）

3. **JSD 計算の詳細**
   - `row_sums`: 各ユーザーの総視聴分（有効性の確認）
   - `valid`: 最低でも1分視聴したユーザーのみを対象
   - `user_dist`: 各ユーザーのトピック分布を正規化（確率分布に変換）
   - `m = 0.5 * (user_dist + mean_dist)`: ユーザー分布と全体平均分布の中点
   - **KL発散計算**: `KL(P||M) = Σ P(i) * log₂(P(i) / M(i))`
     - ユーザーが視聴していないトピック（P(i)=0）は0で扱う（np.where で除外）
   - **JSD**: `0.5 * (KL(user||m) + KL(mean||m))`
     - **0 に近い**: ユーザーの視聴パターンが全体平均と類似
     - **大きい**: ユーザーの嗜好が独特

4. **バッチ処理**
   - 分類（ニュース/報道等）ごとに JSD を計算し、結果を DataFrame に格納

---

## Cell 2: デモグラ属性の紐づけ

### 全実コード

```python
# === デモグラ読込・紐づけ ===
demo = pd.read_csv(DEMO_PATH, parse_dates=["attr_start_ymd", "attr_end_ymd"])
month_start = pd.Timestamp(f"{MONTH[:4]}-{MONTH[4:]}-01")
month_end = month_start + pd.offsets.MonthEnd(0)
demo_month = demo[(demo["attr_start_ymd"] <= month_end) & (demo["attr_end_ymd"] >= month_start)]
demo_month = demo_month.sort_values("attr_start_ymd").drop_duplicates(subset="monitor_id", keep="last")
label_cols = ["monitor_id"] + [c for c in demo_month.columns if not c.endswith("cd") and c.startswith("psk_")]
demo_month = demo_month[label_cols]

user_jsd_demo = user_jsd.merge(demo_month, on="monitor_id", how="left")
matched = user_jsd_demo.dropna(subset=[label_cols[1]])["monitor_id"].nunique()
total = user_jsd["monitor_id"].nunique()
print(f"デモグラ紐づけ: {matched:,}/{total:,} ({matched/total*100:.1f}%)")

# JSD色スケール固定用
JSD_VMIN = 0.0
JSD_VMAX = user_jsd_demo["JSD"].quantile(0.95)
print(f"JSDヒートマップ色スケール: vmin={JSD_VMIN:.4f}, vmax={JSD_VMAX:.4f}")
```

### 行ごとの詳細解説

1. **デモグラ属性の時間的フィルタリング**
   - `attr_start_ymd <= month_end` かつ `attr_end_ymd >= month_start`: 分析月に有効な属性のみ抽出
   - `drop_duplicates(keep="last")`: 重複する場合は最新（より有効期限の長い）ものを採用

2. **ラベルカラムの抽出**
   - `psk_` で始まる（かつ `_cd` で終わらない）カラムのみを対象（コード値ではなくラベル値）
   - 例: `psk_003` = 年代（"15～19才", "20～29才"等）、`psk_003_cd` = コード値

3. **紐づけ**
   - `user_jsd` と `demo_month` を `monitor_id` で左結合
   - デモグラ情報がないユーザーは NaN で補完

4. **色スケール設定**
   - `JSD_VMIN = 0.0`: 下限（JSD は負にならない）
   - `JSD_VMAX = 95パーセンタイル`: 外れ値の影響を減らすため95パーセンタイルを上限に

---

## Cell 3-4: 年代別・性年代別の散布図

### 全実コード

```python
# 年齢別（psk_003）
age_order = sorted(user_jsd_demo["psk_003"].dropna().unique(),
                   key=lambda x: int(''.join(c for c in x if c.isdigit()) or "0"))
age_colors = {a: plt.cm.RdYlBu_r(i / max(len(age_order)-1, 1)) for i, a in enumerate(age_order)}
plot_jsd_scatter(user_jsd_demo, "psk_003", age_order, age_colors,
                 f"{MONTH} 年代別（psk_003）視聴時間 vs JSD")

# 性年代別（psk_007）
sa_order = sorted(user_jsd_demo["psk_007"].dropna().unique())
sa_colors = {s: plt.cm.tab20(i / max(len(sa_order)-1, 1)) for i, s in enumerate(sa_order)}
plot_jsd_scatter(user_jsd_demo, "psk_007", sa_order, sa_colors,
                 f"{MONTH} 性年代別（psk_007）視聴時間 vs JSD")
```

### 行ごとの詳細解説

1. **年代別分析** (psk_003)
   - 年代を数値順でソート（"15～19才" → "20～29才" → ... → "60～69才"）
   - カラーマップ `RdYlBu_r`: 赤（若年）→ 黄 → 青（高齢） のグラデーション
   - `plot_jsd_scatter()` で、各年代の視聴時間 vs JSD を描画

2. **性年代別分析** (psk_007)
   - 例: "男性 15～19才", "女性 15～19才", "男性 20～29才"...
   - `tab20`: 20色のカラーマップ（カテゴリが20以下の場合）

3. **グラフの読み方**
   - 横軸（対数スケール）: 各ユーザーの総視聴分
   - 縦軸: JSD（参照分布との乖離度）
   - **視聴時間が長い** → JSD が大きい：多様な番組を視聴している傾向
   - **視聴時間が短い** → JSD が小さい：限定的な番組を視聴している傾向

---

## Cell 6: Kruskal-Wallis 検定によるデモグラ属性スクリーニング

### 全実コード

```python
# 全psk属性に対してJSD・視聴時間のKruskal-Wallis検定 + η²を計算
psk_cols = [c for c in user_jsd_demo.columns if c.startswith("psk_")]
results_screen = []

for bunrui in sorted(user_jsd_demo["分類"].unique()):
    sub = user_jsd_demo[user_jsd_demo["分類"] == bunrui]
    for col in psk_cols:
        groups = sub.dropna(subset=[col]).groupby(col)
        if len(groups) < 2:
            continue
        jsd_groups = [g["JSD"].values for _, g in groups]
        if any(len(g) < 5 for g in jsd_groups):
            continue
        h_jsd, p_jsd = kruskal(*jsd_groups)
        n_total = sum(len(g) for g in jsd_groups)
        k = len(jsd_groups)
        eta2_jsd = (h_jsd - k + 1) / (n_total - k)

        view_groups = [g["視聴分合計"].values for _, g in groups]
        h_view, p_view = kruskal(*view_groups)
        eta2_view = (h_view - k + 1) / (n_total - k)

        results_screen.append({
            "分類": bunrui, "属性": col, "カテゴリ数": k, "n": n_total,
            "JSD_H": h_jsd, "JSD_p": p_jsd, "JSD_η²": max(eta2_jsd, 0),
            "視聴時間_H": h_view, "視聴時間_p": p_view, "視聴時間_η²": max(eta2_view, 0),
        })

screen_df = pd.DataFrame(results_screen)

print("=== JSD の分布差が大きいデモグラ属性 TOP15 ===")
print(screen_df.sort_values("JSD_η²", ascending=False).head(15)[
    ["分類", "属性", "カテゴリ数", "n", "JSD_η²", "JSD_p"]
].to_string(index=False))

print("\n=== 視聴時間 の分布差が大きいデモグラ属性 TOP15 ===")
print(screen_df.sort_values("視聴時間_η²", ascending=False).head(15)[
    ["分類", "属性", "カテゴリ数", "n", "視聴時間_η²", "視聴時間_p"]
].to_string(index=False))
```

### 行ごとの詳細解説

1. **Kruskal-Wallis 検定**
   - **帰無仮説**: 複数グループ間で分布に差がない
   - **検定統計量 H**: グループ間の分散を反映
   - **p値**: p < 0.05 で帰無仮説を棄却（差があると判定）
   - **式**: `H = (12 / (n(n+1))) * Σ R_i² / n_i - 3(n+1)`
     - R_i: グループ i のランク合計

2. **効果量 η² (エータ二乗)**
   - **式**: `η² = (H - k + 1) / (n - k)`
     - H: Kruskal-Wallis 統計量
     - k: グループ数
     - n: 総サンプル数
   - **解釈**:
     - η² < 0.01: 効果なし
     - 0.01 ≤ η² < 0.06: 小さい効果
     - 0.06 ≤ η² < 0.14: 中程度の効果
     - η² ≥ 0.14: 大きい効果
   - **例**: 年代による JSD の差が大きい → η² が大きい = 年代は JSD に強く影響する属性

3. **スクリーニング**
   - 全 psk 属性（年代、学歴、職業等）について、JSD と視聴時間の両方を検定
   - **重要度ランキング**: η² の降順で並べ、最も影響度の高い属性を特定

---

## Cell 8-9: 視聴行動指標の算出と可視化

### 全実コード

```python
# === 行動指標の前処理 ===
intage["date"] = intage["watch_date"].dt.date
intage["weekday"] = intage["watch_date"].dt.weekday

# チャンネル変更回数
intage_sorted = intage.sort_values(["monitor_id", "watch_date"])
ch_change = (
    (intage_sorted["放送局"] != intage_sorted["放送局"].shift()) |
    (intage_sorted["monitor_id"] != intage_sorted["monitor_id"].shift()) |
    (intage_sorted["date"] != intage_sorted["date"].shift())
).astype(np.int8)
daily_ch = ch_change.groupby([intage_sorted["monitor_id"], intage_sorted["date"]]).sum() - 1
avg_ch_change = daily_ch.clip(lower=0).groupby("monitor_id").mean().rename("avg_ch_change")
del intage_sorted, ch_change, daily_ch

# 平日・休日の1日平均視聴時間（分）
daily_view = intage.groupby(["monitor_id", "date", "weekday"]).size().reset_index(name="視聴分")
avg_weekday_view = daily_view[daily_view["weekday"] < 5].groupby("monitor_id")["視聴分"].mean().rename("avg_weekday_min")
avg_holiday_view = daily_view[daily_view["weekday"] >= 5].groupby("monitor_id")["視聴分"].mean().rename("avg_holiday_min")

print(f"CH変更: {len(avg_ch_change):,}人, 平日: {len(avg_weekday_view):,}人, 休日: {len(avg_holiday_view):,}人")
```

### 行ごとの詳細解説

1. **チャンネル変更回数の計算**
   - `shift()`: 前の行との比較（時系列順）
   - `ch_change = 1`: 放送局が異なる OR モニターが変わる OR 日付が変わる
   - `groupby().sum() - 1`: 各日の変更回数（初回は常に「変更」と見なされるため -1）
   - `clip(lower=0)`: 負の値を0に（チャンネルのみで日付を超える場合を除外）
   - `groupby("monitor_id").mean()`: 月間平均変更回数/日

2. **視聴時間の集計**
   - `weekday < 5`: 月〜金（平日）
   - `weekday >= 5`: 土日（休日）
   - 1日あたりの平均視聴分を月間で平均

3. **意味**
   - `avg_ch_change`: ザッピングの多さ（1日あたり何回チャンネルを変えるか）
   - `avg_weekday_min`, `avg_holiday_view`: 生活パターン（平日は短く、休日は長いか等）

---

## Cell 9: 行動指標 × JSD の散布図

### 全実コード

```python
# 平日視聴時間 / 休日視聴時間 / CH変更回数 × JSD 散布図
for metric, label, merge_src in [
    ("avg_weekday_min", "平日視聴時間", avg_weekday_view),
    ("avg_holiday_min", "休日視聴時間", avg_holiday_view),
    ("avg_ch_change",   "CH変更回数",   avg_ch_change),
]:
    merged = user_jsd.merge(merge_src, on="monitor_id", how="left")
    plot_quantile_scatter(merged, metric, label)
```

### 行ごとの詳細解説

1. **3つの指標を順次分析**
   - **平日視聴時間**: ニュース・情報番組の視聴が多い平日の行動
   - **休日視聴時間**: より広い時間帯での視聴
   - **CH変更回数**: ザッピング傾向

2. **`plot_quantile_scatter()` の表示内容**
   - 上位5%（赤）: 指標が最も高いユーザー
   - 下位5%（緑）: 指標が最も低いユーザー
   - 中間（灰色）: 中程度
   - **折れ線**: ビン別（視聴時間を20分割）での JSD 中央値

3. **解釈例**
   - CH変更が多い → JSD が大きい傾向？ → 多くの番組を試す → 多様なトピック視聴
   - 視聴時間が短い → JSD が小さい傾向？ → 限定的な番組のみ視聴

---

## Cell 11-12: 雇用形態 × 年代ヒートマップ

### 全実コード

```python
# === psk_012 × 分類 クロス集計（箱ひげ図） ===
job_col = "psk_012"
df_job = user_jsd_demo.dropna(subset=[job_col]).copy()

cross = df_job.groupby(["分類", job_col]).agg(
    n=("JSD", "size"), JSD_median=("JSD", "median"), JSD_mean=("JSD", "mean"),
    視聴分_median=("視聴分合計", "median"), 視聴分_mean=("視聴分合計", "mean"),
).reset_index()

# (以下、箱ひげ図描画と ヒートマップ)
```

### 行ごとの詳細解説

1. **職業（psk_012）別分析**
   - 例: "会社員", "自営業", "パート", "主婦", "学生", "無職" 等
   - 各職業内での JSD の中央値・平均値・視聴時間を集計

2. **箱ひげ図**
   - 職業ごとの JSD 分布を視覚化
   - 赤線: 中央値（その職業の典型的な JSD）
   - ボックス: 四分位範囲（データの中央50%）
   - **ヒゲ**: 四分位範囲の1.5倍まで
   - **外れ値**: 散布せず（showfliers=False）

3. **職業 × 年代ヒートマップ**
   - 行: 職業
   - 列: 年代
   - セル: JSD 中央値と サンプル数
   - **意味**: 「会社員 + 30代」の JSD は？ といった詳細な組み合わせを確認

---

## Cell 20-21: CH変更回数とトピック多様性の交互作用

### 全実コード

```python
# === 分類別：CH変更回数 → JSD の関係 ===

df_ch_jsd = user_jsd_demo.merge(avg_ch_change, on="monitor_id", how="inner")

# CH変更回数を十分位（デシル）に分割
df_ch_jsd["ch_decile"] = pd.qcut(df_ch_jsd["avg_ch_change"], 10, labels=False, duplicates="drop") + 1

bunruis = sorted(df_ch_jsd["分類"].unique())

# 1. 分類別のKruskal-Wallis（CH十分位→JSD）
print("=== 分類別: CH変更回数（十分位）→ JSD の効果量 ===")
for bunrui in bunruis:
    sub = df_ch_jsd[df_ch_jsd["分類"] == bunrui]
    groups = [g["JSD"].values for _, g in sub.groupby("ch_decile")]
    h, p = kruskal(*groups)
    n = len(sub)
    k = len(groups)
    eta2 = max((h - k + 1) / (n - k), 0)
    print(f"  {bunrui}: η²={eta2:.4f}, H={h:.1f}, p={p:.2e}, n={n:,}")

# (以下、折れ線・ヒートマップ)
```

### 行ごとの詳細解説

1. **CH十分位（デシル）グループ化**
   - `pd.qcut()`: データを等頻度で分割
   - 例: CH変更が少ない順に10等分 → D1（最少）～ D10（最多）
   - `duplicates="drop"`: ユーザー数が少ないグループは統合

2. **分類別 Kruskal-Wallis 検定**
   - 各分類で、CH十分位 × JSD の関係を検定
   - **大きな η²**: CH変更回数がその分類の JSD を大きく説明する
   - **例**: 「情報/ワイドショー」は η² が大きい → CH変更行動と多様性が強く関連

3. **折れ線グラフ**
   - x軸: 各十分位の平均 CH変更回数
   - y軸: JSD 中央値
   - **上昇傾向**: CH変更が多いユーザーほど多様（JSD が大きい）
   - **平坦**: CH変更と多様性が無関係

4. **ヒートマップ**
   - 行: 分類
   - 列: CH十分位（D1～D10）
   - セル: JSD 中央値
   - **パターン比較**: 分類によって JSD の変化パターンが異なるか？

---

## Cell 23-27: 情報取得層に限定した分析

### 全実コード

```python
# === 情報取得層の抽出 ===
NEWS_GENRES = [g for g in intage["番組ジャンル"].dropna().unique()
               if "ニュース" in g or "報道" in g or "情報" in g or "ワイドショー" in g]

# モニター別：全視聴時間 vs 情報系視聴時間
total_by_monitor = intage.groupby("monitor_id").size().rename("total_min")
news_by_monitor = intage[intage["番組ジャンル"].isin(NEWS_GENRES)].groupby("monitor_id").size().rename("news_min")

monitor_ratio = pd.DataFrame({"total_min": total_by_monitor, "news_min": news_by_monitor}).fillna(0)
monitor_ratio["news_ratio"] = monitor_ratio["news_min"] / monitor_ratio["total_min"]

# 閾値による分類
threshold = 0.50
n_below = (monitor_ratio["news_ratio"] <= threshold).sum()
n_above = (monitor_ratio["news_ratio"] > threshold).sum()
print(f"\n=== 閾値 {threshold*100:.0f}% での分類 ===")
print(f"  情報非取得層（≤{threshold*100:.0f}%）: {n_below:,}人 ({n_below/len(monitor_ratio)*100:.1f}%)")
print(f"  情報取得層  （>{threshold*100:.0f}%）: {n_above:,}人 ({n_above/len(monitor_ratio)*100:.1f}%)")

# 情報取得層のmonitor_idリスト
info_monitors = monitor_ratio[monitor_ratio["news_ratio"] > threshold].index
```

### 行ごとの詳細解説

1. **情報系ジャンルの特定**
   - "ニュース", "報道", "情報", "ワイドショー" を含むジャンル名を抽出

2. **視聴比率の計算**
   - `monitor_ratio["news_ratio"]`: 月間総視聴分に占める情報系の割合
   - **閾値 50%**: 半分以上を情報番組に費やしているユーザーを「情報取得層」と定義

3. **意義**
   - テレビで情報を**意図的に取得**しようとしているユーザーに限定
   - エンタメのみ視聴のユーザーを除外
   - **バイアス除去**: 関心のないジャンルのトピックで JSD が小さいことを理由に、「多様性が低い」と誤判定するのを防ぐ

4. **情報取得層での再分析**
   - Cell 25～27 で、同じ分析を情報取得層のみに限定して実施
   - → より「意図的な視聴」の結果を反映した分析結果が得られる

---

## 統計手法・指標の詳細解説

### Kruskal-Wallis 検定

**用途**: 3つ以上の独立したグループの分布が異なるかを検定（t検定の多群版）

**帰無仮説**: 全グループの分布は同一

**検定統計量**:
```
H = (12 / (n(n+1))) * Σ(R_i² / n_i) - 3(n+1)
R_i: グループ i のランク合計
n_i: グループ i のサンプル数
```

**有意性**: p < 0.05 で帰無仮説を棄却（差あり）

**効果量 η² (Rank Biserial Correlation)**:
```
η² = (H - k + 1) / (n - k)
```
- 0.01～0.06: 小さい効果
- 0.06～0.14: 中程度の効果
- 0.14～: 大きい効果

---

## 重要な変数と出力

| 変数名 | 型 | 内容 |
|--------|----|----|
| `user_jsd` | DataFrame | (monitor_id, 分類, JSD, 視聴分合計) |
| `user_jsd_demo` | DataFrame | user_jsd + デモグラ属性（psk_XXX） |
| `screen_df` | DataFrame | Kruskal-Wallis 検定結果（JSD_η², 視聴時間_η²等） |
| `avg_ch_change` | Series | ユーザー別 平均CH変更回数/日 |
| `avg_weekday_view` | Series | ユーザー別 平日1日平均視聴分 |
| `avg_holiday_view` | Series | ユーザー別 休日1日平均視聴分 |
| `info_monitors` | Index | 情報取得層のモニターID |
| `JSD_VMIN`, `JSD_VMAX` | float | ヒートマップの色スケール（0.0～95パーセンタイル） |

---

## 次のステップ

**→ NB05 (mxi_negindex) で使用**:
- ネガティブ情報（事件・犯罪・災害等）の直後のチャンネル変更行動を分析
- NB04 で算出した JSD（トピック多様性）との相互作用を検討可能
