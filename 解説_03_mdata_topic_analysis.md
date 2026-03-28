# Notebook 03: トピック分析 — メタデータ (mdata) の詳細解説

## 概要
このノートブックは、テレビ番組の放送メタデータ（番組開始年月、分類、放送局、トピック）から、**各放送局のトピック分布の多様性**を Jensen-Shannon Divergence (JSD) とエントロピーで分析する。

---

## Cell 0: インポート・データ読込・JSD/エントロピー計算関数の定義

### 全実コード

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

plt.rcParams["font.family"] = "Noto Sans CJK JP"

df = pd.read_parquet("/home/ciwai/work/data/0014_MTC_information_health/mdata/mdata_with_topics.parquet")
df["シーン放送秒"] = pd.to_timedelta(df["シーン放送時間"]).dt.total_seconds()


def compute_jsd_entropy(data, top_n=20):
    """月別・分類別・放送局別にJSDとエントロピーを計算する"""
    agg = data.groupby(["番組開始年月", "分類", "放送局", "topic"])["シーン放送秒"].sum().reset_index()
    results = []

    for (month, bunrui), grp in agg.groupby(["番組開始年月", "分類"]):
        topic_total = grp.groupby("topic")["シーン放送秒"].sum()
        top = topic_total.nlargest(min(top_n, len(topic_total))).index.tolist()

        grp = grp.copy()
        grp["topic_label"] = grp["topic"].apply(lambda x: x if x in top else -1)
        pivot = grp.groupby(["放送局", "topic_label"])["シーン放送秒"].sum().unstack(fill_value=0)

        all_labels = sorted(top) + [-1]
        for col in all_labels:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[all_labels]

        dist = pivot.div(pivot.sum(axis=1), axis=0)
        mean_dist = dist.mean(axis=0).values

        for station in dist.index:
            p = dist.loc[station].values
            results.append({
                "月": month, "分類": bunrui, "放送局": station,
                "JSD": jensenshannon(p, mean_dist) ** 2,
                "エントロピー": entropy(p, base=2),
            })

    return pd.DataFrame(results)


def plot_jsd_entropy(res_df, title=None, exclude_month="2024-08", top_n=20):
    """JSD×エントロピー散布図を月別×分類別に描画する"""
    ref = res_df[res_df["月"] != exclude_month]
    x_min, x_max = ref["エントロピー"].min(), ref["エントロピー"].max()
    y_min, y_max = ref["JSD"].min(), ref["JSD"].max()
    xm, ym = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
    xlim = (x_min - xm, max(x_max + xm, np.log2(top_n + 1) + 0.2))
    ylim = (max(0, y_min - ym), y_max + ym)

    # 全局平均（十字線用）
    global_ent_mean = res_df["エントロピー"].mean()
    global_jsd_mean = res_df["JSD"].mean()

    # エントロピー理論最大値
    max_entropy = np.log2(top_n + 1)

    months = sorted(res_df["月"].unique())
    bunruis = sorted(res_df["分類"].unique())
    colors = {s: c for s, c in zip(sorted(res_df["放送局"].unique()), plt.cm.tab10.colors)}

    fig, axes = plt.subplots(len(bunruis), len(months),
                             figsize=(4 * len(months), 5 * len(bunruis)), squeeze=False)
    if title:
        fig.suptitle(title, fontsize=14, y=1.01)

    for i, bunrui in enumerate(bunruis):
        for j, month in enumerate(months):
            ax = axes[i][j]
            sub = res_df[(res_df["分類"] == bunrui) & (res_df["月"] == month)]

            # 補助線1: 全局平均（十字線、実線灰色）
            ax.axhline(global_jsd_mean, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
            ax.axvline(global_ent_mean, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)

            # 補助線2: 月・分類ごとの中央値（点線青）
            if len(sub) > 0:
                ax.axhline(sub["JSD"].median(), color="steelblue", linestyle=":", alpha=0.6, linewidth=0.8)
                ax.axvline(sub["エントロピー"].median(), color="steelblue", linestyle=":", alpha=0.6, linewidth=0.8)

            # 補助線3: エントロピー理論最大値（縦線赤）
            ax.axvline(max_entropy, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

            for _, row in sub.iterrows():
                ax.scatter(row["エントロピー"], row["JSD"], color=colors[row["放送局"]], s=60, alpha=0.8)
                ax.annotate(row["放送局"], (row["エントロピー"], row["JSD"]), fontsize=7, ha="left", va="bottom")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("エントロピー (bits)")
            ax.set_ylabel("JS Divergence")
            ax.set_title(f"{bunrui} / {month}", fontsize=9)

    # 凡例
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[s], markersize=8, label=s)
               for s in sorted(colors)]
    handles += [
        plt.Line2D([0], [0], color="gray", linestyle="-", alpha=0.4, label="全局平均"),
        plt.Line2D([0], [0], color="steelblue", linestyle=":", alpha=0.6, label="月別中央値"),
        plt.Line2D([0], [0], color="red", linestyle="--", alpha=0.5, label=f"エントロピー理論最大値{max_entropy:.2f}"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


# 共通フィルタ: トピック割当あり、教育局除外
target = df[(df["topic"].notna()) & (df["放送局"] != "教育")]

print(f"データ件数: {len(df):,} → フィルタ後: {len(target):,}")
```

### 行ごとの詳細解説

1. **インポート**
   - `jensenshannon`: Jensen-Shannon距離の計算用
   - `entropy`: シャノンエントロピーの計算用

2. **データ読込**
   - `mdata_with_topics.parquet`: トピック割当済みのメタデータ
   - `シーン放送秒`: `シーン放送時間` (timedelta) を秒単位の数値に変換

3. **`compute_jsd_entropy(data, top_n=20)` 関数**
   - **目的**: 月別・分類別・放送局別に、各放送局のトピック分布の多様性を計測
   - **入力**: `data` = データフレーム、`top_n` = 上位トピック数（デフォルト20）
   - **処理フロー**:
     - `agg`: (月, 分類, 放送局, トピック) でグループ化し、放送秒数を合計
     - **トピックのランク付け**: 各(月, 分類)ごとに、全体での放送秒数が多いトピックTOP N を「top」リストに格納
     - **ラベル変換**: top N に入っていないトピックは -1 に統一（「その他」カテゴリ）
     - **ピボット**: (放送局 × トピックラベル) のクロス集計。各局が各トピックに費やした秒数を行列化
     - **正規化**: 各行（放送局）の合計を1になるよう正規化 → 分布ベクトル p_i
     - **全体平均分布**: 全放送局の分布の平均 = mean_dist
     - **JSD計算**: 各放送局 i の分布 p_i と全体平均 mean_dist との間の JSD²を計算
       - JSD² = 0: 局の分布が平均と完全に一致
       - JSD² が大きい: その局は独特なトピック選択（視聴者の平均的な嗜好と異なる）
     - **エントロピー計算**: 各局の分布 p_i の多様性（値が1に近い = より多くのトピックが均等に含まれている）
   - **戻り値**: (月, 分類, 放送局, JSD, エントロピー) のデータフレーム

4. **`plot_jsd_entropy(res_df, ...)` 関数**
   - **目的**: JSD × エントロピー散布図を月別×分類別に描画
   - **キー引数**:
     - `exclude_month="2024-08"`: 軸の決定から除外する月（最後の月で通常データが不完全なため）
     - `top_n=20`: エントロピーの理論最大値計算用
   - **軸範囲決定**: `exclude_month` を除いたデータから x_min, x_max, y_min, y_max を算出（外れ値の影響を除去）
   - **補助線**:
     - 灰色実線: 全局の平均値（グローバルベンチマーク）
     - 青点線: 月・分類ごとの中央値（局ごとのばらつきの中心）
     - 赤破線: エントロピー理論最大値（分布が均等な時の最大エントロピー）
   - **プロット**: 各(分類, 月)ごとに1つのサブプロット（分類数 × 月数 のグリッド）

5. **フィルタリング**
   - `target`: トピック割当ありかつ教育局除外（教育はニュース番組が少ないため除外）

---

## Cell 1: トピックの確度分析 — topic_prob >= 0.8 の残存率

### 全実コード

```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"

has_topic = df[df["annual_topic_id"].notna()]

# 分類×トピック別の件数集計
total = has_topic.groupby(["分類", "annual_topic_id"]).size().rename("全件")
over_08 = has_topic[has_topic["topic_prob"] >= 0.8].groupby(["分類", "annual_topic_id"]).size().rename("prob>=0.8")
rep_words = has_topic.groupby(["分類", "annual_topic_id"])["topic_representative_words"].first()

comp = pd.concat([total, over_08, rep_words], axis=1).fillna({"prob>=0.8": 0}).astype({"prob>=0.8": int})
comp["残存率(%)"] = (comp["prob>=0.8"] / comp["全件"] * 100).round(1)

# 代表語付きラベル
labels = [f"[{b}] t{int(t)}: {comp.loc[(b, t), 'topic_representative_words'][:20]}" for b, t in comp.index]

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(labels) * 0.35)))

y = np.arange(len(labels))

ax = axes[0]
ax.barh(y, comp["全件"], label="全件", alpha=0.5)
ax.barh(y, comp["prob>=0.8"], label="prob>=0.8", alpha=0.8)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("件数")
ax.set_title("トピック別 件数比較")
ax.legend()
ax.invert_yaxis()

ax = axes[1]
colors = ["#e74c3c" if r < 50 else "#f39c12" if r < 70 else "#2ecc71" for r in comp["残存率(%)"]]
ax.barh(y, comp["残存率(%)"], color=colors)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("残存率 (%)")
ax.set_title("topic_prob >= 0.8 フィルタ後の残存率")
ax.axvline(x=50, color="red", linestyle="--", alpha=0.5)
ax.axvline(x=80, color="green", linestyle="--", alpha=0.5)
ax.set_xlim(0, 105)
ax.invert_yaxis()

plt.tight_layout()
plt.show()

# サマリ表示
print(comp[["全件", "prob>=0.8", "残存率(%)"]].to_string())
print(f"\n全体: {total.sum():,} → {int(over_08.sum()):,} ({over_08.sum()/total.sum()*100:.1f}%)")
```

### 行ごとの詳細解説

1. **トピック確度の集計**
   - `has_topic`: `annual_topic_id` がある行（トピック割当がある）のみ
   - `total`: 分類×トピックごとの全件数
   - `over_08`: 同じグループで `topic_prob >= 0.8` の件数
   - `rep_words`: 各トピックの代表語（最初の行から取得）

2. **残存率の計算**
   - `comp["残存率(%)"]`: prob >= 0.8 のフィルタで何%が残るかの指標
   - **低い場合**: トピック確度が不安定で、フィルタを強くしすぎると多くが除外される
   - **高い場合**: トピック割当が確度高い

3. **可視化**
   - **左グラフ**: 全件 vs prob >= 0.8 の比較（積み上げ）
   - **右グラフ**: 残存率を色分け
     - 赤（<50%）: 確度が低い
     - 橙（50-70%）: 中程度
     - 緑（≥70%）: 確度が高い

4. **サマリ表示**: 全体での残存率を最後に出力

---

## Cell 2: 全番組ジャンル（TOP 20 + その他）の JSD × エントロピー分析

### 全実コード

```python
TOP_N = 20
res_all = compute_jsd_entropy(target, top_n=TOP_N)
plot_jsd_entropy(res_all, title=f"全番組ジャンル（top {TOP_N} + その他）")
```

### 行ごとの詳細解説

1. **`TOP_N = 20`**: 上位20トピックを考慮（それ以外は「その他」 = -1）
2. **`compute_jsd_entropy(target, top_n=20)`**:
   - `target` データで JSD・エントロピーを計算（→ Cell 0 参照）
3. **`plot_jsd_entropy(...)`**:
   - 全月・全分類（ニュース/報道、情報/ワイドショー等）の散布図を描画
   - **各プロットは1つの放送局を表す**
   - JSD が大きい = その局は平均的な視聴者とトピック嗜好が異なる
   - エントロピーが大きい = より多くのトピックが均等に放送されている

---

## Cell 3: ジャンル別フィルタリング（ニュース/報道、情報/ワイドショー）

### 全実コード

```python
genre_filters = {
    "ニュース/報道": "ニュース/報道 A",
    "情報/ワイドショー": "情報/ワイドショー B",
}

# 軸範囲を統一するため全ジャンルの結果を結合
all_res = {}
for label, val in genre_filters.items():
    all_res[label] = compute_jsd_entropy(target[target["番組ジャンル"] == val], top_n=TOP_N)

combined = pd.concat(all_res.values())
ref = combined[combined["月"] != "2024-08"]
x_min, x_max = ref["エントロピー"].min(), ref["エントロピー"].max()
y_min, y_max = ref["JSD"].min(), ref["JSD"].max()
xm, ym = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
shared_xlim = (x_min - xm, x_max + xm)
shared_ylim = (max(0, y_min - ym), y_max + ym)

for label, res in all_res.items():
    plot_jsd_entropy(res, title=f"番組ジャンル: {label}（top {TOP_N} + その他）")
```

### 行ごとの詳細解説

1. **ジャンル選別**
   - `genre_filters`: 分析対象の番組ジャンルを指定
   - ニュース/報道系と情報/ワイドショー系に分けて分析

2. **軸の統一化**
   - `combined`: 全ジャンルの結果を結合
   - `shared_xlim`, `shared_ylim`: 複数のプロットで同じ軸範囲を使用し、比較可能にする

3. **ジャンル別プロット**
   - 各ジャンルごとに個別に JSD × エントロピー散布図を生成
   - **ニュース/報道**: 情報系番組としてフォーマル
   - **情報/ワイドショー**: より多様なテーマ（生活情報、エンタメ等）を含む可能性

---

## Cell 4: 参照分布の保存（→ NB04 で使用）

### 全実コード

```python
import pickle

TOP_N = 20
mdata_target = df[(df["topic"].notna()) & (df["放送局"] != "教育")]
agg = mdata_target.groupby(["番組開始年月", "分類", "放送局", "topic"])["シーン放送秒"].sum().reset_index()

ref_data = {}  # {(month, bunrui): {"top_n": [...], "mean_dist": array, "labels": [...]}}

for (month, bunrui), grp in agg.groupby(["番組開始年月", "分類"]):
    topic_total = grp.groupby("topic")["シーン放送秒"].sum()
    top = topic_total.nlargest(min(TOP_N, len(topic_total))).index.tolist()

    grp = grp.copy()
    grp["topic_label"] = grp["topic"].apply(lambda x: x if x in top else -1)
    pivot = grp.groupby(["放送局", "topic_label"])["シーン放送秒"].sum().unstack(fill_value=0)

    all_labels = sorted(top) + [-1]
    for col in all_labels:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[all_labels]

    dist = pivot.div(pivot.sum(axis=1), axis=0)
    mean_dist = dist.mean(axis=0).values

    ref_data[(month, bunrui)] = {
        "top_n": top,
        "labels": all_labels,
        "mean_dist": mean_dist,
    }

save_path = "/home/ciwai/work/data/0014_MTC_information_health/mdata/topic_ref_distributions.pkl"
with open(save_path, "wb") as f:
    pickle.dump(ref_data, f)

print(f"保存完了: {save_path}")
print(f"キー数: {len(ref_data)}")
for k, v in list(ref_data.items())[:3]:
    print(f"  {k}: top_n={len(v['top_n'])}件, labels={len(v['labels'])}カテゴリ")
```

### 行ごとの詳細解説

1. **参照分布の構築**
   - Cell 0 の計算と同じロジックで、各(月, 分類)ごとに以下を保存:
     - `top_n`: TOP_N のトピック ID のリスト
     - `mean_dist`: 全放送局の平均トピック分布（正規化済み）
     - `labels`: ソート済みのトピックラベル（計算順序を固定）

2. **ピクル形式での保存**
   - Python オブジェクト（辞書）を直列化してファイルに保存
   - **用途**: NB04 で同じ参照分布を読み込み、個別ユーザーの JSD を計算

3. **検証出力**
   - キー数（月×分類の組み合わせ数）を表示
   - 最初の3つのエントリの内容を表示

---

## 統計手法・指標の詳細解説

### Jensen-Shannon Divergence (JSD)

**定義**:
```
JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
M = 0.5 * (P + Q)  # P と Q の平均分布
KL(P || M) = Σ P(i) * log(P(i) / M(i))  # Kullback-Leibler 発散
```

**意味**:
- **0 に近い**: P と Q の分布が類似（多くの番組が類似したトピック構成）
- **大きい**: P と Q が大きく異なる（独特なトピック選択）
- **例**:
  - 全局平均が[ニュース 50%, 天気 30%, スポーツ 20%]で、ある局が[ニュース 80%, 天気 20%]の場合、JSD が大きくなる

### シャノンエントロピー

**定義**:
```
H(P) = -Σ P(i) * log₂(P(i))
```

**意味**:
- **0 に近い**: 1つのトピックに集中（例：ニュースのみ）
- **大きい**: 複数のトピックが均等に分散（多様な番組構成）
- **最大値**: log₂(K) = log₂(top_n + 1)（K個のカテゴリが完全に均等）
- **例**:
  - ニュースのみ 100% → H = 0
  - 21個のカテゴリに均等 → H ≈ 4.39 bits
  - ニュース 50%, その他 50% → H ≈ 1 bit

### エントロピー理論最大値

**式**: `max_entropy = log₂(top_n + 1)`
- top_n = 20 の場合: log₂(21) ≈ 4.39 bits
- **グラフの赤破線**: この値以上のエントロピーは理論的にあり得ない
- エントロピーがこの値に近い局 = 全トピックが均等に含まれている

---

## 重要な変数・出力

| 変数名 | 型 | 内容 |
|--------|----|----|
| `target` | DataFrame | トピック割当ありかつ教育局除外のデータ |
| `res_all` | DataFrame | (月, 分類, 放送局, JSD, エントロピー) |
| `ref_data` | dict | `{(month, bunrui): {top_n, mean_dist, labels}}` |
| `save_path` | str | `topic_ref_distributions.pkl` の保存パス |

---

## 次のステップ

**→ NB04 (mxi_user_analysis) で使用**:
- `ref_data` を読み込み、個別ユーザーのトピック視聴分布を参照分布と比較
- 各ユーザーの JSD を計算し、デモグラとの関連を分析
