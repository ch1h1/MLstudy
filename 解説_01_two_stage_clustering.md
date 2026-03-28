# Notebook 01: 二段階クラスタリング - コード全文解説

## 概要
このノートブックは、月次×分類別のBERTopicクラスタリングで得たトピックを、年間単位でさらにクラスタリングすることで、ニューステーマの1年間の推移を追跡するシステムです。3つのステップに分かれています：

1. **Step1**: 月次トピックの代表ベクトル（重心）を抽出
2. **Step2**: 代表ベクトルを年間でクラスタリング（分類ごと）
3. **Step3**: 結果の整理・可視化

---

## Cell 0: タイトルと概要（Markdown）

```markdown
# 二段階クラスタリング：月次トピックの年間追跡

月次×分類別BERTopicクラスタリングの粒度を保ちつつ、年間を通じたトピックの継続・消長を追跡する。

- **Step1**: 月次クラスタの代表ベクトル抽出（embedding平均）
- **Step2**: 代表ベクトルの年間クラスタリング（分類ごとにAgglomerativeClustering）
- **Step3**: 結果整理・可視化
```

**説明**：ノートブック全体の目的と処理の流れを説明するセルです。

---

## Cell 1: ライブラリインポートと設定

```python
import pandas as pd
import numpy as np
import glob
import os
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
warnings.filterwarnings('ignore')

# === 設定 ===
EMBEDDING_MODEL_NAME = "cl-nagoya/ruri-large"
# 代替: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 分類ごとのdistance_threshold（cosine距離、0〜2の範囲。小さいほど細かく分かれる）
DISTANCE_THRESHOLDS = {
    '社会': 0.1,
    '政治・国際': 0.1,
    '暮らし': 0.03,
}

# TARGET_CATEGORIES = ['社会', '政治・国際', '暮らし']
TARGET_CATEGORIES = ['政治・国際', '社会']

DATA_PATH = "/home/ciwai/work/data/0014_MTC_information_health/mdata"
OUTPUT_PATH = "/home/ciwai/work/0014_MTC/information_health/dynamic_topic_sys/2026_information_health/information_health/"
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 1-9 | ライブラリインポート | pandas/numpy：データ処理、glob：ファイル一覧取得、SentenceTransformer：テキスト埋め込みモデル、AgglomerativeClustering：階層的クラスタリング、cosine_distances：コサイン距離計算、matplotlib：グラフ描画 |
| 12 | `matplotlib.rcParams['font.family']` | グラフに日本語を表示するフォント設定（Noto Sans CJK JP） |
| 13 | `warnings.filterwarnings('ignore')` | 警告メッセージを非表示にする設定 |
| 16 | `EMBEDDING_MODEL_NAME` | テキストを数値ベクトルに変換するモデル名。ruri-largeは日本語に特化した高精度モデル |
| 20-24 | `DISTANCE_THRESHOLDS` | AgglomerativeClusteringの距離閾値を分類別に定義。例えば「社会」は0.1以下のコサイン距離なら同じトピックと見なす。数値が小さいほど細かく分かれる（具体例：0.03なら厳密、0.3なら緩い） |
| 27 | `TARGET_CATEGORIES` | 分析対象の分類。現在は「政治・国際」と「社会」の2つ |
| 29-30 | パスの定義 | データの入出力パス。DATA_PATHにCSVが格納、OUTPUT_PATHに結果を保存 |

---

## Cell 2: データ読み込みセクション（Markdown）

```markdown
## データ読み込み
```

**説明**：続くセルの処理内容を示すセクションヘッダです。

---

## Cell 3: CSVの読み込みと年月抽出

```python
files = sorted(glob.glob(os.path.join(DATA_PATH, "*.csv")))
df = pd.concat([pd.read_csv(f, encoding='cp932') for f in files], ignore_index=True)

# 番組開始日時24から年月を抽出して「番組開始年月」列を追加
df['番組開始年月'] = pd.to_datetime(df['番組開始日時24']).dt.to_period('M').astype(str)

print(f"ファイル数: {len(files)}")
print(f"全体の行数: {len(df)}")
print(f"分類の種類: {df['分類'].unique()}")
print(f"年月の種類: {sorted(df['番組開始年月'].unique())}")
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 1 | `sorted(glob.glob(...))` | DATA_PATH配下の全CSVファイルのパスをアルファベット順で取得。例：['file1.csv', 'file2.csv', ...] |
| 2 | `pd.concat([...])` | 複数のCSVを1つのDataFrameに結合。encoding='cp932'は日本語テキストのエンコーディング（Windows形式） |
| 5 | `pd.to_datetime(...)` | 「番組開始日時24」列を日時型に変換 |
| 5 | `.dt.to_period('M')` | 日時を「年月」単位に変換（例：2025-03-15 → 2025-03）。→ Cell 1で定義したdatetime操作 |
| 5 | `.astype(str)` | 「年月」を文字列に変換（例：'2025-03'）。以降の分類・集計で使用 |
| 7-10 | print文 | データの統計情報を出力。ファイル数、総行数、分類の種類、年月の範囲を確認 |

**データフロー例**：
```
入力: DATA_PATH配下のCSVファイル群
  ↓
全CSVを読み込み・結合
  ↓
「番組開始日時24」→ 「番組開始年月」に変換
  ↓
出力: df（全データ、新列「番組開始年月」を含む）
```

---

## Cell 4: 対象分類へのフィルタリングとテキスト前処理

```python
# 対象分類のみに絞る
df_target = df[df['分類'].isin(TARGET_CATEGORIES)].copy()

# シーン放送時間を秒数に変換
def parse_airtime(val):
    """'0 days 00:06:26' のような文字列を秒数に変換"""
    try:
        td = pd.to_timedelta(val)
        return td.total_seconds()
    except Exception:
        return 0.0

df_target['シーン放送時間_sec'] = df_target['シーン放送時間'].apply(parse_airtime)

# === メモのクリーニング ===
import re

# タグ名+内容ごと全削除（ニューステーマに無関係）
REMOVE_TAGS = {
    '出演', '声の出演', '告知', '資料・協力', '制作著作', '制作',
    '音楽', 'エンディング曲', 'オープニング曲', '提供',
    'ゲスト', '解説・コメンテーター', '次回出演', '電話コメント',
    '原作', '地図提供', 'プレゼント', '出場',
}

# タグ名のみ除去し内容は残す（本文と地続きで意味がある）
STRIP_TAG_ONLY = {
    '施設', '商品', '人物', 'コメント', 'コメント文',
    '会見', '議会・演説', '飲食店', '店舗', '宿泊施設',
    'ロケ地', 'ランキング',
}

def clean_memo(text):
    """メモからタグ部分を処理（全削除 or タグ名のみ除去）"""
    if pd.isna(text) or text.strip() == '':
        return ''
    # 全削除: 【タグ名】から次の【または文末までを削除
    for tag in REMOVE_TAGS:
        text = re.sub(rf'【{re.escape(tag)}】[^【]*', '', text)
    # タグ名のみ除去: 【タグ名】だけ消して後続テキストは残す
    for tag in STRIP_TAG_ONLY:
        text = re.sub(rf'【{re.escape(tag)}】', '', text)
    # 全角・半角数字を除去
    text = re.sub(r'[0-9０-９]+', '', text)
    return text.strip()

df_target['メモ_clean'] = df_target['メモ'].apply(clean_memo)

# === ヘッドラインのクリーニング ===
# ＜＞で囲まれた番組名等を除去（例: ＜ＮＮＮ・ＮＥＷＳＺＩＰ！＞）
df_target['ヘッドライン_clean'] = df_target['ヘッドライン'].str.replace(r'＜[^＞]*＞', '', regex=True).str.strip()

# === ヘッドライン + クリーンメモの結合テキスト ===
df_target['text_combined'] = (
    df_target['ヘッドライン_clean'].fillna('') + '。' + df_target['メモ_clean']
).str.strip('。').str.strip()

# クリーニング効果の確認
n_empty_before = (df_target['メモ'].fillna('').str.strip() == '').sum()
n_empty_after = (df_target['メモ_clean'] == '').sum()
print(f"対象行数: {len(df_target)}")
print(f"メモ空（クリーニング前）: {n_empty_before}")
print(f"メモ空（クリーニング後）: {n_empty_after} (+{n_empty_after - n_empty_before})")
print(f"\nクリーニング例:")
sample_idx = df_target[df_target['メモ'].fillna('').str.contains('【施設】|【声の出演】', na=False)].head(3).index
for idx in sample_idx:
    print(f"  BEFORE: {str(df_target.loc[idx, 'メモ'])[:150]}...")
    print(f"  AFTER:  {df_target.loc[idx, 'メモ_clean'][:150]}...")
    print()
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 2 | `df_target = df[df['分類'].isin(TARGET_CATEGORIES)].copy()` | → Cell 1で定義したTARGET_CATEGORIES（['政治・国際', '社会']）に属する行のみを抽出して新DataFrameを作成。`.copy()`で元のdfに影響を与えないようコピーを取得 |
| 5-10 | `parse_airtime関数` | 引数：val（'0 days 00:06:26'形式の時間文字列）、戻り値：秒数（float）。`pd.to_timedelta`で時間型に変換、`.total_seconds()`で秒に変換。エラー時は0.0を返す |
| 12 | `df_target['シーン放送時間_sec']` | parse_airtime関数をdf_target['シーン放送時間']の各行に適用。→月別・トピック別の放送時間合計を計算する際に利用 |
| 19-27 | `REMOVE_TAGS` | メモから完全に削除すべきタグ（出演者名、音楽など、ニューステーマと無関係）を定義 |
| 29-36 | `STRIP_TAG_ONLY` | タグ名のみを削除し、内容は保持すべきタグ（施設名、人物名など、本文と地続きで意味がある）を定義 |
| 38-49 | `clean_memo関数` | 引数：text（メモ文字列）、戻り値：クリーニング済みテキスト。処理フロー：①空文字判定、②REMOVE_TAGSのタグと内容を削除、③STRIP_TAG_ONLYのタグ名のみ削除、④数字を削除、⑤前後の空白除去 |
| 38行の正規表現 | `rf'【{re.escape(tag)}】[^【]*'` | 【tag】から次の【または文末までの全テキストにマッチ。`re.escape()`で特殊文字をエスケープ |
| 42行の正規表現 | `rf'【{re.escape(tag)}】'` | タグ名（【...】）のみにマッチ。内容は保持 |
| 44行 | `re.sub(r'[0-9０-９]+', '', text)` | 半角数字と全角数字を削除。行番号や数値は意味がないため除去 |
| 48 | `df_target['メモ_clean']` | clean_memo関数を全メモに適用 |
| 51 | `.str.replace(r'＜[^＞]*＞', '', regex=True)` | ＜＞で囲まれたテキスト（番組名など）を削除。例：「＜ＮＮＮ・ＮＥＷＳＺＩＰ！＞イスラエル軍が...」→「イスラエル軍が...」 |
| 55-58 | `text_combined` | ヘッドラインとクリーンメモを「。」で連結。例：「イスラエル軍が侵攻。戦闘が激化」。両方空なら「。」のみ残るので、`.strip('。')`で除去 |
| 61-68 | クリーニング効果の確認 | クリーニング前後でメモの空行数を比較。クリーニング例を表示して手作業確認 |

**具体例**：
```
BEFORE: 【出演】山田太郎。【施設】東京駅。【数字】5名が負傷
AFTER:  東京駅。5名が負傷
  →「出演」は完全削除、「施設」はタグ名のみ削除、数字は削除
```

**データフロー**：
```
入力: df（全データ）
  ↓
TARGET_CATEGORIESでフィルタ → df_target
  ↓
シーン放送時間を秒数に変換
ヘッドライン・メモをクリーニング
text_combinedを生成
  ↓
出力: df_target（テキスト前処理済み）
```

---

## Cell 5: Step1セクションヘッダ（Markdown）

```markdown
## Step1: 月次クラスタの代表ベクトル抽出
```

**説明**：以降の処理ステップを示すセクションヘッダです。

---

## Cell 6: Embeddingモデルの読み込み

```python
# embeddingモデルの読み込み
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"モデル読み込み完了: {EMBEDDING_MODEL_NAME}")
print(f"embedding次元数: {model.get_sentence_embedding_dimension()}")
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 2 | `SentenceTransformer(EMBEDDING_MODEL_NAME)` | → Cell 1で定義したEMBEDDING_MODEL_NAME（"cl-nagoya/ruri-large"）を読み込む。このモデルは日本語テキストを数値ベクトル（embedding）に変換 |
| 3-4 | print文 | モデル名と出力ベクトルの次元数を表示。ruri-largeは通常1024次元 |

---

## Cell 7: テキスト全体のembedding化

```python
# 結合テキストで一括embedding
texts = df_target['text_combined'].fillna('').tolist()

# ruri系モデルはプレフィックスを使用
if 'ruri' in EMBEDDING_MODEL_NAME.lower():
    texts = ['文章: ' + t for t in texts]
elif 'e5' in EMBEDDING_MODEL_NAME.lower():
    texts = ['passage: ' + t for t in texts]

print(f"embeddingするテキスト数: {len(texts)}")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
df_target['embedding_idx'] = range(len(df_target))
print(f"embedding完了: shape={embeddings.shape}")
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 2 | `df_target['text_combined'].fillna('').tolist()` | クリーニング済みの結合テキストをリスト化。NaNは空文字に置換 |
| 5-7 | プレフィックス付与 | ruri系モデルは「文章: 」、e5系モデルは「passage: 」というプレフィックスを先頭に付与すると性能が向上。この指示は各モデルのドキュメント参照 |
| 10 | `model.encode(texts, ...)` | テキストリストをembeddingベクトルに変換。batch_size=256は一度に256件処理（メモリ効率化）。show_progress_bar=Trueで進捗表示 |
| 11 | `df_target['embedding_idx']` | 後で特定行のembeddingを参照する際に使用するインデックス。例：embedding_idx=5なら embeddings[5] |
| 12 | `embeddings.shape` | embeddingsの形状を表示。例：(10000, 1024)なら10000件のテキスト、1024次元のベクトル |

**データフロー**：
```
入力: df_target['text_combined']（クリーニング済みテキスト）
  ↓
プレフィックス付与（ruri系なら「文章: 」を先頭に追加）
  ↓
model.encode() で数値ベクトルに変換
  ↓
出力: embeddings（NumPy配列、shape: (n_docs, 1024)）
```

---

## Cell 8: Step1.5セクションヘッダ（Markdown）

```markdown
## Step1.5: 月次×分類別BERTopicクラスタリング（topic列の生成）
```

**説明**：月次ごと・分類ごとにBERTopicを実行してトピック番号を割り当てるステップを示すヘッダです。

---

## Cell 9: BERTopicクラスタリング（月次×分類別）

このセルは極めて複雑なため、部分ごとに説明します。

### 9-1: 日本語トークナイザの定義

```python
from bertopic import BERTopic
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from fugashi import Tagger

# fugashiベースの日本語トークナイザ（名詞連続を複合語として結合）
_tagger = Tagger()

# 汎用的でトピック弁別力のない頻出動詞（原形カタカナ）
STOP_VERBS = {
    'イル', 'スル', 'ナル', 'アル', 'ツク', 'オコナウ', 'ヨル',
    'イウ', 'ウケル', 'シメス', 'タイスル', 'ムケル', 'デキル',
    'ツヅク', 'ワカル', 'ノベル', 'ツタエル', 'イク', 'ハジマル',
    'ツヅケル', 'ツグ', 'オル', 'コエル', 'ダス', 'ハイル',
    'トル', 'モツ', 'ノボル', 'アワセル', 'フエル', 'アガル',
    'ミル', 'クル', 'オク', 'モラウ',
}

def japanese_tokenizer(text):
    """名詞連続を複合語として結合 + 汎用動詞以外の動詞も抽出（2文字以上）"""
    tokens = []
    noun_buf = []  # 名詞の連続バッファ

    for word in _tagger(text):
        pos = word.feature[0]
        surface = word.surface

        if pos == '名詞' and len(surface) >= 1:
            noun_buf.append(surface)
        else:
            # 名詞の連続が途切れたらバッファをフラッシュ
            if noun_buf:
                compound = ''.join(noun_buf)
                if len(compound) >= 2:
                    tokens.append(compound)
                noun_buf = []
            # 動詞: 頻出汎用動詞以外を原形で追加
            if pos == '動詞' and len(surface) >= 2:
                lemma = word.feature[6] if len(word.feature) > 6 and word.feature[6] != '*' else surface
                if lemma not in STOP_VERBS and len(lemma) >= 2:
                    tokens.append(surface)

    # 末尾バッファのフラッシュ
    if noun_buf:
        compound = ''.join(noun_buf)
        if len(compound) >= 2:
            tokens.append(compound)

    return tokens

# トークナイザのテスト
test_texts = [
    "イスラエル軍がガザ地区への地上侵攻を開始した",
    "千葉大学医学部附属病院の肝胆膵外科に所属",
    "岸田首相が自民党の政治資金問題について会見",
]
print("=== トークナイザテスト ===")
for t in test_texts:
    print(f"  {t}")
    print(f"  → {japanese_tokenizer(t)}")
    print()
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 1-4 | ライブラリインポート | BERTopic：トピックモデリング、HDBSCAN：クラスタリングアルゴリズム、CountVectorizer：BoW特徴化、UMAP：次元削減、Tagger：日本語形態素解析 |
| 7 | `_tagger = Tagger()` | fugashi（日本語形態素解析）のタガーをグローバル初期化。後で関数内で利用 |
| 10-24 | `STOP_VERBS` | トピック生成に無用な一般動詞を定義。「いる」「する」「なる」など、どのテキストにも出現する汎用動詞はトピック特性を表現しないため除外 |
| 26-50 | `japanese_tokenizer関数` | 引数：text（日本語テキスト）、戻り値：トークンリスト（['複合名詞1', '動詞1', ...]）。処理フロー：①形態素解析で単語ごとにPOS（品詞）を取得、②名詞は連続させて複合語を形成、③名詞以外（名詞の連続が途切れた時点）で複合語をバッファからフラッシュ、④汎用動詞以外の動詞を2文字以上で抽出、⑤末尾の名詞バッファをフラッシュ |
| 34-39 | 名詞バッファ処理 | 「東京」「駅」→「東京駅」と複合語化。単語1文字なら除外（ノイズ減少） |
| 41-45 | 動詞処理 | STOP_VERBSに入っていない動詞のみを抽出。lemma（原形）を参照（ある場合）。2文字以上の条件で短い動詞を除外 |
| 53-60 | トークナイザテスト | 3つのテスト文を実行して動作を確認。例期待値：  `['イスラエル軍', 'ガザ地区', '地上侵攻', '開始']` |

**具体例**：
```
入力: "イスラエル軍がガザ地区への地上侵攻を開始した"
形態素解析:
  イスラエル（名）→ バッファ=['イスラエル']
  軍（名）        → バッファ=['イスラエル','軍']
  が（助詞）      → 名詞途切れ → '複合イスラエル軍'をtokensに追加
  ガザ（名）      → バッファ=['ガザ']
  地区（名）      → バッファ=['ガザ','地区']
  へ（助詞）      → 名詞途切れ → '複合ガザ地区'をtokensに追加
  ...
  開始（動詞）    → STOP_VERBSに無い → tokensに追加
  した（動詞）    → STOP_VERBSに無い → tokensに追加

出力: ['イスラエル軍', 'ガザ地区', '地上侵攻', '開始', ...]
```

### 9-2: 月次×分類別BERTopicクラスタリング

```python
# 月次×分類別にBERTopicクラスタリングを実行し、topic列を生成
df_target['topic'] = -1
df_target['topic_prob'] = 0.0
df_target['topic_representative_words'] = ''

for (cat, month), group in df_target.groupby(['分類', '番組開始年月']):
    print(f"\n--- {cat} / {month} (n={len(group)}) ---")

    # text_combinedが空でない行のみ抽出
    has_text = group['text_combined'].fillna('').str.strip().astype(bool)
    sub = group[has_text]
    n_empty = len(group) - len(sub)
    if n_empty > 0:
        print(f"  テキスト空: {n_empty}件除外 → 対象: {len(sub)}件")

    if len(sub) < 10:
        print("  データ少 → スキップ")
        continue

    # このグループのembeddingを取得
    idxs = sub['embedding_idx'].values
    group_embeddings = embeddings[idxs]
    docs = sub['text_combined'].tolist()

    # UMAP次元削減（高次元embeddingのノイズ除去）
    n_neighbors = min(15, len(sub) - 1)
    umap_model = UMAP(
        n_components=min(50, len(sub) - 2),
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
    )

    # HDBSCANの設定
    min_cs = max(10, len(sub) // 100)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cs,
        min_samples=5,
        metric='euclidean',
    )

    # 日本語用のCountVectorizer（複合名詞対応）
    vectorizer = CountVectorizer(
        tokenizer=japanese_tokenizer,
        ngram_range=(1, 3),
        max_features=10000,
    )

    # BERTopicの実行（UMAPを明示的に指定）
    topic_model = BERTopic(
        language="japanese",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        verbose=False,
    )

    try:
        topics, probs = topic_model.fit_transform(docs, group_embeddings)
    except ValueError as e:
        print(f"  BERTopicエラー: {e} → スキップ")
        continue

    # 代表語の取得（複合名詞がそのまま出る）
    topic_info = topic_model.get_topic_info()
    topic_words_map = {}
    for _, row in topic_info.iterrows():
        tid = row['Topic']
        if tid == -1:
            topic_words_map[tid] = f"{tid}_outlier"
        else:
            words = topic_model.get_topic(tid)
            top_words = '_'.join([w for w, _ in words[:4]])
            topic_words_map[tid] = f"{tid}_{top_words}"

    # 結果を格納
    df_target.loc[sub.index, 'topic'] = topics
    df_target.loc[sub.index, 'topic_prob'] = probs
    df_target.loc[sub.index, 'topic_representative_words'] = [topic_words_map.get(t, '') for t in topics]

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    outlier_pct = n_outliers / len(sub) * 100
    print(f"  トピック数: {n_topics}, アウトライア: {n_outliers}/{len(sub)} ({outlier_pct:.1f}%)")

print(f"\n=== クラスタリング完了 ===")
print(f"topic列のユニーク値例: {df_target['topic'].value_counts().head(10)}")
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 64-66 | 初期化 | df_targetに3つの列を追加。topic=-1（未割当）、topic_prob=0.0、topic_representative_words=''で初期化 |
| 68 | `for (cat, month), group in df_target.groupby(['分類', '番組開始年月'])` | 分類と年月の組み合わせでグループ化。例：('政治・国際', '2025-03')ごとにループ |
| 74-76 | テキスト空判定 | text_combinedが空の行を除外。embeddingが存在しない行は処理不可のため除去 |
| 78-80 | データ不足チェック | 10件未満のグループはクラスタリング不可のためスキップ |
| 83-85 | グループのembeddingを取得 | embedding_idxを使って該当グループのembeddings行を抽出。group_embeddings shape: (グループサイズ, 1024) |
| 88-94 | UMAP設定 | embedding高次元ノイズを除去し、クラスタリング用に50次元以下に削減。n_neighbors=15で近傍グラフを構築（小さいグループはmin()で調整）。metric='cosine'でコサイン距離を使用 |
| 96-102 | HDBSCAN設定 | 密度ベースのクラスタリング。min_cluster_size は「クラスタと見なす最小要素数」（グループサイズの1%、最小10）。min_samples=5 でノイズ判定の閾値 |
| 104-109 | CountVectorizer設定 | BORTopic用のBag-of-Words特徴化。tokenizer=japanese_tokenizer で複合名詞対応。ngram_range=(1,3) で1～3グラムを抽出。max_features=10000 で上位10000特徴を保持 |
| 111-116 | BERTopic実行 | embedding（768/1024次元）+ UMAP（50次元）+ HDBSCAN + CountVectorizer を組み合わせてトピック抽出。topics：各ドキュメントのトピックID、probs：確信度 |
| 117-125 | 代表語取得 | get_topic_info()で全トピックのメタ情報取得。tid=-1（アウトライア）は'_outlier'とマーク。その他は上位4キーワードを連結（例：'3_イスラエル軍_ガザ_地上侵攻_被害'） |
| 127-129 | 結果をdf_targetに格納 | sub.indexの行に対して、topic/topic_prob/topic_representative_wordsを割り当て |
| 131-134 | 統計出力 | トピック数（アウトライア除外）、アウトライア数・割合を表示。例：「トピック数: 5, アウトライア: 12/100 (12.0%)」なら、正常（12%程度が許容範囲） |

**データフロー**：
```
入力: df_target（テキスト + embedding_idx列を含む）, embeddings配列
  ↓
月次×分類ごとにグループ分割
  ↓
各グループについて：
  1. embeddingを取得 → UMAP で次元削減
  2. HDBSCANでクラスタリング
  3. CountVectorizer + 日本語トークナイザでトピック抽出
  4. BERTopicで統合 → topics（ドキュメント→トピックID）、probs（確信度）
  ↓
出力: df_target（新列topic/topic_prob/topic_representative_wordsを含む）
```

**具体例**：
```
グループ: 社会 / 2025-03（50件のドキュメント）
→ UMAP: (50, 1024) → (50, 40)
→ HDBSCAN + CountVectorizer: 3つのトピックを発見
→ topic=[0, 0, 1, 1, 2, 2, ...], probs=[0.95, 0.88, 0.92, ...]
→ topic_representative_words=['0_火災_建物_延焼_消防', '1_盗難_窃盗_逮捕_容疑', '2_水害_浸水_対策_復旧']
```

### 反例：distance_thresholdを0.01にした場合

結果は Cell 12 の年間クラスタリングで効果が現れます。値を小さくすると：
- 月次トピックが細かく分割される
- 同じテーマでも月によって別トピックと見なされる可能性
- 年間クラスタリング時により多くのクラスタが生成される

---

## Cell 10: 月次トピックの重心（代表ベクトル）抽出

```python
# 分類×月×topicごとに代表ベクトル（重心）とメタ情報を集計
# topic=-1（アウトライア）は年間クラスタリングの対象外とする
group_cols = ['分類', '番組開始年月', 'topic']
cluster_records = []

for (cat, month, topic_id), group in df_target.groupby(group_cols):
    if topic_id == -1:
        continue

    idxs = group['embedding_idx'].values
    group_embs = embeddings[idxs]
    centroid = group_embs.mean(axis=0)

    # クラスタ内分散（centroidからの平均コサイン距離）
    if len(group_embs) > 1:
        dists = cosine_distances(group_embs, centroid.reshape(1, -1)).flatten()
        variance = dists.mean()
    else:
        variance = 0.0

    record = {
        '分類': cat,
        '番組開始年月': month,
        'topic': int(topic_id),
        'topic_representative_words': group['topic_representative_words'].iloc[0],
        'doc_count': len(group),
        'total_airtime': group['シーン放送時間_sec'].sum(),
        'mean_topic_prob': group['topic_prob'].astype(float).mean(),
        'centroid': centroid,
        'centroid_variance': variance,
    }
    cluster_records.append(record)

df_clusters = pd.DataFrame(cluster_records)
print(f"代表ベクトル数: {len(df_clusters)}（topic=-1を除外済み）")
for cat in TARGET_CATEGORIES:
    n = len(df_clusters[df_clusters['分類'] == cat])
    print(f"  {cat}: {n}個")
print(f"\n分散の統計量:")
print(df_clusters['centroid_variance'].describe())
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 4-5 | 初期化 | group_cols：グループ化キー、cluster_records：記録用リスト |
| 7 | `for (cat, month, topic_id), group` | 分類×月×トピックIDで3段階グループ化。例：('社会', '2025-03', 2)ごとに処理 |
| 8-9 | アウトライア除外 | topic_id==-1（HDBSCAN時のノイズ）は除外。年間クラスタリング対象外 |
| 12-13 | 重心（centroid）計算 | グループのembeddingを平均。shape: (1024,)。各次元の平均値を計算 |
| 16-19 | クラスタ内分散 | centroidからの平均コサイン距離を計算。variance が小さい→グループ内の同質性が高い、値が大きい→散在している |
| 16行 | `cosine_distances(group_embs, centroid.reshape(1, -1))` | (n_docs, 1024) と (1, 1024) の距離行列。結果は(n_docs, 1) |
| 16行 | `.flatten()` | (n_docs, 1) を (n_docs,) の1次元配列に変換 |
| 22-30 | record辞書 | 分類、月、トピックID、代表語、ドキュメント数、合計放送時間、平均確信度、重心、分散を格納 |
| 32 | `df_clusters = pd.DataFrame(cluster_records)` | 記録をDataFrameに変換。1行 = 1つの月次×分類×トピック の重心ベクトル |
| 33-39 | 統計出力 | 代表ベクトル総数と分類別内訳、分散の統計量（min/25%/50%/75%/max）を表示 |

**データフロー**：
```
入力: df_target（topic列を含む）, embeddings配列
  ↓
topic=-1（アウトライア）を除外
  ↓
分類×月×トピックごとにembeddingを集計：
  - 重心（平均） centroid
  - クラスタ内分散 variance
  - メタ情報（doc_count, total_airtime等）
  ↓
出力: df_clusters（各行は1つの月次トピック、centroid列を含む）
```

**具体例**：
```
分類: 社会
月: 2025-03
topic_id: 2
グループサイズ: 10件
  → 10個のembedding（各1024次元）の平均 → centroid（1024次元）
  → centroidからの平均距離 → variance = 0.15
```

---

## Cell 11: Step2セクションヘッダ（Markdown）

```markdown
## Step2: 代表ベクトルの年間クラスタリング（分類ごと）
```

**説明**：以下の処理ステップを示すセクションヘッダです。

---

## Cell 12: AgglomerativeClusteringによる年間クラスタリング

このセルは極めて複雑なため、部分に分割します。

### 12-1: 距離分布の分析と閾値決定

```python
df_clusters['annual_topic_id'] = -1

thresholds_to_test = np.arange(0.02, 0.21, 0.01)

for cat in TARGET_CATEGORIES:
    mask = df_clusters['分類'] == cat
    cat_data = df_clusters[mask]

    if len(cat_data) < 2:
        print(f"{cat}: データ不足、スキップ")
        continue

    X = np.stack(cat_data['centroid'].values)

    # --- (D) クラスタ分散・サイズを考慮した距離行列 ---
    base_dist = cosine_distances(X)

    variances = cat_data['centroid_variance'].values
    sizes = cat_data['doc_count'].values

    # 分散が大きいクラスタ同士は統合しやすく（距離を縮小）
    var_matrix = np.add.outer(variances, variances) / 2
    alpha = 0.1
    var_norm = var_matrix / (var_matrix.max() + 1e-10)
    adjusted_dist = base_dist * (1 - alpha * var_norm)
    np.fill_diagonal(adjusted_dist, 0)

    upper = base_dist[np.triu_indices_from(base_dist, k=1)]

    # --- 距離分布の統計量 ---
    print(f"\n{'='*50}")
    print(f"--- {cat} ---")
    print(f"  距離: min={upper.min():.4f}, median={np.median(upper):.4f}, "
          f"mean={upper.mean():.4f}, max={upper.max():.4f}")
    for p in [5, 10, 15, 20, 25]:
        print(f"  {p}パーセンタイル: {np.percentile(upper, p):.4f}")

    # --- 閾値ごとのクラスタ数 ---
    n_clusters_list = []
    for thresh in thresholds_to_test:
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thresh,
            metric='precomputed',
            linkage='average',
        ).fit_predict(adjusted_dist)
        n_clusters_list.append(len(set(labels)))

    # --- 本番クラスタリング ---
    threshold = 0.08
    final_labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='precomputed',
        linkage='average',
    ).fit_predict(adjusted_dist)
    df_clusters.loc[cat_data.index, 'annual_topic_id'] = final_labels
    print(f"  → threshold={threshold} で {len(set(final_labels))}個の年間トピックに統合"
          f" (元: {len(cat_data)}個)")

    # --- プロット: 左にヒストグラム、右にエルボー ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.hist(upper, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(threshold, color='red', linestyle='--', label=f'採用: {threshold}')
    ax1.set_title(f'{cat} - コサイン距離分布')
    ax1.set_xlabel('cosine distance')
    ax1.legend()

    ax2.plot(thresholds_to_test, n_clusters_list, marker='o')
    ax2.axvline(threshold, color='red', linestyle='--', label=f'採用: {threshold}')
    ax2.set_xlabel('distance_threshold')
    ax2.set_ylabel('クラスタ数')
    ax2.set_title(f'{cat} - threshold vs クラスタ数 (average)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 1 | `df_clusters['annual_topic_id'] = -1` | 初期化。年間トピックIDを-1で埋める |
| 3 | `thresholds_to_test = np.arange(0.02, 0.21, 0.01)` | テスト対象の距離閾値。0.02, 0.03, ..., 0.20 の19パターン |
| 5 | `for cat in TARGET_CATEGORIES` | 各分類（'政治・国際', '社会'）ごとにループ |
| 8-9 | データ不足チェック | 2個未満のクラスタはクラスタリング不可 |
| 11 | `X = np.stack(cat_data['centroid'].values)` | 分類別の全月次トピックの重心を積み上げ。shape: (n_clusters, 1024) |
| 14 | `base_dist = cosine_distances(X)` | 重心間のコサイン距離行列。shape: (n_clusters, n_clusters)。例：値0.1なら「セマンティックに似ている」 |
| 16-17 | 分散・サイズを取得 | variance：各月次トピックのばらつき度合い、sizes：ドキュメント数 |
| 20-23 | 分散を考慮した距離調整 | var_matrix：2つのクラスタの分散の平均。分散大なら「ばらつきが大きい→統合しやすく」という論理で距離を縮小（adjusted_dist = base_dist * (1 - alpha * var_norm)）。alpha=0.1で調整強度を制御 |
| 24 | `np.fill_diagonal(adjusted_dist, 0)` | 対角線（自分自身との距離）を0に |
| 26 | `upper = base_dist[np.triu_indices_from(...)]` | 距離行列の上三角部分を抽出（重複排除）。以後の統計計算用 |
| 29-33 | 距離分布統計 | 最小値、中央値、平均値、最大値、パーセンタイル値を表示。閾値決定の根拠となる |
| 35-41 | 閾値ごとのクラスタ数シミュレーション | 19種類の閾値について、それぞれクラスタ数がいくつになるかを計算。エルボー図のデータ |
| 35行 | `AgglomerativeClustering(n_clusters=None, distance_threshold=thresh, ...)` | 引数説明：n_clusters=Noneで「距離がthreshを超えたら別クラスタ」という条件で動作、metric='precomputed'で事前計算の距離行列を使用、linkage='average'で平均リンケージ（クラスタ間平均距離）を採用 |
| 43-48 | 本番クラスタリング | threshold=0.08で確定実行。結果をfinal_labels に格納（0, 1, 2, ...）。df_clustersのannual_topic_idを更新 |
| 51-62 | グラフ描画 | 左図：コサイン距離ヒストグラム（red縦線が採用閾値）、右図：エルボー曲線（閾値 vs クラスタ数） |

**データフロー**：
```
入力: df_clusters（各行は月次トピック、centroid列を含む）
  ↓
分類ごとにフィルタ
  ↓
1. 重心間の距離行列を計算（コサイン距離）
2. 分散を考慮して距離を調整
3. 19種類の閾値でシミュレーション
4. 距離分布の統計を確認 → 最適な閾値を決定（0.08）
5. 確定した閾値でAgglomerativeClustering実行
  ↓
出力: df_clusters（新列annual_topic_idを含む）
```

**具体例**：
```
分類: 社会
月次トピック数: 20個
  → 各月次トピックの重心（各1024次元）を計算
  → 20×20の距離行列を生成
  → 距離分布: min=0.02, median=0.12, max=0.35
  → threshold=0.08で実行 → 5個の年間トピックに統合
```

**反例：thresholdを0.15に変更した場合**
- より多くの月次トピックが統合される
- 年間トピック数が減少（例：20個 → 3個）
- テーマの粒度が粗くなる（「社会全般」など）
- 月ごとの違いが見えにくくなる

---

## Cell 13: 空のセル（マークダウン）

```markdown

```

**説明**：マークダウン形式の空セル。レイアウト調整用です。

---

## Cell 14: Step3セクションヘッダ（Markdown）

```markdown
## Step3: 結果の整理と可視化
```

**説明**：以下の処理ステップを示すセクションヘッダです。

---

## Cell 15: 結果をCSVに出力

```python
# 出力用DataFrame（centroid列を除外）
df_result = df_clusters.drop(columns=['centroid']).sort_values(
    ['分類', 'annual_topic_id', '番組開始年月']
).reset_index(drop=True)

# CSV出力
output_csv = os.path.join(OUTPUT_PATH, 'annual_topic_tracking.csv')
df_result.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"CSV出力完了: {output_csv}")
df_result.head(20)
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 2-4 | DataFrameの準備 | df_clustersから centroid列を除外（ファイルサイズ削減）。分類・annual_topic_id・番組開始年月でソート |
| 4 | `.reset_index(drop=True)` | ソート後のインデックスをリセット（0, 1, 2, ... になる） |
| 7 | `os.path.join(OUTPUT_PATH, 'annual_topic_tracking.csv')` | 出力ファイルパスを組み立て（→ Cell 1で定義したOUTPUT_PATH） |
| 8 | `df_result.to_csv(..., encoding='utf-8-sig')` | DataFrameをCSVに出力。encoding='utf-8-sig'で UTF-8（BOM付き）で保存（Excel互換性） |
| 9 | print文 | 出力確認メッセージ |
| 10 | `df_result.head(20)` | 先頭20行を表示確認 |

**出力CSVの列**：
- 分類: 政治・国際 or 社会
- 番組開始年月: 2025-01, 2025-02, ...
- topic: 月次トピックID（BERTopicで自動採番）
- topic_representative_words: 代表キーワード
- doc_count: そのトピックのドキュメント数
- total_airtime: 合計放送時間（秒）
- mean_topic_prob: 平均確信度
- centroid_variance: クラスタ内分散
- annual_topic_id: 年間トピックID

---

## Cell 16: 年間トピックの代表キーワード集約と規模統計

```python
# 各年間トピックの代表キーワード集約
def aggregate_keywords(group):
    """配下の月次トピックのキーワードを集約"""
    all_words = []
    for words_str in group['topic_representative_words'].dropna():
        # 'topicID_word1_word2_...' 形式からキーワードを抽出
        parts = str(words_str).split('_')
        if len(parts) > 1:
            all_words.extend(parts[1:])  # topicIDの部分を除外
    # 頻度順に上位キーワードを返す
    from collections import Counter
    counter = Counter(all_words)
    top_words = [w for w, _ in counter.most_common(10)]
    return ', '.join(top_words)

keyword_summary = df_result.groupby(['分類', 'annual_topic_id']).apply(
    aggregate_keywords
).reset_index(name='aggregated_keywords')

# 年間トピックごとの合計doc_count（規模の目安）
size_summary = df_result.groupby(['分類', 'annual_topic_id']).agg(
    total_doc_count=('doc_count', 'sum'),
    total_airtime_sum=('total_airtime', 'sum'),
    months_active=('番組開始年月', 'nunique'),
).reset_index()

topic_summary = keyword_summary.merge(size_summary, on=['分類', 'annual_topic_id'])
topic_summary = topic_summary.sort_values(['分類', 'total_doc_count'], ascending=[True, False])

print("=== 年間トピック一覧（分類別・規模順） ===")
for cat in TARGET_CATEGORIES:
    cat_data = topic_summary[topic_summary['分類'] == cat]
    print(f"\n--- {cat} ({len(cat_data)}トピック) ---")
    for _, row in cat_data.head(20).iterrows():
        print(f"  annual_topic={row['annual_topic_id']:3d} | "
              f"docs={row['total_doc_count']:5d} | "
              f"months={row['months_active']:2d} | "
              f"{row['aggregated_keywords']}")
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 3-14 | `aggregate_keywords関数` | 引数：group（同一年間トピックの全月次トピック行）、戻り値：上位10キーワードの文字列（カンマ区切り）。処理：topic_representative_words（'0_イスラエル軍_ガザ_...'形式）から キーワード部分を抽出、全月次トピックから集約、Counter で頻度順にソート |
| 15-17 | `keyword_summary` | 年間トピックごと（分類×annual_topic_id）にキーワード集約を実行 |
| 19-25 | `size_summary` | 年間トピックごとに：total_doc_count（全月次トピックのdoc_count合計）、total_airtime_sum（全月次トピックの放送時間合計）、months_active（活動月数）を集計 |
| 27-28 | マージ・ソート | keyword_summaryとsize_summaryを分類×annual_topic_idでマージ。規模順にソート（降順） |
| 30-35 | 出力 | 分類ごとに上位20年間トピックを表示。annual_topic_id, ドキュメント数、活動月数、集約キーワードを表示 |

**データフロー**：
```
入力: df_result（月次トピック×分類×年月の集計DataFrame）
  ↓
1. 年間トピックごとにキーワード集約
   topic_representative_words から topicID を除外、全月次トピックのキーワードを合算
  ↓
2. 年間トピックごとに規模統計
   doc_count合計、airtime合計、活動月数を集計
  ↓
3. マージ・ソート
   分類×annual_topic_idでキーワード×規模を統合
   total_doc_countで降順ソート
  ↓
出力: topic_summary（1行＝1つの年間トピック、規模順）
```

**具体例**：
```
年間トピック: 社会 - 5
  月次トピックA(01月): 火災, 建物, 延焼, 消防
  月次トピックB(02月): 火災, 対策, 予防, 管理
  月次トピックC(05月): 火災, 事故, 被害, 対応
  → aggregated_keywords: 火災 (3回), 対策 (1回), 被害 (1回), ...
  → 出力: "火災, 対策, 被害, 建物, ..."
  → total_doc_count: 150, months_active: 3
```

---

## Cell 17: 月別推移グラフの描画

```python
# 月別の推移を折れ線グラフでプロット
# 年間トピックごとに月別のdoc_count/total_airtimeを集計
months_order = sorted(df_result['番組開始年月'].unique())

def plot_trends(cat, metric, ylabel, title_suffix):
    cat_data = df_result[df_result['分類'] == cat]

    # 年間トピックごとに月別集計
    pivot = cat_data.groupby(['annual_topic_id', '番組開始年月'])[metric].sum().unstack(fill_value=0)
    pivot = pivot.reindex(columns=months_order, fill_value=0)

    # 規模上位のトピックのみプロット（見やすさのため）
    top_topics = pivot.sum(axis=1).nlargest(20).index
    pivot_top = pivot.loc[top_topics]

    fig, ax = plt.subplots(figsize=(14, 6))
    for topic_id in pivot_top.index:
        # そのトピックのキーワードをラベルに
        kw = topic_summary[
            (topic_summary['分類'] == cat) &
            (topic_summary['annual_topic_id'] == topic_id)
        ]['aggregated_keywords'].values
        label = f"T{topic_id}: {kw[0][:30]}" if len(kw) > 0 else f"T{topic_id}"
        ax.plot(pivot_top.columns, pivot_top.loc[topic_id], marker='o', label=label)

    ax.set_xlabel('月')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{cat} - {title_suffix}（上位10トピック）')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

for cat in TARGET_CATEGORIES:
    plot_trends(cat, 'doc_count', 'シーン数', '月別シーン数の推移')
    plot_trends(cat, 'total_airtime', '放送時間(秒)', '月別放送時間の推移')
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 4 | `months_order = sorted(df_result['番組開始年月'].unique())` | 年月をソート（例：['2025-01', '2025-02', ...]）。後でX軸を順序正しく表示するため |
| 6 | `def plot_trends(cat, metric, ylabel, title_suffix)` | 引数：cat（分類）、metric（'doc_count' or 'total_airtime'）、ylabel（Y軸ラベル）、title_suffix（グラフタイトル）。戻り値なし（グラフ表示） |
| 7 | `cat_data = df_result[df_result['分類'] == cat]` | 指定分類のデータのみを抽出 |
| 10 | `pivot = cat_data.groupby(['annual_topic_id', '番組開始年月'])[metric].sum().unstack(fill_value=0)` | 年間トピック×年月のクロス表を作成（行＝年間トピックID、列＝年月、値＝metricの合計）。fill_value=0で欠損値を0埋め |
| 11 | `pivot.reindex(columns=months_order, fill_value=0)` | months_order の順序で列を並び替え。存在しない月は0埋め |
| 14 | `top_topics = pivot.sum(axis=1).nlargest(20).index` | 年別集計値が大きい上位20年間トピックのIDを取得 |
| 17-27 | グラフ描画ループ | top_topicsの各トピックについてX（月）-Y（doc_count/airtime）の折れ線を描画。ラベルにはtopic_summaryからキーワード集約を取得して「T[ID]: [キーワード]」形式で表示 |
| 24-25 | キーワード取得 | topic_summaryから該当する年間トピックのキーワードを検索。存在すれば「T0: 火災, 建物」、なければ「T0」のみ表示 |
| 29-30 | グラフ装飾 | X軸ラベル'月'、Y軸ラベルはmetricに応じて動的、凡例を右側に配置（fontsize=8で見やすく）、X軸ラベルを45度回転（月の文字が重ならない） |
| 33-34 | メイン処理 | 各分類について、doc_count と airtime の2種類のグラフを描画（計4グラフ） |

**データフロー**：
```
入力: df_result（月次トピック×分類×年月の集計DataFrame）, topic_summary（年間トピックのキーワード）
  ↓
分類・メトリクス（doc_count/airtime）ごとにループ
  ↓
1. 分類でフィルタ
2. annual_topic_id × 番組開始年月 のクロス表を作成
3. 規模上位20トピックのみ抽出
4. 各トピックについて月別の値を折れ線で描画
  ↓
出力: グラフ（4枚：社会×doc_count, 社会×airtime, 政治・国際×..., ...）
```

**具体例**：
```
社会 / doc_count
   2025-01: T5(火災) 45件, T10(盗難) 32件, ...
   2025-02: T5(火災) 52件, T10(盗難) 28件, ...
   2025-03: T5(火災) 38件, T10(盗難) 35件, ...
   → T5の折れ線: 45 → 52 → 38
   → T10の折れ線: 32 → 28 → 35
```

---

## Cell 18: 元のDataFrameに年間トピック情報を追加

```python
# dfにtopic列・topic_prob列・annual_topic_id列を追加
# df_targetの結果をdfのインデックス経由でマージ

# df_targetからtopic関連列を取得
topic_cols = df_target[['topic', 'topic_prob', 'topic_representative_words']].copy()

# annual_topic_idはdf_clustersから月次トピック経由で紐付け
annual_map = df_clusters[['分類', '番組開始年月', 'topic', 'annual_topic_id']].copy()
df_target_with_annual = df_target[['分類', '番組開始年月', 'topic']].merge(
    annual_map, on=['分類', '番組開始年月', 'topic'], how='left'
)
topic_cols['annual_topic_id'] = df_target_with_annual['annual_topic_id'].values

# dfにマージ（df_targetに含まれない行はNaN）
for col in ['topic', 'topic_prob', 'topic_representative_words', 'annual_topic_id']:
    df[col] = np.nan

df.loc[topic_cols.index, 'topic'] = topic_cols['topic']
df.loc[topic_cols.index, 'topic_prob'] = topic_cols['topic_prob']
df.loc[topic_cols.index, 'topic_representative_words'] = topic_cols['topic_representative_words']
df.loc[topic_cols.index, 'annual_topic_id'] = topic_cols['annual_topic_id']

# 確認
print(f"df shape: {df.shape}")
print(f"topic非NaN: {df['topic'].notna().sum()} / {len(df)}")
print(f"annual_topic_id非NaN: {df['annual_topic_id'].notna().sum()} / {len(df)}")
print(f"\ntopic列の分布:")
print(df['topic'].value_counts().head(10))
print(f"\nannual_topic_id列の分布:")
print(df['annual_topic_id'].value_counts().head(10))
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 4 | `topic_cols = df_target[['topic', 'topic_prob', 'topic_representative_words']].copy()` | df_targetから既に計算されたトピック関連列を抽出 |
| 7 | `annual_map = df_clusters[['分類', '番組開始年月', 'topic', 'annual_topic_id']].copy()` | df_clustersから年間トピックのマッピング情報を抽出（月次トピックID → 年間トピックID） |
| 8-11 | annual_topic_id の紐付け | df_target（原始データ）から分類×月×月次トピックを取得し、annual_mapとマージ。左結合で該当する年間トピックIDを追加 |
| 12 | `topic_cols['annual_topic_id']` | マージ結果をtopic_colsに追加 |
| 14-17 | 初期化 | dfに4つの新列を追加、全てNaNで埋める |
| 19-22 | 値の割り当て | df_targetに対応するdf の行（topic_cols.index）に対して、トピック情報を代入。df_targetに含まれない行（TARGET_CATEGORIESに属さない分類）はNaNのまま |
| 25-30 | 確認 | df全体のサイズ、非NaN値の件数、topic/annual_topic_id の分布を表示 |

**データフロー**：
```
入力: df（全データ）, df_target（TARGET_CATEGORIESでフィルタ済み）, df_clusters（月次トピック×年間トピック）
  ↓
1. df_targetのトピック情報を抽出
2. df_clustersから月次→年間トピックマッピングを取得
3. df_targetをマッピングとマージして annual_topic_id を追加
4. dfに4つの新列を初期化・代入
  ↓
出力: df（新列topic/topic_prob/topic_representative_words/annual_topic_idを含む）
```

**注意**：
- df_targetに含まれない行（例：「暮らし」分類、TARGET_CATEGORIESに未含）はtopic=NaN のまま
- df_targetに含まれるが topic_id=-1（アウトライア）の行も topic=-1, annual_topic_id=NaN で反映される

---

## Cell 19: ネガティブ情報フラグセクションヘッダ（Markdown）

```markdown
## Step 4: ネガティブ情報フラグの付与
```

**説明**：続くセルの処理内容を示すセクションヘッダです。

---

## Cell 20: ネガティブキーワード辞書の定義と試験マッチ

```python
# Step 4-2: ネガティブキーワード辞書（確定版v3）+ マッチ件数確認
import re

# 対象分類を限定（ドラマ・バラエティ等の偽陽性を排除）
TARGET_NEG = ["政治・国際", "社会"]
df_neg_target = df[df["分類"].isin(TARGET_NEG)]
headline = df_neg_target["ヘッドライン"].fillna("")

# === カテゴリ別キーワード辞書 ===
NEG_KEYWORDS = {
    "事件・犯罪": ["殺人", "刺殺", "強盗", "詐欺", "逮捕", "容疑", "犯行", "暴行", "窃盗",
                  "放火", "誘拐", "恐喝", "横領", "傷害", "ひき逃げ", "虐待", "わいせつ",
                  "盗撮", "不正", "脅迫", "飲酒運転", "闇バイト", "性犯罪"],
    "薬物": ["薬物", "覚醒剤", "大麻"],
    "事故": ["事故", "墜落", "火災", "崩落", "転落"],
    "死亡・被害": ["死亡", "死者", "犠牲", "遺体", "負傷", "行方不明", "溺れ", "自殺",
                  "重傷", "重体"],
    "災害": ["地震", "津波", "洪水", "噴火", "土砂崩れ", "土砂災害", "豪雨", "暴風"],
    "紛争・武力": ["侵攻", "空爆", "砲撃", "爆撃", "戦闘", "攻撃", "銃撃", "襲撃",
                  "ミサイル", "武力", "暴動", "クーデター", "人質"],
    "テロ": ["自爆テロ", "テロ事件", "テロリスト", "テロ攻撃"],
    "爆発": ["爆発"],
    "汚染": ["汚染"],
}

# === 除外ワード ===
EXCLUDE_WORDS = ["防ぐ", "守る", "対策", "防止", "防災", "訓練", "爆発的"]

# マッチ
all_neg_keywords = []
for cat, words in NEG_KEYWORDS.items():
    all_neg_keywords.extend(words)

neg_pattern = "|".join(re.escape(w) for w in all_neg_keywords)
excl_pattern = "|".join(re.escape(w) for w in EXCLUDE_WORDS)

hit_neg = headline.str.contains(neg_pattern, na=False)
hit_excl = headline.str.contains(excl_pattern, na=False)
hit_final = hit_neg & ~hit_excl

print(f"対象データ（政治・国際 + 社会）: {len(df_neg_target):,}件")
print(f"ネガティブキーワードヒット: {hit_neg.sum():,}件 ({hit_neg.mean()*100:.1f}%)")
print(f"除外ワードで除外: {(hit_neg & hit_excl).sum():,}件")
print(f"最終ネガティブ判定: {hit_final.sum():,}件 ({hit_final.mean()*100:.1f}%)")

# カテゴリ別内訳
print("\n=== カテゴリ別ヒット件数 ===")
for cat, words in NEG_KEYWORDS.items():
    pat = "|".join(re.escape(w) for w in words)
    cat_hit = headline.str.contains(pat, na=False)
    cat_final = cat_hit & ~hit_excl
    print(f"  {cat:12s}: {cat_hit.sum():>6,}件 → 除外後 {cat_final.sum():>6,}件")

# 最終ネガティブ判定のランダムサンプル
final_neg = df_neg_target[hit_final]
print(f"\n=== 最終ネガティブ判定サンプル（ランダム15件） ===")
for _, row in final_neg.sample(n=min(30, len(final_neg)), random_state=42).iterrows():
    print(f"  [{row.get('分類','')}] {row['ヘッドライン'][:90]}")
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 4 | `TARGET_NEG = ["政治・国際", "社会"]` | ネガティブキーワード判定の対象分類。ドラマ・バラエティなど「創作」の分類は対象外 |
| 5-6 | フィルタ | TARGET_NEGに属する行のみを抽出して、ヘッドラインを取得 |
| 9-22 | `NEG_KEYWORDS` | 9つのカテゴリ別に、ネガティブニュアンスのキーワード群を定義。各キーワードは「その語が見出しに含まれれば、ニュースはネガティブ」という判定基準 |
| 25 | `EXCLUDE_WORDS` | キーワードはマッチするが、ネガティブではない場合を除外。「防ぐ」「対策」は「火災を防ぐ」など、ポジティブな文脈で使用されることが多いため除外 |
| 28-30 | パターン構築 | NEG_KEYWORDS と EXCLUDE_WORDS の全キーワードを「\|」（OR）で連結した正規表現パターンを生成。`re.escape()` で特殊文字をエスケープ |
| 32-34 | マッチング | headline にネガティブパターンと除外パターンがマッチするか判定。hit_neg=True AND hit_excl=False なら最終ネガティブ判定 |
| 36-39 | 統計出力 | 対象行数、ネガティブヒット数、除外数、最終判定数を表示（パーセンテージ付き） |
| 42-46 | カテゴリ別内訳 | 各カテゴリのキーワードについて、マッチ件数と除外後件数を表示。例：「事件・犯罪」は1234件マッチ → 除外後1200件のように、除外ワードで除去された件数を確認 |
| 49-51 | サンプル表示 | 最終ネガティブ判定の行からランダムに30件（最大）を表示。手作業で検証（正しく判定されているか確認） |

**データフロー**：
```
入力: df（全データ）
  ↓
TARGET_NEGの分類でフィルタ
  ↓
ヘッドラインに対して：
  1. ネガティブキーワード正規表現でマッチング
  2. 除外ワード正規表現でマッチング
  3. AND NOT で最終判定（ネガティブ=True AND 除外=False）
  ↓
出力: 統計情報（ヒット件数、除外数、最終判定数）、サンプル表示
```

**具体例**：
```
headline: "火災で建物延焼、対策急ぐ"
  → "火災"でhit_neg=True
  → "対策"でhit_excl=True
  → hit_final = True AND NOT True = False （ネガティブではない）

headline: "火災で10人負傷"
  → "火災"でhit_neg=True
  → "負傷"でhit_neg=True（重複だが真値）
  → "対策"「防ぐ」いずれもなし → hit_excl=False
  → hit_final = True AND NOT False = True （ネガティブと判定）
```

---

## Cell 21: is_negative列の追加（本番実行）

```python
# Step 4-3: dfにis_negative列を追加
import re

NEG_KEYWORDS = {
    "事件・犯罪": ["殺人", "刺殺", "強盗", "詐欺", "逮捕", "容疑", "犯行", "暴行", "窃盗",
                  "放火", "誘拐", "恐喝", "横領", "傷害", "ひき逃げ", "虐待", "わいせつ",
                  "盗撮", "不正", "脅迫", "飲酒運転", "闇バイト", "性犯罪"],
    "薬物": ["薬物", "覚醒剤", "大麻"],
    "事故": ["事故", "墜落", "火災", "崩落", "転落"],
    "死亡・被害": ["死亡", "死者", "犠牲", "遺体", "負傷", "行方不明", "溺れ", "自殺",
                  "重傷", "重体"],
    "災害": ["地震", "津波", "洪水", "噴火", "土砂崩れ", "土砂災害", "豪雨", "暴風"],
    "紛争・武力": ["侵攻", "空爆", "砲撃", "爆撃", "戦闘", "攻撃", "銃撃", "襲撃",
                  "ミサイル", "武力", "暴動", "クーデター", "人質"],
    "テロ": ["自爆テロ", "テロ事件", "テロリスト", "テロ攻撃"],
    "爆発": ["爆発"],
    "汚染": ["汚染"],
}
EXCLUDE_WORDS = ["防ぐ", "守る", "対策", "防止", "防災", "訓練", "爆発的"]
TARGET_NEG = ["政治・国際", "社会"]

all_neg_keywords = []
for words in NEG_KEYWORDS.values():
    all_neg_keywords.extend(words)
neg_pattern = "|".join(re.escape(w) for w in all_neg_keywords)
excl_pattern = "|".join(re.escape(w) for w in EXCLUDE_WORDS)

headline = df["ヘッドライン"].fillna("")
is_target = df["分類"].isin(TARGET_NEG)
hit_neg = headline.str.contains(neg_pattern, na=False)
hit_excl = headline.str.contains(excl_pattern, na=False)

df["is_negative"] = (is_target & hit_neg & ~hit_excl).astype(int)

print(f"is_negative=1: {df['is_negative'].sum():,}件 / {len(df):,}件 ({df['is_negative'].mean()*100:.1f}%)")
print(f"\n分類別:")
print(df.groupby("分類")["is_negative"].agg(["sum", "mean", "count"]).rename(
    columns={"sum": "neg件数", "mean": "neg率", "count": "全件"}).to_string())
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 3-22 | 設定の再定義 | Cell 20と同じ設定を再度定義。→セルの独立性確保（別セルから単独実行可能にするため） |
| 24-27 | パターン構築 | → Cell 20 と同じ。全キーワードのOR正規表現を生成 |
| 29-32 | df全体に対してマッチング | df（全データ）のヘッドラインに対してマッチング。is_targetで「対象分類かつネガティブキーワードマッチ」を判定 |
| 34 | `(is_target & hit_neg & ~hit_excl).astype(int)` | 3つの条件をAND/NOTで組み合わせ、Bool値をint（0 or 1）に変換。df["is_negative"]に格納 |
| 36-39 | 統計出力 | is_negative=1（ネガティブ）の件数・割合、分類別の詳細統計を表示 |

**データフロー**：
```
入力: df（全データ）
  ↓
全行に対して：
  1. 分類が TARGET_NEG（政治・国際 or 社会）か判定
  2. ヘッドラインにネガティブキーワード含まれるか判定
  3. ヘッドラインに除外ワード含まれるか判定
  4. is_negative = (1 AND 2 AND NOT 3)
  ↓
出力: df（新列is_negative、0/1の値）
```

---

## Cell 22: 最終データの保存

```python
# dfを保存
save_path = "/home/ciwai/work/data/0014_MTC_information_health/mdata"
os.makedirs(save_path, exist_ok=True)

output_file = os.path.join(save_path, "mdata_with_topics.parquet")
df.to_parquet(output_file, index=False, compression='snappy')
file_size_mb = os.path.getsize(output_file) / 2**20
print(f"保存完了: {output_file} ({file_size_mb:.1f}MB)")
print(f"行数: {len(df):,}, 列数: {len(df.columns)}")
```

**行ごと説明**：

| 行 | 内容 | 説明 |
|----|------|------|
| 2 | `save_path = ...` | 保存先ディレクトリを定義 |
| 3 | `os.makedirs(save_path, exist_ok=True)` | ディレクトリが存在しなければ作成。exist_ok=Trueで「既存時エラー」を回避 |
| 5 | `output_file = os.path.join(save_path, "mdata_with_topics.parquet")` | 出力ファイル名を組み立て（parquet形式） |
| 6 | `df.to_parquet(output_file, index=False, compression='snappy')` | DataFrameをParquet形式で保存。compression='snappy'で圧縮。index=Falseでインデックス列を除外 |
| 7-8 | ファイルサイズ計算 | os.path.getsize()でファイルサイズ取得、2^20（1048576）で割ってMB単位に変換 |
| 9 | print文 | 保存完了メッセージ、ファイルサイズ、行数、列数を表示 |

**データフロー**：
```
入力: df（トピック + is_negative列を含む最終DataFrame）
  ↓
ディレクトリ確保
  ↓
Parquet形式で圧縮保存
  ↓
出力: mdata_with_topics.parquet
```

**注意**：
- Parquet形式は CSV より圧縮率が高く、読み込み速度も高速
- index=False でインデックス列を除外（ストレージ削減）
- snappy圧縮で中程度の圧縮率とデコード速度のバランスを実現

---

## Cell 23: 空のセル

```python

```

**説明**：処理終了を示す空セルです。

---

## 全体データフローの総括

```
[入力：CSVファイル群]
  ↓
[Cell 3] データ読み込み・年月抽出
  ↓
[Cell 4] テキスト前処理（タグ除外、数字除去、ヘッドライン結合）
  ↓
[Cell 6-7] Embedding化（テキスト → 1024次元ベクトル）
  ↓
[Cell 9] 月次×分類別 BERTopic クラスタリング
    - 日本語トークナイザで複合名詞抽出
    - UMAP + HDBSCAN + CountVectorizer で月次トピック生成
    - topic列, topic_prob列, topic_representative_words列を生成
  ↓
[Cell 10] 月次トピック重心計算
    - embeddingの平均値（重心）= 代表ベクトル
    - 分散計算（クラスタ内のばらつき度合い）
  ↓
[Cell 12] 代表ベクトルの年間クラスタリング（分類ごと）
    - AgglomerativeClustering（階層的クラスタリング）
    - distance_threshold=0.08 で月次トピックを年間トピックに統合
    - annual_topic_id列を生成
  ↓
[Cell 15] df_result 出力（CSVファイル）
  ↓
[Cell 16] 年間トピックの集約キーワードと統計計算
  ↓
[Cell 17] 月別推移グラフの描画（4枚）
  ↓
[Cell 18] 元のdfに topic / annual_topic_id 列をマージ
  ↓
[Cell 20-21] ネガティブキーワード判定 → is_negative列
  ↓
[Cell 22] dfを Parquet形式で保存
  ↓
[出力：mdata_with_topics.parquet]
```

---

## 主要パラメータ一覧と調整ガイド

| パラメータ | 現在値 | 範囲 | 効果 |
|-----------|-------|------|------|
| `EMBEDDING_MODEL_NAME` | "cl-nagoya/ruri-large" | 日本語対応モデル | 値が高い → セマンティック精度UP ↔ 計算時間DOWN |
| `DISTANCE_THRESHOLDS` | {'社会': 0.1, ...} | 0.01～1.0 | 小さい → より細かく分かれる。大きい → 統合されやすい |
| `UMAP n_components` | min(50, グループサイズ-2) | 10～100 | 大きい → 情報保持UP ↔ 計算時間UP |
| `HDBSCAN min_cluster_size` | max(10, サイズ//100) | 5～20 | 小さい → より多くトピック生成。大きい → 統合されやすい |
| `distance_threshold (年間)` | 0.08 | 0.02～0.20 | 小さい → より細かく年間トピックを分割。大きい → 統合されやすい |

---

## 反例と応用例

### 反例1：distance_thresholdを0.20に増加
```
月次トピック: 20個
  ↓ threshold=0.20 で実行
年間トピック: 3個（非常に粗い粒度）

結果：全ての「政治系」が1つに統合される
  ↑ テーマの違いが見えず、推奨しない
```

### 反例2：HDBSCAN min_cluster_size を30に増加
```
グループサイズ: 100件
  ↓ min_cluster_size=30
月次トピック: 2個（多くの行がアウトライア判定）
  ↑ 情報の喪失が大きい

推奨：min(10, サイズ//100) の自動計算を使用
```

### 応用例：「暮らし」分類も分析対象に

```python
# Cell 1 を修正
TARGET_CATEGORIES = ['政治・国際', '社会', '暮らし']  # 追加
DISTANCE_THRESHOLDS['暮らし'] = 0.03  # より厳密（細かいテーマが多い）

# Cell 20 を修正
TARGET_NEG = ["政治・国際", "社会"]  # 「暮らし」は除外（偽陽性が多い）
```

---

## 最終出力物の説明

### annual_topic_tracking.csv
各行 = 1つの月次トピック
- `分類`: 政治・国際/社会
- `番組開始年月`: 2025-01, 2025-02, ...
- `topic`: 月次トピックID
- `topic_representative_words`: 代表キーワード（例：'2_イスラエル軍_ガザ_地上侵攻_被害'）
- `doc_count`: シーン数
- `total_airtime`: 合計放送時間（秒）
- `mean_topic_prob`: 平均確信度
- `centroid_variance`: クラスタ内分散
- `annual_topic_id`: 年間トピックID（統合後）

### mdata_with_topics.parquet
元のDataFrameに加えて：
- `topic`: 月次トピックID（TARGET_CATEGORIESのみ非NaN）
- `topic_prob`: 月次トピック確信度
- `topic_representative_words`: 代表キーワード
- `annual_topic_id`: 年間トピックID（TARGET_CATEGORIESのみ非NaN）
- `is_negative`: ネガティブニュースフラグ（0 or 1、政治・国際/社会のみ）

---

## トラブルシューティング

### 問題1：BERTopicがエラーで失敗する
```
原因: 月次グループのサイズが小さすぎる（10件未満）
対策: Cell 9 の n_neighbors や min_cluster_size を調整
```

### 問題2：アウトライア率が高い（>30%）
```
原因: HDBSCANのパラメータが厳密すぎる
対策: Cell 9 の min_cluster_size を小さくする
```

### 問題3：年間トピック数が多すぎる（>50個）
```
原因: distance_threshold が小さすぎる
対策: Cell 12 の threshold=0.08 を 0.10 ～ 0.15 に増加
```

