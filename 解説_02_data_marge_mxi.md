# ノートブック 02: データ統合（Mデータ × インテージデータ）コード解説

本ノートブックは、Mデータ（テレビ放映番組シーン）とインテージデータ（視聴ログ）を統合するプロセスを実装しています。メモリ効率とパフォーマンスを最優先に設計された月次ループ版です。

---

## Cell 0: ノートブック概要（マークダウン）

```markdown
# データ統合：Mデータ × インテージデータ（月次ループ版）
## - Mデータ（シーンテーブル）は全期間を1回だけ読み込み
## - インテージデータは1ヶ月ずつ読み込み→結合→CSV保存→メモリ解放
```

**説明：**
このマークダウンセルはノートブックの処理戦略を明記しています。重要なポイントは：
- **Mデータの読み込み方式**：全期間分を一度だけメモリに読み込み、その後のループで参照し続ける（メモリ効率的）
- **インテージデータの読み込み方式**：毎月のCSVをループで1ヶ月ずつ読み込み→Mデータと結合→出力→メモリ解放（ストレージ効率的）
- この戦略により、メモリ使用量を制限しながら大量のデータを処理可能

---

## Cell 1: ライブラリインポートと環境設定

```python
import pandas as pd
import pyarrow.csv as pa_csv
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
import gc
import shutil
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# === パス設定 ===
BASE_PATH = "/home/ciwai/work/data/0014_MTC_information_health"
INTAGE_PATH = os.path.join(BASE_PATH, "intage2308-2404")
MDATA_PATH = "/home/ciwai/work/data/0014_MTC_information_health/mdata/mdata_with_topics.parquet"

# 出力先
OUTPUT_PATH = "/home/ciwai/work/data/0014_MTC_information_health/intage_mdata_connect"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 不要カラム（インテージ読み込み時に除外）
DROP_COLS = {'household_id', 'watch_flg', 'program_start_date',
             'program_end_date', 'ch_cf', 'bc_area_id'}

# 放送局名マッピング: インテージ channel_2 → Mデータ 放送局
CHANNEL_MAP = {
    'NHK': 'NHK', 'ETV': '教育', 'NTV': 'NTV',
    'TBS': 'TBS', 'CX': 'CX', 'EX': 'EX', 'TX': 'TX',
}

print(f"出力先: {OUTPUT_PATH}")
```

**行ごと解説：**

1. **`import pandas as pd`** - pandas ライブラリをインポート。DataFrameの操作に使用
2. **`import pyarrow.csv as pa_csv`** - PyArrowのCSV読み込み機能。pandas.read_csv()より高速で、メモリ効率が良い
3. **`import pyarrow as pa`** - PyArrowコア。テーブル操作用
4. **`import pyarrow.parquet as pq`** - Parquetフォーマットの読み書き機能。圧縮率が高く効率的
5. **`import glob`** - ワイルドカードでファイルパスを一括取得
6. **`import os`** - ファイルパス操作とディレクトリ管理
7. **`import gc`** - ガベージコレクション。メモリ解放の強制実行に使用
8. **`import shutil`** - ファイル・ディレクトリの削除操作
9. **`import warnings`** - 警告メッセージの管理
10. **`from tqdm import tqdm`** - プログレスバーを表示するライブラリ。ループ進捗の可視化
11. **`warnings.filterwarnings('ignore')`** - 不要な警告を非表示化

**パス設定：**
- **`BASE_PATH`** - データ保存ベースディレクトリ
- **`INTAGE_PATH`** - インテージ月別CSVの親ディレクトリ。サフォルダは `tb_sml_intage_fixed_programbasic_202308` のようなフォーマット
- **`MDATA_PATH`** - Mデータ（シーンテーブル）のParquetファイルパス。事前にCell 02 や別ノートブックで生成（→ NB01 で生成）
- **`OUTPUT_PATH`** - 統合後のParquetファイル出力先。フォルダがなければ作成

**不要カラム定義 `DROP_COLS`：**
- インテージCSVから除外すべきカラム。以下が不要：
  - `household_id` - 個人識別情報
  - `watch_flg` - 視聴フラグ（フィルタリング後は不要）
  - `program_start_date`, `program_end_date` - 番組開始終了日（`watch_date` で統合）
  - `ch_cf`, `bc_area_id` - 放送局・エリアの別コード

**放送局マッピング `CHANNEL_MAP`：**
インテージのカラム `channel_2` の値をMデータの「放送局」カラムの値に統一するための辞書。
- 例：インテージで `'ETV'` → Mデータの `'教育'` に統一
- マップにない値は除外される

---

## Cell 2: Mデータ読み込み＆シーンテーブル準備（マークダウン）

```markdown
## Mデータ読み込み & シーンテーブル準備（全期間・1回のみ）
```

**説明：**
このセクションではMデータを1回だけ読み込み、シーンテーブルを全期間分準備します。このテーブルは後続のループで参照され続けます。

---

## Cell 3: Mデータの読み込み・加工

```python
# Mデータ読み込み（必要カラムのみ）
MDATA_COLS = ['放送局', 'シーン開始日時24', 'シーン終了日時24',
              'ヘッドライン', 'メモ', '番組ジャンル', '分類', '番組名',
              'topic', 'topic_prob', 'annual_topic_id','is_negative']

print(f"Mデータ読み込み: {MDATA_PATH}")
df_scene = pd.read_parquet(MDATA_PATH, columns=MDATA_COLS)
print(f"Mデータ行数: {len(df_scene):,}")

# 日時変換
df_scene['シーン開始dt'] = pd.to_datetime(df_scene['シーン開始日時24'])
df_scene['シーン終了dt'] = pd.to_datetime(df_scene['シーン終了日時24'])
df_scene = df_scene.drop(columns=['シーン開始日時24', 'シーン終了日時24'])

# 0秒シーン除外
n_before = len(df_scene)
df_scene = df_scene[df_scene['シーン開始dt'] != df_scene['シーン終了dt']].reset_index(drop=True)
print(f"0秒シーン除外: {n_before - len(df_scene):,}件 → 残り{len(df_scene):,}件")

df_scene = df_scene.sort_values(['放送局', 'シーン開始dt']).reset_index(drop=True)
gc.collect()

print(f"シーンテーブル準備完了: {len(df_scene):,}件")
print(f"メモリ使用量: {df_scene.memory_usage(deep=True).sum() / 2**20:.0f} MB")
```

**行ごと解説：**

1. **`MDATA_COLS = [...]`** - 読み込みカラムリスト。不要なカラムを除外し、メモリ使用量を削減
   - 放送局情報：`'放送局'`
   - 時系列情報：`'シーン開始日時24'`, `'シーン終了日時24'`
   - メタデータ：`'ヘッドライン'`, `'メモ'`, `'番組ジャンル'`, `'分類'`, `'番組名'`
   - トピック情報：`'topic'`, `'topic_prob'`, `'annual_topic_id'`
   - 感情分析：`'is_negative'`

2. **`df_scene = pd.read_parquet(MDATA_PATH, columns=MDATA_COLS)`** - Parquetから指定カラムのみ読み込み。フルカラム読み込みより高速かつ省メモリ

3. **`df_scene['シーン開始dt'] = pd.to_datetime(...)`** - 文字列型の日時をdatetime64型に変換
   - `pd.to_datetime()` は自動的にタイムゾーン対応、フォーマット判定を行う
   - 例：`'2023-08-15 14:30:00'` → `Timestamp('2023-08-15 14:30:00')`

4. **`df_scene['シーン終了dt'] = pd.to_datetime(...)`** - シーン終了時刻も同様に変換

5. **`df_scene = df_scene.drop(columns=[...])`** - 元の文字列カラムを削除（datetime型カラムに置き換え済み）

6. **`n_before = len(df_scene)`** - フィルタ前の行数を保存（統計出力用）

7. **`df_scene = df_scene[df_scene['シーン開始dt'] != df_scene['シーン終了dt']].reset_index(drop=True)`**
   - **フィルタ条件**：開始日時と終了日時が同じ行を除外（0秒シーン）
   - **具体例**：開始 `2023-08-15 14:30:00` と終了 `2023-08-15 14:30:00` が一致 → 除外
   - **反例**：開始 `2023-08-15 14:30:00` と終了 `2023-08-15 14:35:00` → 保持
   - `.reset_index(drop=True)` でインデックスを 0 から振り直す

8. **`print(f"0秒シーン除外: {n_before - len(df_scene):,}件 → 残り{len(df_scene):,}件")`** - 除外行数と残り行数を表示
   - `{:,}` で3桁区切りを挿入（可読性向上）

9. **`df_scene = df_scene.sort_values(['放送局', 'シーン開始dt']).reset_index(drop=True)`**
   - **ソート順序**：放送局でグループ化し、各放送局内で開始日時昇順に並べ替え
   - **重要性**：merge_asof は左テーブル（インテージ）がソート済みでないと正動作しない。右テーブル（Mデータ）も同じ キー でソート済みが望ましい
   - **具体例**：
     ```
     放送局   シーン開始dt
     NHK     2023-08-15 08:00:00
     NHK     2023-08-15 10:00:00
     NTV     2023-08-15 09:00:00
     NTV     2023-08-15 11:00:00
     ```

10. **`gc.collect()`** - ガベージコレクションを強制実行。不要なメモリを即座に解放（大規模データ処理では重要）

11. **`print(f"シーンテーブル準備完了: {len(df_scene):,}件")`** - 最終行数を表示

12. **`print(f"メモリ使用量: {df_scene.memory_usage(deep=True).sum() / 2**20:.0f} MB")`**
    - `memory_usage(deep=True)` - 各カラムのメモリ使用量を計算（オブジェクト型は詳細分析）
    - `.sum()` - 全カラムの合計
    - `/ 2**20` - バイト → MB に変換（1 MB = 2^20 バイト）
    - 例：`df_scene.memory_usage(deep=True).sum() = 536870912` → `512` MB

---

## Cell 4: 月次ループセクション説明（マークダウン）

```markdown
## 月次ループ：インテージ読み込み → 結合 → CSV保存 → メモリ解放
```

**説明：**
このセクションの処理フロー：
1. 月別フォルダを列挙
2. 各月について：
   - Phase 1: CSVを放送局別に一時Parquetに振り分け
   - Phase 2: 一時Parquetをmerge_asofで結合
   - Phase 3: 最終Parquetに追記
   - Phase 4: 一時ファイルを削除してメモリ解放

---

## Cell 5: 月次ループメインロジック

```python
# 月別フォルダを取得
intage_dirs = sorted(glob.glob(os.path.join(INTAGE_PATH, "tb_sml_intage_fixed_programbasic_*")))
print(f"処理対象月数: {len(intage_dirs)}")
for d in intage_dirs:
    print(f"  {os.path.basename(d)}")

SCENE_COLS = ['ヘッドライン', 'メモ', '番組ジャンル', '分類', '番組名', 'topic', 'topic_prob', 'annual_topic_id', 'is_negative']
CHANNELS = list(CHANNEL_MAP.values())
TMP_DIR = os.path.join(OUTPUT_PATH, "_tmp")
summary = []

for month_dir in intage_dirs:
    month_name = os.path.basename(month_dir)
    month_code = month_name.split('_')[-1]
    output_file = os.path.join(OUTPUT_PATH, f"merged_{month_code}.parquet")

    if os.path.exists(output_file):
        print(f"\n=== {month_code} === スキップ（既に存在: {output_file}）")
        summary.append({'month': month_code, 'status': 'skipped'})
        continue

    print(f"\n{'='*60}")
    print(f"=== {month_code} 処理開始 ===")

    csv_files = sorted(glob.glob(os.path.join(month_dir, "*.csv")))
    print(f"  ファイル数: {len(csv_files)}")

    # ========================================
    # Phase 1: CSV1回読 → 放送局別に一時parquetへ振り分け
    # ========================================
    os.makedirs(TMP_DIR, exist_ok=True)
    ch_writers = {}  # 放送局 → ParquetWriter

    for f in tqdm(csv_files, desc=f"  {month_code} CSV読込・振分"):
        # PyArrow CSVで高速読み込み
        table = pa_csv.read_csv(f)
        # 不要カラム除外
        keep_cols = [c for c in table.column_names if c not in DROP_COLS]
        table = table.select(keep_cols)
        df_tmp = table.to_pandas()
        del table

        # 放送局マッピング & マッピング外を除外
        df_tmp['放送局'] = df_tmp['channel_2'].map(CHANNEL_MAP)
        df_tmp = df_tmp.dropna(subset=['放送局'])

        if len(df_tmp) == 0:
            continue

        # 放送局別に振り分けて一時parquetに追記
        for ch, df_ch in df_tmp.groupby('放送局'):
            tmp_path = os.path.join(TMP_DIR, f"{month_code}_{ch}.parquet")
            pa_table = pa.Table.from_pandas(df_ch, preserve_index=False)

            if ch not in ch_writers:
                ch_writers[ch] = pq.ParquetWriter(tmp_path, pa_table.schema)
            ch_writers[ch].write_table(pa_table)
            del pa_table

        del df_tmp

    # Writerを全て閉じる
    for w in ch_writers.values():
        w.close()
    del ch_writers
    gc.collect()

    # ========================================
    # Phase 2: 放送局別に一時parquet読み → merge_asof → 最終parquet追記
    # ========================================
    n_total = 0
    n_success = 0
    writer = None

    for ch in tqdm(CHANNELS, desc=f"  {month_code} merge_asof"):
        tmp_path = os.path.join(TMP_DIR, f"{month_code}_{ch}.parquet")
        if not os.path.exists(tmp_path):
            continue

        intage_ch = pd.read_parquet(tmp_path)
        # datetime解像度をdf_scene(ns)に揃える
        intage_ch['watch_dt'] = pd.to_datetime(intage_ch['watch_date']).astype('datetime64[ns]')
        intage_ch = intage_ch.sort_values('watch_dt').reset_index(drop=True)

        scene_ch = df_scene[df_scene['放送局'] == ch]

        if len(scene_ch) == 0:
            for col in SCENE_COLS + ['シーン開始dt', 'シーン終了dt']:
                intage_ch[col] = pd.NaT if 'dt' in col else None
            merged_ch = intage_ch
        else:
            merged_ch = pd.merge_asof(
                intage_ch,
                scene_ch,
                left_on='watch_dt',
                right_on='シーン開始dt',
                by='放送局',
                direction='backward',
            )
            del intage_ch
            gc.collect()

            out_of_range = merged_ch['watch_dt'] >= merged_ch['シーン終了dt']
            for col in SCENE_COLS:
                merged_ch.loc[out_of_range, col] = None

        # is_negativeの型を統一（NaNが入るとfloatになるので明示的にInt64に）
        if 'is_negative' in merged_ch.columns:
            merged_ch['is_negative'] = merged_ch['is_negative'].astype('Int64')

        n_total += len(merged_ch)
        n_success += merged_ch['分類'].notna().sum()

        # 最終parquetに追記（メモリに全局分を載せない）
        pa_table = pa.Table.from_pandas(merged_ch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_file, pa_table.schema)
        writer.write_table(pa_table)
        del merged_ch, pa_table
        gc.collect()

    if writer is not None:
        writer.close()
    del writer
    gc.collect()

    # 一時ファイル削除
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    rate = n_success / n_total * 100
    file_size_mb = os.path.getsize(output_file) / 2**20
    print(f"  結合後行数: {n_total:,}")
    print(f"  紐付け率: {rate:.1f}% ({n_success:,} / {n_total:,})")
    print(f"  保存完了: {output_file} ({file_size_mb:.0f}MB)")

    summary.append({'month': month_code, 'rows': n_total, 'rate': f"{rate:.1f}%", 'size_mb': f"{file_size_mb:.0f}"})

print(f"\n{'='*60}")
print("=== 全月処理完了 ===")
pd.DataFrame(summary)
```

### 詳細解説：

#### ステップ 1: 月別フォルダの列挙

```python
intage_dirs = sorted(glob.glob(os.path.join(INTAGE_PATH, "tb_sml_intage_fixed_programbasic_*")))
print(f"処理対象月数: {len(intage_dirs)}")
for d in intage_dirs:
    print(f"  {os.path.basename(d)}")
```

- **`glob.glob(...)`** - ワイルドカード `*` で全マッチディレクトリを取得
  - 例：`tb_sml_intage_fixed_programbasic_202308`, `tb_sml_intage_fixed_programbasic_202309`, ...
- **`sorted(...)`** - アルファベット順（タイムスタンプ順）にソート
- **`os.path.basename(d)`** - フルパスからフォルダ名のみ抽出
  - 例：`/home/.../tb_sml_intage_fixed_programbasic_202308` → `tb_sml_intage_fixed_programbasic_202308`

#### ステップ 2: グローバル変数設定

```python
SCENE_COLS = ['ヘッドライン', 'メモ', '番組ジャンル', '分類', '番組名', 'topic', 'topic_prob', 'annual_topic_id', 'is_negative']
CHANNELS = list(CHANNEL_MAP.values())
TMP_DIR = os.path.join(OUTPUT_PATH, "_tmp")
summary = []
```

- **`SCENE_COLS`** - Mデータから追加されるカラム。merge_asof後にこれらがインテージに付与される
- **`CHANNELS`** - `CHANNEL_MAP.values()` を取得 → `['NHK', '教育', 'NTV', 'TBS', 'CX', 'EX', 'TX']`
- **`TMP_DIR`** - 一時Parquetファイル保存ディレクトリ（Phase 1の出力先）
- **`summary`** - 各月の処理結果を格納するリスト（最後に統計表示）

#### ステップ 3: 月別ループ開始

```python
for month_dir in intage_dirs:
    month_name = os.path.basename(month_dir)
    month_code = month_name.split('_')[-1]
    output_file = os.path.join(OUTPUT_PATH, f"merged_{month_code}.parquet")
```

- **`month_name`** - ディレクトリ名を抽出
  - 例：`tb_sml_intage_fixed_programbasic_202308`
- **`month_code`** - ディレクトリ名を `_` で分割し、最後の要素を取得
  - 例：`'tb_sml_intage_fixed_programbasic_202308'.split('_')[-1]` → `'202308'`
- **`output_file`** - 月別出力ファイルパス
  - 例：`/home/.../intage_mdata_connect/merged_202308.parquet`

#### ステップ 4: 既存ファイルのスキップ判定

```python
    if os.path.exists(output_file):
        print(f"\n=== {month_code} === スキップ（既に存在: {output_file}）")
        summary.append({'month': month_code, 'status': 'skipped'})
        continue
```

- **べき等性（Idempotent）の実現**：同じ月の処理を重複実行しない
- **用途**：ノートブック再実行時に、既に処理済み月をスキップ
- **`continue`** - 次の month_dir に移動

#### ステップ 5: Phase 1 - CSV読込・放送局別振り分け

```python
    csv_files = sorted(glob.glob(os.path.join(month_dir, "*.csv")))
    print(f"  ファイル数: {len(csv_files)}")

    os.makedirs(TMP_DIR, exist_ok=True)
    ch_writers = {}  # 放送局 → ParquetWriter

    for f in tqdm(csv_files, desc=f"  {month_code} CSV読込・振分"):
        # PyArrow CSVで高速読み込み
        table = pa_csv.read_csv(f)
        # 不要カラム除外
        keep_cols = [c for c in table.column_names if c not in DROP_COLS]
        table = table.select(keep_cols)
        df_tmp = table.to_pandas()
        del table

        # 放送局マッピング & マッピング外を除外
        df_tmp['放送局'] = df_tmp['channel_2'].map(CHANNEL_MAP)
        df_tmp = df_tmp.dropna(subset=['放送局'])

        if len(df_tmp) == 0:
            continue

        # 放送局別に振り分けて一時parquetに追記
        for ch, df_ch in df_tmp.groupby('放送局'):
            tmp_path = os.path.join(TMP_DIR, f"{month_code}_{ch}.parquet")
            pa_table = pa.Table.from_pandas(df_ch, preserve_index=False)

            if ch not in ch_writers:
                ch_writers[ch] = pq.ParquetWriter(tmp_path, pa_table.schema)
            ch_writers[ch].write_table(pa_table)
            del pa_table

        del df_tmp

    # Writerを全て閉じる
    for w in ch_writers.values():
        w.close()
    del ch_writers
    gc.collect()
```

**目的：** インテージのCSVをメモリ効率的に読み込み、放送局別に一時Parquetに分散保存

**処理フロー：**

1. **`csv_files = sorted(glob.glob(...))`** - 月内のCSVファイルを全取得し、ソート

2. **`table = pa_csv.read_csv(f)`**
   - PyArrow CSVリーダーで高速読み込み
   - 従来の pandas.read_csv() より 5-10倍高速（C++実装）
   - メモリ効率も優れている

3. **カラムフィルタリング**
   ```python
   keep_cols = [c for c in table.column_names if c not in DROP_COLS]
   table = table.select(keep_cols)
   ```
   - 不要カラムを除外
   - 例：`DROP_COLS = {'household_id', ...}` なら、このカラムを削除
   - 理由：メモリ削減 + 個人情報保護

4. **`df_tmp = table.to_pandas()`** - PyArrowテーブルをPandas DataFrameに変換

5. **放送局マッピング**
   ```python
   df_tmp['放送局'] = df_tmp['channel_2'].map(CHANNEL_MAP)
   ```
   - インテージの `channel_2` カラムを Mデータの「放送局」にマッピング
   - 例：
     ```
     元の channel_2   →  マッピング後の放送局
     'NHK'            →  'NHK'
     'ETV'            →  '教育'
     'XXX'（不在）     →  NaN
     ```

6. **`df_tmp = df_tmp.dropna(subset=['放送局'])`**
   - マッピング失敗した行（`放送局` が NaN）を除外
   - 例：インテージに `channel_2='UNKNOWN'` があって CHANNEL_MAP に未登録 → 除外

7. **空チェック**
   ```python
   if len(df_tmp) == 0:
       continue
   ```
   - フィルタ後に行がない場合、次のCSVへ

8. **放送局別グループ化と一時Parquet追記**
   ```python
   for ch, df_ch in df_tmp.groupby('放送局'):
       tmp_path = os.path.join(TMP_DIR, f"{month_code}_{ch}.parquet")
       pa_table = pa.Table.from_pandas(df_ch, preserve_index=False)

       if ch not in ch_writers:
           ch_writers[ch] = pq.ParquetWriter(tmp_path, pa_table.schema)
       ch_writers[ch].write_table(pa_table)
   ```
   - **具体例**：
     - `df_tmp` に `放送局=['NHK', 'NHK', 'NTV', 'NTV']` があるとき
     - `groupby('放送局')` は `(ch='NHK', df_ch=rows 0,1)` と `(ch='NTV', df_ch=rows 2,3)` に分割
   - **ParquetWriter の役割**：
     - `pq.ParquetWriter(path, schema)` でファイルを初期化
     - `.write_table()` でテーブルを追記
     - 複数CSVから同じ放送局データが来ても、Writerが保持されていれば追記される
   - **メモリ最適化**：
     - `del pa_table` で各テーブルを即座に削除
     - `del df_tmp` で元データも削除

9. **Writerの閉じる処理**
   ```python
   for w in ch_writers.values():
       w.close()
   del ch_writers
   gc.collect()
   ```
   - 全Writerをクローズ（ファイルをフラッシュ）
   - 不要な参照を削除
   - ガベージコレクション実行

**メモリ最適化戦略（Phase 1の鍵）：**
- **大規模CSVを分割処理**：1ファイルずつ読み込みながら、放送局別に振り分け
- **一時Parquetの活用**：メモリに全月データを保持せず、ディスクに分散
- **Writerの再利用**：同じ放送局への複数追記で、ファイルオープン・クローズのコストを削減
- **即時削除**：不要なオブジェクトは使用直後に `del` して削除

#### ステップ 6: Phase 2 - merge_asof による時系列結合

```python
    n_total = 0
    n_success = 0
    writer = None

    for ch in tqdm(CHANNELS, desc=f"  {month_code} merge_asof"):
        tmp_path = os.path.join(TMP_DIR, f"{month_code}_{ch}.parquet")
        if not os.path.exists(tmp_path):
            continue

        intage_ch = pd.read_parquet(tmp_path)
        # datetime解像度をdf_scene(ns)に揃える
        intage_ch['watch_dt'] = pd.to_datetime(intage_ch['watch_date']).astype('datetime64[ns]')
        intage_ch = intage_ch.sort_values('watch_dt').reset_index(drop=True)

        scene_ch = df_scene[df_scene['放送局'] == ch]
```

**目的：** 各放送局について、インテージデータ（視聴ログ）とMデータ（番組シーン）を時系列で結合

**前処理：**

1. **`intage_ch = pd.read_parquet(tmp_path)`**
   - Phase 1で保存した、放送局別の一時Parquetを読み込み
   - 例：`merged_202308_NHK.parquet`

2. **datetime解像度の統一**
   ```python
   intage_ch['watch_dt'] = pd.to_datetime(intage_ch['watch_date']).astype('datetime64[ns]')
   ```
   - **重要**：merge_asof は左右のキーの時間解像度が同じである必要がある
   - インテージの `watch_date` は可能性として秒単位だが、Mデータ（`df_scene`）はナノ秒単位
   - `.astype('datetime64[ns]')` で明示的にナノ秒に統一
   - **具体例**：
     ```
     'watch_date' → '2023-08-15 14:30:00'
     → pd.to_datetime() → Timestamp('2023-08-15 14:30:00')
     → astype('datetime64[ns]') → 2023-08-15T14:30:00.000000000
     ```

3. **インテージデータのソート**
   ```python
   intage_ch = intage_ch.sort_values('watch_dt').reset_index(drop=True)
   ```
   - merge_asof は左テーブルがソート済みであることが必須
   - `.reset_index(drop=True)` でインデックスを振り直し

4. **フィルタ局のMデータ抽出**
   ```python
   scene_ch = df_scene[df_scene['放送局'] == ch]
   ```
   - 全体Mデータから、処理中の放送局 `ch` に限定
   - 例：`ch = 'NHK'` なら、`df_scene` から放送局='NHK' 行のみ

#### merge_asof の詳細仕組み

```python
        if len(scene_ch) == 0:
            for col in SCENE_COLS + ['シーン開始dt', 'シーン終了dt']:
                intage_ch[col] = pd.NaT if 'dt' in col else None
            merged_ch = intage_ch
        else:
            merged_ch = pd.merge_asof(
                intage_ch,
                scene_ch,
                left_on='watch_dt',
                right_on='シーン開始dt',
                by='放送局',
                direction='backward',
            )
            del intage_ch
            gc.collect()

            out_of_range = merged_ch['watch_dt'] >= merged_ch['シーン終了dt']
            for col in SCENE_COLS:
                merged_ch.loc[out_of_range, col] = None
```

**ケース1：該当放送局にMデータがない場合**
```python
        if len(scene_ch) == 0:
            for col in SCENE_COLS + ['シーン開始dt', 'シーン終了dt']:
                intage_ch[col] = pd.NaT if 'dt' in col else None
            merged_ch = intage_ch
```
- Mデータカラムをすべて NaN (NaT for datetime) で初期化
- インテージデータ単独で出力

**ケース2：Mデータが存在する場合**

```python
            merged_ch = pd.merge_asof(
                intage_ch,
                scene_ch,
                left_on='watch_dt',
                right_on='シーン開始dt',
                by='放送局',
                direction='backward',
            )
```

**pd.merge_asof の動作説明：**

- **用途**：非完全一致の時系列結合。「この視聴時刻に、どの番組シーンが放映中か」を特定

- **パラメータ詳細**：
  - `left_on='watch_dt'` - インテージの視聴時刻キー
  - `right_on='シーン開始dt'` - Mデータのシーン開始時刻キー
  - `by='放送局'` - この放送局に限定して結合
  - `direction='backward'` - 視聴時刻より前のシーン開始を探す

- **具体例**：
  ```
  インテージ (intage_ch):
    watch_dt              放送局
  2023-08-15 14:30:00   NHK
  2023-08-15 14:35:00   NHK
  2023-08-15 14:40:00   NHK

  Mデータ (scene_ch):
    シーン開始dt          シーン終了dt          ヘッドライン        放送局
  2023-08-15 14:20:00  2023-08-15 14:32:00  "天気予報"         NHK
  2023-08-15 14:32:00  2023-08-15 14:38:00  "ニュース"         NHK
  2023-08-15 14:38:00  2023-08-15 14:45:00  "スポーツ"         NHK

  merge_asof (direction='backward') の結果：
    watch_dt              放送局  シーン開始dt          ヘッドライン
  2023-08-15 14:30:00   NHK   2023-08-15 14:20:00  "天気予報"
  2023-08-15 14:35:00   NHK   2023-08-15 14:32:00  "ニュース"
  2023-08-15 14:40:00   NHK   2023-08-15 14:38:00  "スポーツ"
  ```

  説明：
  - 視聴時刻 14:30:00 は、14:20:00 の「天気予報」シーン開始以降・14:32:00 のニュース開始以前 → 「天気予報」にマッチ
  - 視聴時刻 14:35:00 は、14:32:00 のニュース開始以降 → 「ニュース」にマッチ
  - 視聴時刻 14:40:00 は、14:38:00 のスポーツ開始以降 → 「スポーツ」にマッチ

**out_of_range フィルタ（重要）：**

```python
            out_of_range = merged_ch['watch_dt'] >= merged_ch['シーン終了dt']
            for col in SCENE_COLS:
                merged_ch.loc[out_of_range, col] = None
```

- **問題**：merge_asof は「開始時刻以降の最新シーン」を見つけるが、シーン終了後の視聴も結合してしまう
- **対策**：視聴時刻がシーン終了時刻以降なら、Mデータカラムを NaN に置き換え
- **具体例**：
  ```
  元の結果:
    watch_dt              シーン開始dt          シーン終了dt          ヘッドライン
  2023-08-15 14:35:00   2023-08-15 14:32:00  2023-08-15 14:38:00  "ニュース"
  2023-08-15 14:50:00   2023-08-15 14:38:00  2023-08-15 14:45:00  "スポーツ"  ← out_of_range

  フィルタ後:
    watch_dt              シーン開始dt          シーン終了dt          ヘッドライン
  2023-08-15 14:35:00   2023-08-15 14:32:00  2023-08-15 14:38:00  "ニュース"
  2023-08-15 14:50:00   NaT                  NaT                  NaN
  ```
  - 14:50:00 は 14:45:00 より後なので、シーン終了後 → NaN化

**型統一処理：**

```python
        # is_negativeの型を統一（NaNが入るとfloatになるので明示的にInt64に）
        if 'is_negative' in merged_ch.columns:
            merged_ch['is_negative'] = merged_ch['is_negative'].astype('Int64')
```

- **背景**：Mデータの `is_negative` は整数（0, 1）だが、NaN が入るとfloat型に自動変換される
  - 例：`[0, 1, 0, nan]` → dtype は float64
- **対策**：Pandas 拡張整数型 `Int64` を使用（大文字）
  - `Int64` は NaN をサポート（通常の `int64` はしない）
- **具体例**：
  ```python
  df['is_negative'] = [0, 1, 0, NaN]  # float64
  df['is_negative'] = df['is_negative'].astype('Int64')  # Int64（NaN対応）
  ```

#### ステップ 7: Phase 3 - 最終Parquetへの追記と統計集計

```python
        n_total += len(merged_ch)
        n_success += merged_ch['分類'].notna().sum()

        # 最終parquetに追記（メモリに全局分を載せない）
        pa_table = pa.Table.from_pandas(merged_ch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_file, pa_table.schema)
        writer.write_table(pa_table)
        del merged_ch, pa_table
        gc.collect()

    if writer is not None:
        writer.close()
    del writer
    gc.collect()
```

**統計情報の積算：**

```python
        n_total += len(merged_ch)
        n_success += merged_ch['分類'].notna().sum()
```

- **`n_total`** - 全局のインテージ行数合計
- **`n_success`** - Mデータとマッチした行数（「分類」カラムが NaN でない）
  - `merged_ch['分類'].notna().sum()` は、分類カラムが存在する行数を数える
  - 例：`['TV', 'NEWS', NaN, 'DRAMA']` → `notna()` は `[True, True, False, True]` → `.sum()` は 3

**Parquet追記戦略：**

```python
        pa_table = pa.Table.from_pandas(merged_ch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_file, pa_table.schema)
        writer.write_table(pa_table)
        del merged_ch, pa_table
        gc.collect()
```

- **メモリ効率**：全放送局をメモリに保持せず、局ごとに処理・出力・削除
- **Writerの再利用**：複数局を同一ファイルに追記するため、Writerは1ファイル1つ
- **処理フロー**：
  1. 1局目：`writer is None` → 初期化 → `write_table()`
  2. 2局目以降：`writer is not None` → `write_table()` のみ
  3. 最後に `writer.close()`

#### ステップ 8: クリーンアップと統計出力

```python
    # 一時ファイル削除
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    rate = n_success / n_total * 100
    file_size_mb = os.path.getsize(output_file) / 2**20
    print(f"  結合後行数: {n_total:,}")
    print(f"  紐付け率: {rate:.1f}% ({n_success:,} / {n_total:,})")
    print(f"  保存完了: {output_file} ({file_size_mb:.0f}MB)")

    summary.append({'month': month_code, 'rows': n_total, 'rate': f"{rate:.1f}%", 'size_mb': f"{file_size_mb:.0f}"})

print(f"\n{'='*60}")
print("=== 全月処理完了 ===")
pd.DataFrame(summary)
```

**処理：**

1. **`shutil.rmtree(TMP_DIR, ignore_errors=True)`**
   - 一時フォルダ（`_tmp`）を削除
   - `ignore_errors=True` でフォルダが存在しなくてもエラーにしない

2. **統計計算**
   ```python
   rate = n_success / n_total * 100
   ```
   - 「紐付け率」 = Mデータとマッチした割合
   - 例：`n_success=900, n_total=1000` → `rate=90.0%`

3. **ファイルサイズ取得**
   ```python
   file_size_mb = os.path.getsize(output_file) / 2**20
   ```
   - バイト → MB 変換

4. **結果をサマリーリストに追加**
   ```python
   summary.append({'month': month_code, 'rows': n_total, 'rate': f"{rate:.1f}%", 'size_mb': f"{file_size_mb:.0f}"})
   ```

5. **全月処理完了後、サマリーテーブル表示**
   ```python
   pd.DataFrame(summary)
   ```
   - 結果例：
     ```
        month       rows    rate  size_mb
     0  202308    5000000  87.3%    450
     1  202309    4800000  88.1%    430
     2  202310    5200000  86.5%    460
     ```

---

## Cell 6: 空セル

```python

```

このセルは処理内容がありません。ノートブック終了。

---

## 全体アーキテクチャ図

```
┌─────────────────────────────────────────────────────────────┐
│ Cell 3: Mデータ読み込み（1回）                                │
│ df_scene: 全期間 × 全局の番組シーン情報                        │
│ ソート済み: (放送局, シーン開始dt)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   月別ループ (202308, 202309, ...)     │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ Phase 1: CSV → 放送局別一時Parquet     │
        │ (NHK, 教育, NTV, TBS, CX, EX, TX)     │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ Phase 2: merge_asof (時系列結合)      │
        │ インテージ + Mデータ → 統合データ       │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ Phase 3: 最終Parquet出力              │
        │ merged_202308.parquet など             │
        └───────────────────────────────────────┘
```

---

## メモリ最適化戦略の総括

### なぜPhase 1（CSV→一時Parquet）が必要か？

インテージのCSVは膨大なデータ量を持つため、直接メモリに全読み込みすると Out of Memory になります。

**戦略：**
1. **CSVを1ファイルずつ読み込み**
2. **放送局別にディスクの一時Parquetに書き込み**（ParquetWriter追記で効率化）
3. **次のCSVを読み込み**（前ファイルはメモリから削除）
4. **繰り返す**

**メモリピーク：** 1CSVファイル分のメモリのみ（全月データではなく）

### なぜPhase 2（放送局別処理）が必要か？

merge_asof後も、全局のマージ結果を同時にメモリに保持すると Out of Memory になります。

**戦略：**
1. **1放送局ずつ一時Parquetを読み込み**
2. **merge_asof で結合**
3. **最終Parquetに追記**（ParquetWriter追記）
4. **メモリから削除**
5. **次の放送局を処理**

**メモリピーク：** 1放送局分のインテージデータ + Mデータ（1局）のメモリのみ

### ParquetWriter の活用

通常、Parquetを複数回`append`すると非効率です。当実装では **ParquetWriter** を使用：

```python
# 複数回の append をまとめて、1度の書き込み処理で実装
writer = pq.ParquetWriter(path, schema)
writer.write_table(table1)
writer.write_table(table2)
writer.close()  # 一度の書き込み完了
```

このため、メモリ効率とディスクI/O効率の両立が可能です。

---

## merge_asof の実用例と注意点

### 例1: 正常マッチ

```
視聴時刻: 2023-08-15 14:32:30
シーン開始: 2023-08-15 14:32:00 (ニュース開始)
シーン終了: 2023-08-15 14:38:00 (ニュース終了)

判定: 32:30 は 32:00 以降・38:00 以前 → ニュース放映中 ✓
```

### 例2: シーン終了後（out_of_range フィルタで除外）

```
視聴時刻: 2023-08-15 14:50:00
シーン開始: 2023-08-15 14:38:00 (スポーツ開始)
シーン終了: 2023-08-15 14:45:00 (スポーツ終了)

判定: 50:00 は 45:00 より後 → シーン終了後 → NaN化 ✗
```

### 例3: シーン開始前

```
視聴時刻: 2023-08-15 14:30:00
該当シーン: 2023-08-15 14:32:00 以降の最新シーン

判定: backward 方向では、30:00 より前のシーン開始を探す → なし → NaN ✗
```

---

## データフロー図（具体例）

### インテージ (視聴ログ)

```
watch_date   channel_2  ...
2023-08-15   NHK
2023-08-15   NHK
2023-08-15   NTV
2023-08-15   NTV
```

↓ Channel Mapping (`channel_2 → 放送局`)

```
watch_date   放送局
2023-08-15   NHK
2023-08-15   NHK
2023-08-15   NTV
2023-08-15   NTV
```

↓ Phase 1: 一時Parquet振り分け

```
202308_NHK.parquet  (NHK rows)
202308_NTV.parquet  (NTV rows)
```

↓ Phase 2: merge_asof

```
Mデータ (df_scene filtered by ch='NHK'):
  シーン開始dt          ヘッドライン
  2023-08-15 14:00:00  ニュース

インテージ (ch='NHK'):
  watch_dt (= 2023-08-15 14:30:00)

結合結果:
  watch_dt             ヘッドライン
  2023-08-15 14:30:00  ニュース  ✓
```

↓ Phase 3: 最終Parquetに追記

```
merged_202308.parquet
  watch_dt   ヘッドライン  メモ  topic  ...
  ...        ...         ...   ...   ...
```

---

## 注意点とトラブルシューティング

### 1. datetime 解像度のズレ

**問題：** merge_asof でマッチしない

**原因：** 左右のテーブルの datetime 型の解像度が異なる
- インテージ：秒単位（datetime64[s]）
- Mデータ：ナノ秒単位（datetime64[ns]）

**対策：** 明示的に統一
```python
intage_ch['watch_dt'] = intage_ch['watch_date'].astype('datetime64[ns]')
```

### 2. ソート順序の問題

**問題：** merge_asof でエラーが出る

**原因：** 左テーブル（インテージ）がソート済みでない

**対策：**
```python
intage_ch = intage_ch.sort_values('watch_dt').reset_index(drop=True)
```

### 3. メモリ不足

**問題：** Out of Memory エラー

**原因：** Phase 1 または Phase 2 で大量データをメモリに保持

**対策：**
- CSVファイルサイズを縮小（事前に分割）
- `gc.collect()` を頻繁に実行
- 本スクリプトのように放送局別に処理

### 4. merge_asof 後の幽霊マッチ

**問題：** シーン終了後の視聴が、古いシーンにマッチしている

**原因：** backward 方向での merge_asof は「前のシーン開始」を見つけるだけで、終了は考慮しない

**対策：** out_of_range フィルタを適用
```python
out_of_range = merged_ch['watch_dt'] >= merged_ch['シーン終了dt']
merged_ch.loc[out_of_range, col] = None
```

---

## 出力ファイル仕様

### ファイルフォーマット

- **形式：** Parquet（列指向、圧縮フォーマット）
- **出力先：** `/home/ciwai/work/data/0014_MTC_information_health/intage_mdata_connect/`
- **ファイル名：** `merged_202308.parquet`, `merged_202309.parquet`, ...
- **1ファイルのサイズ：** 300～500 MB 程度

### スキーマ（カラム）

```
インテージ元カラム:
  - watch_date (datetime)
  - channel_2 (str)
  - ... (その他インテージデータ)

Mデータ追加カラム（merge_asof後）:
  - 放送局 (str)
  - シーン開始dt (datetime)
  - シーン終了dt (datetime)
  - ヘッドライン (str)
  - メモ (str)
  - 番組ジャンル (str)
  - 分類 (str)
  - 番組名 (str)
  - topic (str)
  - topic_prob (float)
  - annual_topic_id (str)
  - is_negative (Int64)
```

---

## 処理時間の目安

- **Mデータ読み込み（Cell 3）：** 30 秒～ 1 分
- **月別処理（Cell 5）：** 月あたり 3～5 分
  - Phase 1: 1～2 分（CSV読み込み・振り分け）
  - Phase 2: 2～3 分（merge_asof）
  - Phase 3: 即座（Parquet出力）
- **全12ヶ月処理：** 約 1 時間

---

## 参考資料

### merge_asof の詳細仕様

- **用途：** 非完全一致の時系列マージ。「この時点で、どのカテゴリに属しているか」を特定
- **direction パラメータ：**
  - `'backward'` - キーより前の最新行を探す（本実装）
  - `'forward'` - キーより後の最古行を探す
  - `'nearest'` - 最も近い行を探す
- **tolerance パラメータ：** マッチ可能な最大時間差（オプション）

### PyArrow CSV の利点

- **速度：** pandas.read_csv() の 5～10倍
- **メモリ：** より効率的な内部表現
- **型推論：** 自動的に最適な型を選択

---

## まとめ

本ノートブックは以下の特徴を持つデータ統合パイプラインを実装しています：

1. **メモリ効率性**
   - Phase 1: CSV ストリーミング処理で全月データをメモリに保持しない
   - Phase 2: 放送局別処理で、全局同時保持を回避

2. **処理効率性**
   - ParquetWriter による効率的なファイル追記
   - PyArrow CSV による高速読み込み
   - ガベージコレクション活用

3. **データ正確性**
   - merge_asof による時系列マッチング
   - out_of_range フィルタでシーン終了後の誤マッチを防止
   - datetime 解像度の統一

4. **保守性**
   - 既存ファイルのスキップ（べき等性）
   - 詳細なプログレスバーと統計出力
   - エラーハンドリング（ファイルなし時の処理分岐）

