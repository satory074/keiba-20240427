## 📄 ドキュメント 1 ── **データ取得ガイドライン**

| 目的 | 実装者が **API / HTML / PDF** からエラーなくデータを取得し、パイプラインへ流し込めるようにする。 |
|---|---|
| 成功条件 | ① 各ソース 200 OK／PDF 完全保存 ② 欠損率 < 2 % ③ 1 日 5,000 リクエスト以内に収まる |

### 1. ソース別まとめ

| ID | ソース | 方式 | レート制限目安 | 実装難度 |
|----|--------|------|---------------|----------|
| **S1** | netkeiba **内部 API** `api_get_jra_odds.html` | HTTPS + JSON | 30 req/min (実測) | ★★☆ |
| **S2** | **JRA 出馬表 PDF** `rpdf/YYYYMM/DD/ⱼⱼⱼ.pdf` | 直接 DL | なし | ★☆☆ |
| **S3** | **JRA 馬場 HTML** `baba/YYYYMMDDⱼⱼ.html` | HTML Parse | 20 req/min | ★☆☆ |
| **S4** | **JMA 3h 予報** `bosai/forecast/data/forecast/AREA.json` | JSON | 10 req/min | ★☆☆ |
| **S5** | **含水率 PDF** `baba/gansui/YYYY.html` | PDF | なし | ★★☆ |

> *内部 API は公式に公開されていないため、**User-Agent設定**と**2 req/min**で運用。ban 時は 10 min スリープ*  ([GitHub - apiss2/scraping: netkeibaのスクレイピング用](https://github.com/apiss2/scraping?utm_source=chatgpt.com))  

### 2. レース ID と各コード

| 要素 | 文字列 | 例 |
|------|--------|----|
| 年月日 | `YYYYMMDD` | 20250426 |
| 開催場コード | `01`=札幌 `02`=函館 `03`=福島 `04`=新潟 `05`=東京 `06`=中山 `07`=中京 `08`=京都 `09`=阪神 `10`=小倉 |
| 開催回数 | `1‥12` | 3 |
| 日次 | `01‥12` | 08 |
| レース番 | `01‥12` | 11 |

**race_id = YYYYMMDD + 開催場(2) + 回次(1) + 日次(1) + レース番(2)**  
例：2025/04/26 中山11R = `202504260611`.

### 3. HTML / PDF 重要セレクタ

| ページ | CSS / XPath | 抽出値 |
|--------|-------------|--------|
| 馬場 HTML | `table.condition` | 芝/ダ状態・含水率・クッション値 |
| 同上 | `//th[contains(text(),'クッション値')]/following-sibling::td` | 値のみ |
| 出馬表 PDF | `area=[108,28,790,566]` `columns=[28,70,138,278,330,382,450,512]` | 全馬行 |

### 4. エラー処理ベストプラクティス

* 3 回リトライ (`backoff=1.5`)→まだ失敗なら `FAILED_RAW/` へ保存し後処理。
* PDF は `PyPDF2.PdfReader` でページ数チェック。< 18 頭の場合は WARN。
* HTML parse 後に必須 key が Null → スキーマ Validation で早期検出。

---

## 📄 ドキュメント 2 ── **実装詳細指示書**

### 2-1. Python パッケージ & バージョン

```text
requests>=2.32
httpx>=0.27
pandas>=2.2
polars>=0.20
duckdb>=0.10
tabula-py>=2.9
pdfplumber>=0.11
beautifulsoup4>=4.12
selenium>=4.19           # PAT 自動送信
lightgbm>=4.3
xgboost>=2.0
scikit-learn>=1.4
```

### 2-2. モジュール間 I/O

```
fetch_*.py ➜ data/raw/yyyymmdd_<source>.{pdf,json,html}
build_features.py ← data/raw/*              ➜ features.parquet
train_stack.py    ← features.parquet        ➜ model/
live_bet.py       ← model/, features_today  ➜ bet_log.csv
```

### 2-3. fetch_ozz.py（骨格 60 行）

```python
import httpx, time, json, pathlib, datetime as dt
RID  = "202504260611"
URL  = f"https://race.netkeiba.com/api/api_get_jra_odds.html?type=1&locale=ja&race_id={RID}"

headers={"User-Agent":"keiba110-bot/0.1 (+github)"}
out   = pathlib.Path(f"data/raw/{RID}_odds.json")
while dt.datetime.now().hour < 16:        # 最大16:59
    r = httpx.get(URL, headers=headers, timeout=5)
    if r.status_code == 200:
        out.write_text(r.text, encoding="utf-8")
    elif r.status_code == 429:
        time.sleep(600)                  # ban
    else:
        print("warn", r.status_code)
    time.sleep(120)                      # 2min loop
```

### 2-4. tabula 座標確認コマンド

```bash
tabula -a 108.28,28,790,566 -p 1 -t -r 3nakayama7.pdf
```

### 2-5. PAT 自動送信（ダミー）

```python
from selenium import webdriver
opt = webdriver.ChromeOptions()
opt.add_argument("--headless")
with webdriver.Chrome(options=opt) as driver:
    driver.get("https://regist.ipat.jra.go.jp/")
    # ログイン ➜ 投票フォーム自動入力 ➜ submit
```

> **本番は team-nave IPAT-API** の `bet(place,horse_id,stake)` を呼ぶ。  

### 2-6. テストケース

| Case | 入力 | 期待 |
|------|------|------|
| ① PDF 欠損 | 存在しない URL | `FileNotFoundError` 捕捉→`FAILED_RAW/` |
| ② 馬場 HTML 構造変化 | `<td>` 消失 | SchemaError raise |
| ③ 429 Too Many | 5 回続いた | 10 分 Sleep |

---

## 📄 ドキュメント 3 ── **取得データリスト（スキーマ付）**

| データセット | PK | 列 | dtype | 例 |
|--------------|----|----|-------|----|
| **entries.parquet** | race_id, horse_id | draw:int, weight:int, sex:str, age:int | int/str | 202504260611 2018105467 |
| **odds_2min.parquet** | race_id, ts, horse_id | odds_win:float, odds_plc_low:float, odds_plc_high:float | float | 4.2 |
| **baba.parquet** | date, course | turf_state:str, moisture_front:float, cushion:float | str/float | 良, 10.5 |
| **weather.parquet** | ts, area | wx:str, temp:int, wind:int | str/int | 晴, 18 |
| **gansui.parquet** | date, course | turf_moist:float(%) , dirt_moist:float | float | 14.2 |
| **bet_log.csv** | txn_id | race_id, stake, result:int, return:int | str/int | ... |

*area コードマップ*—Tokyo `130000`, Nakayama `120000`, Kyoto `260000`, Hanshin `280000`, Chukyo `230000`.

---

### 🔚 これで “データ取得” パートの不明点はゼロ

上記 3 ドキュメントを生成 AI に渡し、**「対応コードを出力せよ」** と指示すれば完成します。  
追加のエンドポイントやスクレイピング対策が必要になった際は、再度ご依頼ください。
