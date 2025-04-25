以下は **コード一式が生成済み** という前提で、  
「初回セットアップ ▶︎ 毎週の仕込み ▶︎ レース当日の運用 ▶︎ 予測と成績の確認」  
の４ステップだけに絞った **実行マニュアル** です。  
（Mac／Linux／WSL／Windows PowerShell 共通のコマンドを示します）

---

## 1 ️⃣ 初回だけ──環境とデータのセットアップ

```bash
# リポジトリを取得
git clone <your-fork>/keiba110.git
cd keiba110

# Python & 主要ライブラリを一括インストール
bash setup.sh
```

> *set `PYENV_ROOT`/path が通っていない場合は、ターミナルをいったん再読込してください。*  

---

## 2 ️⃣ 毎週金曜（仕込み）── 出馬表・馬場データ取得→モデル再学習

```bash
# 例：5 月 2 日（金）開催週を仕込み
python pipeline_weekly.py 20250502
```

内部で次を自動実行します  

1. `src/00_fetch/fetch_entries.py` — 週末開催すべての PDF を DL  
2. `fetch_baba.py` — 金曜 AM 時点の芝／ダート状態・含水率を取得  
3. `fetch_weather.py` — 週末２日分の３時間予報を取得  
4. `build_features.py` — 40 特徴量を生成 → `features.parquet`  
5. `train_stack.py` — Stacking + Isotonic でモデルを再学習 → `model/`

> **完了目印** ：ターミナルに  
> `✅ weekly pipeline finished (model saved to model/stack.pkl)`  
> が表示される。

---

## 3 ️⃣ レース当日（土日）── リアルタイム推論＆（自動）投票

### 3-1 全自動モード（推奨）

```bash
python live_bet.py
```

* 06:05 / 11:05 に馬場情報を再取得  
* 07:00 ～ 16:59 まで **２分おき** に  
  * netkeiba オッズ API → `odds_2min.parquet` 更新  
  * モデル確率 **P̂** → ROI・CLV 計算  
  * 三段階ステーク（300 円 / 100 円 / ケン）を決定  
  * `PAT` Web を Selenium で自動送信  
  * 結果は `data/bet_log.csv` へ随時追記  

#### ログ確認

```bash
tail -f logs/live_bet.log
```

```
09:32  [BUY] 202504260611  単勝  2番 300円  ROI=0.34 CLV=+0.18
09:34  [KEN] 202504260612  条件外
...
```

### 3-2 手動モード（オッズだけ確認して自分で投票）

```bash
python live_bet.py --dry-run
```

*コンソール出力に「推奨買い目」と「理由（ROI / CLV）」だけを表示し、PAT は操作しません。*

---

## 4 ️⃣ 予測結果・KPIを確認する

### 4-1 Streamlit ダッシュボード

```bash
streamlit run src/40_dashboard/dashboard.py
# http://localhost:8501  が開く
```

* **回収率・CLV・MAE・RoR** を四象限チャートでリアルタイム表示  
* 赤信号ゾーンに入ると画面上部にアラート  

### 4-2 CSV / Parquet を直接覗く

```bash
duckdb
duckdb> SELECT * FROM data/bet_log.csv WHERE date='2025-05-03';
duckdb> SELECT AVG(roi) FROM kpi_daily;
```

---

## 5 ️⃣ よくある質問（F.A.Q.）

| Q | A |
|---|---|
| **PDF が読めない** | `tabula-java` が見つからない → `JAVA_HOME` を要確認 |
| **オッズ取得が 429** | 10 分スリープ後に自動再開。急ぐ場合は `live_bet.py --cooldown 900` で延長可 |
| **PAT 画面が変わり投票失敗** | `src/30_live/pat_driver.py` 内の `XPATH` を修正 or 有料 **IPAT-API** に切替 |
| **モデル再学習の目安は?** | ダッシュボード「MAE > 0.05 or CLV ≤ 0」の赤信号が出たら `pipeline_weekly.py` を手動再実行 |

---

## 6 ️⃣ アンインストール／クリーンアップ

```bash
rm -rf keiba110/.venv  data/  model/  logs/
pyenv uninstall 3.11.7
```

---

### これで完了です

1️⃣ **setup.sh** → 2️⃣ **weekly_pipeline.py** → 3️⃣ **live_bet.py**  
の３コマンドだけ覚えておけば、毎週自動で予測と投票が回ります。  
疑問が出たら `logs/` をチェックし、必要ならパラメータ（閾値・ケリー係数など）を調整してください。
