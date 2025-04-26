# Keiba110 - 競馬予測・投資システム

競馬レースの予測と投資を自動化するシステムです。目標ROI 110%を目指します。

## 機能

- 5つのデータソースからのデータ収集
- 特徴量エンジニアリング（40カラムのParquet特徴量セット）
- 機械学習モデル（LightGBM + XGBoost + Logit → 等温較正 → スタッキングブレンダー）
- ライブベッティングシステム（ROI/CLV確認、3段階ステーキング）
- KPI表示用ダッシュボード

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/keiba110.git
cd keiba110

# セットアップスクリプトを実行
chmod +x setup.sh
./setup.sh
```

## 使用方法

### データ収集

```bash
# レースIDを指定してデータを収集
python main.py 202504260611 --fetch
```

### 特徴量構築

```bash
# 特徴量を構築
python main.py --features
```

### モデルトレーニング

```bash
# モデルをトレーニング
python main.py --train
```

### ライブベッティング

```bash
# レースIDを指定してライブベッティングを実行
python main.py 202504260611 --bet
```

### ダッシュボード

```bash
# ダッシュボードを起動
python main.py --dashboard
```

### 全工程実行

```bash
# レースIDを指定して全工程を実行
python main.py 202504260611 --all
```

## クロンジョブ

`cron_jobs.txt`に記載されているスケジュールに従って、以下のコマンドをcrontabに追加してください：

```
# 金曜
0 9  * * 5  python src/00_fetch/fetch_baba.py
0 23 * * 5  python pipeline_weekly.sh
# 土日
5 6,11 * * 6,7 python src/00_fetch/fetch_baba.py
*/2 7-16 * * 6,7 python src/30_live/live_bet.py
0 17 * * 6,7 python src/40_dashboard/update_kpi.py
```

## テスト

```bash
# テストを実行
python -m unittest discover tests
```

## ライセンス

MIT

## 謝辞

- データソース: netkeiba.com、JRA、JMA
