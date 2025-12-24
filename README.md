<p align="center"><h1 align="center">Horse Race Prediction</h1></p>
<p align="center">
	競馬のレース結果を予測するnotebook
</p>
<p align="center">
	<a href="./LICENSE">
    <img src="https://img.shields.io/github/license/icchon/Horse" alt="license">
</a>
</p>
<br>

## Contents

- [Overview](#overview)
- [Features](#features)
- [Data](#data)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
- [License](#license)

---

## Overview

netkeiba.comからレース、馬、騎手のデータをスクレイピングし、LightGBMを用いてレースの着順を予測する。分類モデルと回帰モデルの両方でアプローチを試みている。

---

## Features

- **Web Scraping**: `requests`と`BeautifulSoup`を使用してnetkeiba.comからデータを取得。
- **Data Preprocessing**: `pandas`と`numpy`、および`preprocessing_mojule.py`によるカスタム前処理。
- **Feature Engineering**: 過去の成績、血統、レース情報などから特徴量を生成。
- **Machine Learning**: `lightgbm`を使用した分類（`work_clf.ipynb`）および回帰（`work_reg1.ipynb`）モデルの構築。
- **Evaluation**: モデルの性能を回収率やF1スコアなどで評価。

---

## Data

プロジェクトで使用される主要なデータファイル。これらはスクレイピングによって生成される。

- `results.pickle`: レース結果
- `race_infos.pickle`: レース情報（天候、馬場状態など）
- `horse_history.pickle`: 各馬の過去のレース履歴
- `jockey_history.pickle`: 各騎手の過去の成績
- `dict_horse_ped.pickle`: 各馬の血統情報
- `pay_dict.pickle`: レースの払い戻し情報

---

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook / Jupyter Lab
- requirements.txt 参照

### Usage

1.  リポジトリをクローンする:
    ```sh
    git clone https://github.com/icchon/Horse
    cd Horse
    ```
2.  （もしあれば）データ収集用のスクリプトを実行して`.pickle`ファイルを生成する。
3.  Jupyter Notebookを開いて分析・予測を実行する:
    ```sh
    jupyter notebook work_clf.ipynb
    # または
    jupyter notebook work_reg1.ipynb
    ```

---

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

© 2025 icchon
