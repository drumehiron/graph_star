# Graph Star

基本的に本家の実装を使用しています．

https://github.com/graph-star-team/graph_star

本家の実装からの差分は

1. 自分のデータを学習できる形に変形`data.py`
2. sansan データで Link Prediction モデルを作成する `sansan_lp.py`
3. 学習データからそれぞれの辺の Link Prediction Score を出す `use_score.py`

です．

学習データのサイズとして node_num 10^5, edge_num 10^6 程度を想定しています．

GPU は gcp をお借りしました．メモリ数は16Gでした．

## 実行準備

```
$ python --version
3.6.6
$ python -m venv venv
$ source venv/bin/avtivate
$ pip install -r requirement.txt
```

ほか pytorch_geometric の install などは下記 URL を参考にしてください

https://github.com/rusty1s/pytorch_geometric#installation

CUDA の install も必要かと思います．

## Make Own Dataset

### 準備

* 頂点を 0-indexに置き換えられたcsvファイルを`data/rawdata/`配下に用意

* 頂点数の確認．


### data.py の書き換え

```data.py
# node_num
num_nodes = ####

# raw_csv_path 
edge_df = pd.read_csv('data/raw_data/filename.csv')

# node column_name_1,column_name_2
edge_company = torch.tensor(edge_df[["column_name_1","column_name_2"]].values, dtype=torch.long)
```

### 実行

```
$ python data.py
```

処理が終わってるデータとして`sansan_2r.pkl`と`big_company.pkl`を用意してます．

`sansan_2r.pkl`   ：sansan から距離2のデータセット
`big_company.pkl` ：会社，Userの2部グラフデータセットで次数でデータセットを減らしています．

## Make Model

```
$ bash sansan_lp.sh 
```

```
$ bash big_company_sansan.sh 
```

引数の詳細は

https://github.com/graph-star-team/graph_star#options

とほとんど同じですが，`modelname`だけ追加しております．


## User Model

```
$ python use_model.py
```

csvにtestデータのscoreと正解ラベルに保存されてます．


### もくもく会で使用するデータセットを作成するとき

```
MOKUMOKU = True
```

`sansan_all_egde.pkl`      ：sansan から距離2のユーザに対する辺のデータ
`big_company_all_egde.pkl` ：sansan から距離2のユーザに対する辺のデータ

