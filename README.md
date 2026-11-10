# Embedding MCP と CPU 埋め込み実験

llama.cpp を CPU で動かす埋め込み MCP サーバーと、再現可能な比較実験を収めたリポジトリです。MCP から多言語テキストや画像・音声・動画の埋め込みを取得し、アテンションマスクを層・ヘッド単位で変更できます。最初の実測は Linux ARM64 の Neoverse N1 4 コアで行いました。

## MCP サーバー

同期バッチは同じモデルで最大16入力を処理します。さらに多い入力には、途中から再開できる永続ジョブを使えます。フロントは常駐し、CPU ワーカーは必要になったときに起動します。アイドルが5分続くとワーカーを終了し、次の入力処理でモデルをロードします。

| 知りたいこと | 参照先 |
|---|---|
| ローカル起動、MCP クライアントでの確認、ツールとマスクの使い方 | [サービス README](service/README.md) |
| Python と llama.cpp の分担、モデルの寿命、ジョブの復旧、通信の境界 | [MCP サーバー設計](service/DESIGN.md) |
| 別のエージェントが作業を引き継ぐときの入口 | [AGENTS.md](AGENTS.md) |
| 実機で確認した埋め込み、再起動後の復旧、アイドル解放 | [初回の検証記録](results/2026-11-09-embedding-mcp.json) |

## すぐに再実験

Python 3.12、`curl`、Gitを使います。以下はリポジトリのルートで実行します。llama.cppの実行ファイルとモデルは [assets.lock.json](assets.lock.json) に記録したURLとSHA-256で検証します。

```bash
python3 experiments/fetch.py runtime e5
python3 experiments/run.py text e5
```

既存のファイルも再検証します。検証に失敗したファイルは別の場所に移してから、同じ取得コマンドを再実行してください。`EMBED_ASSET_CACHE` に絶対パスを指定すると、複数の作業コピーで重みを共有できます。

## 実験の選び方

| 実験 | 準備 | 実行 |
|---|---|---|
| 日英のテキスト検索と対応文の照合 | `python3 experiments/fetch.py text` | `python3 experiments/run.py text` |
| EmbeddingGemma 2 の画像・音声・動画・混合入力 | `python3 experiments/fetch.py multimodal` | `python3 experiments/run.py multimodal` |
| E5・MiniLM・LaBSEの元モデルとの比較 | `python3 experiments/fetch.py reference` | `uv run --locked python experiments/run.py reference` |
| LASER2・Model2Vec | `python3 experiments/fetch.py nonbert` | `uv run --locked python experiments/run.py nonbert laser2` または末尾を `potion` に変更 |
| MiniLM で日英の語集合を比較し、要素間アテンションを開く層を動かす | `python3 experiments/fetch.py minilm` と [CPU ワーカーの準備](service/README.md#準備)、`g++` | `python3 experiments/set_context.py` |

- テキスト実験のモデルIDは `e5`、`minilm`、`labse`、`gemma2`、`qwen3` です。例えば `text minilm qwen3` で2本に絞れます。取得側は `fetch.py runtime minilm qwen3` を指定します。
- マルチモーダル実験は `ffmpeg` と `ffprobe` を使います。画像・WAV・MP4はスクリプトで生成します。MP4の生成には `libx264` を含むFFmpegが必要です。
- Transformersと非BERTの実験には `uv` が必要です。`uv sync --locked --python /usr/bin/python3` で、Python 3.12とCPU版PyTorchを含む依存関係を準備できます。MiniLMのマスク変更は `reference minilm`、LaBSEのトークン化とDense＋Tanhの切り分けは `reference labse` で実行します。
- 全重みを取得する場合は `python3 experiments/fetch.py all` を使います。ダウンロードは約6.1 GBで、展開後のランタイムとPython環境にも空き容量が要ります。
- MiniLM の集合実験では、[日英18例](fixtures/set-context-ja-en-v1.json)について、先頭から要素間の参照を遮断する層数を0〜12で変えます。語義の順位とフレーズの組み替えの距離を別々に記録します。グラフも保存する場合は `uv run --no-project --with matplotlib==3.10.7 python experiments/set_context.py --plot` を使います。[測定結果](results/2026-11-10-minilm-set-context/summary.json)

## 結果の記録

- [初回の実測](results/2026-11-08-baseline.json) と [再実行の検証結果](results/2026-11-08-reproduction.json) を保存しています。[日英の入力セット](fixtures/retrieval-ja-en-v1.json) は10トピックの小規模な比較用です。テキスト比較では128トークンの上限を指定しているので、新しい入力も短文から試せます。
- 実行ごとに `runs/` に新しいディレクトリを作ります。`metadata.json` にGitコミット、作業ツリーの変更有無、実行コマンド、アセットと入力セットのハッシュ、スレッド数を記録します。`summary.json` は比較用、各実験のJSONとログは詳細確認用です。
- 実験を追加するときは、コード、入力セット、アセットの固定情報と、共有したい計測の `metadata.json` および `summary.json` を `results/<日付>-<実験名>/` に保存してGitにコミットします。入力を変えた比較では、既存の入力セットを残し、同じ形式で別のJSONを作って `EMBED_FIXTURE=fixtures/retrieval-ja-en-v2.json` を指定します。

```bash
python3 experiments/run.py --threads 4 --out runs/e5-repeat text e5
mkdir -p results/e5-repeat
cp runs/e5-repeat/{metadata,summary}.json results/e5-repeat/
git add results/e5-repeat
git commit -m "Record E5 CPU rerun"
```
