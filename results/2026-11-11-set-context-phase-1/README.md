# 語・句の集合における要素間アテンションの第一段階実験

実施日 2026年11月10〜11日（JST）。日英の語・句を順序のない集合として入力し、各要素の表現が周囲の要素によってどの程度変わるかを測った。別の要素を参照できるようにする層を一層ずつ動かし、文脈に合う語義の順位と、属性や動作の組み替えを区別する能力を調べた。

4モデルとも、同じ集合を入力順だけ変えて渡したときは同じベクトルを得た。語義判定は MiniLM・E5・EmbeddingGemma 2 で先頭の数層を遮断した条件が良く、Qwen3 では全層で要素間を許可する条件が最良だった。4モデルに共通する組み替え6例では、全80境界条件で、元の語を組み替えた集合の方が、意味を保った言い換えより近かった。これは少数の作例を使った探索実験であり、モデル間の順位は語義参照の作り方にも依存した。

## 予備実験

1. **シーケンス分離と位置指定。** 上流 llama.cpp b12110 の C API で「白い猫」と「黒い犬」を異なるシーケンス ID に入れ、位置番号を各シーケンスで振り直した。4モデルでトークン出力とシーケンス単位のプーリングを取得でき、片方のシーケンスで長さを保ってトークンを置換しても、もう一方の出力は変わらなかった。物理的な投入順の変更では数値差が生じたが、トークン列を正規順に並べて位置を1始まりにした条件では一致した。この検査はモデル本来のシーケンス間遮断を使った。 [ソース](evidence/layout-probe.cpp) · [出力](evidence/layout-results.jsonl)
2. **独立埋め込みの基準。** MCP の `embed_set` で、要素ごとに独立に取得した埋め込みを等重みで平均し、L2 正規化した。順番を逆転した出力は全モデルで一致した。「白い猫・黒い犬」と「白い犬・黒い猫」のコサイン類似度は MiniLM 0.985192、E5 0.999760、EmbeddingGemma 2 0.999488、Qwen3 0.996375 だった。E5 の入力には `passage: ` を付けた。この基準は各モデルの通常のプーリングを使うため、後述するトークン平均や本測定の全層遮断条件とは別の測定である。MCP の作業ツリーは実行時に変更を含んでおり、結果にサービスソースの SHA-256 を記録した。 [呼び出しコード](evidence/mcp-probe.py) · [出力と実装の識別子](evidence/mcp-baseline.json)
3. **前半遮断・後半開放。** 本測定と同じ要素内位置と層別マスクを用いて「ジャガー」「マウス」「bank」および色と動物の組み替えを測った。MiniLM の「ジャガー」は、動物側と車側のトークン表現のコサイン類似度が全層遮断で 0.999999、先頭6層だけ遮断で 0.901545 だった。一方、色と動物の組み替えの類似度は同じ二条件で 0.987514 から 0.999663 に上がった。この観察を受け、語義と組み替えを分けて全境界を測ることにした。 [ソース](evidence/half-depth-probe.cpp) · [4モデルの出力](evidence/half-depth-results.jsonl) · [メタデータ](evidence/half-depth-metadata.json)

## 本測定の条件

| モデル | 固定した GGUF | 層数 | 出力次元 | 遮断層数 $k$ |
|---|---|---:|---:|---|
| paraphrase-multilingual-MiniLM-L12-v2 | F16 | 12 | 384 | 0〜12 |
| multilingual-e5-small | F16 | 12 | 384 | 0〜12 |
| EmbeddingGemma 2 | Q8_0 | 24 | 768 | 0〜24 |
| Qwen3-Embedding-0.6B | Q8_0 | 28 | 1,024 | 0〜28 |

重みの取得元、サイズ、SHA-256 は [assets.lock.json](../../assets.lock.json) と各実行メタデータに記録した。Linux ARM64（Neoverse-N1）、CPU 4スレッド、コンテキスト・バッチ・マイクロバッチ各128トークンで実行した。Flash Attention とランタイムのプーリングは無効にした。本測定は llama.cpp b12110 のソースコミット `cd959bae99ff4bed22bef2604f45241c09bbb8c8` に [CPU ワーカー用パッチ](../../service/native/) を適用した `0.9.0-dev+local.6a9045a9e9b2` を使用した。ワーカー成果物 ID は `f0e07667d8513b72d31c90a17327952b60a7f7535f6f8b34df2e42931be82bd4`。E5 はこの測定ではタスク用プレフィックスを付けなかった。 [MiniLM のメタデータ](../2026-11-10-minilm-set-context/metadata.json) · [他3モデルのメタデータ](../2026-11-11-other-model-boundaries/experiment-metadata.json)

入力は [日英の評価用データ](../../fixtures/set-context-ja-en-v1.json) を使用した。語義は日本語5語（ジャガー、マウス、コート、カキ、ライト）と英語5語（bank、bat、seal、spring、bark）。各語に二種類の文脈があり、それぞれ対象語と同じ語義を示唆する2語の集合、および評価用の定義文を用意した。組み替えは色と物体、主体と動作、主体と位置などを含む日英8例で、各例に元の集合、組み替え、意味を保った言い換えを用意した。

組み替えでは、元の集合と組み替え後の集合の `(要素内位置, トークンID)` の多重集合が一致することを、そのモデルのトークナイザーで検査した。これにより、トークンの種類や位置の違いを取り除いてフレーズへの所属だけを変えられる。MiniLM と E5 は8例すべてで一致した。候補8例のうち Qwen3 は7例、EmbeddingGemma 2 は6例が一致したため、この2モデルの本測定には [4モデルに共通する6例](../2026-11-11-other-model-boundaries/fixture.json) を使った。共通6例は日本語2例（動作、位置）と英語4例。「白い猫・黒い犬」の組み替えは共通比較に含まれない。

## 位置、アテンション、出力ベクトル

各要素を独立にトークン化し、自動付加される特殊トークンを含め、要素内で位置を0から振った。要素を文字列順に整列し、各位置のトークンを安定ソートでまとめた。たとえば二つの要素が3トークンずつなら、バッチ上は `A0, B0, A1, B1, A2, B2`、位置番号は `0, 0, 1, 1, 2, 2` になる。本測定では全トークンをシーケンス ID `0` に入れ、バッチの行番号から要素の所属を追跡した。

層番号を0始まりとし、$0\le\ell<k$ の全ヘッドで別要素への参照を遮断した。残る層 $\ell\ge k$ にはモデル本来のマスクを適用した。$k=0$ は追加遮断なし、$k=L$ は全 $L$ 層で要素間を遮断する条件である。Qwen3 では、後半でもキーの位置がクエリの位置を超える場合は因果マスクで遮断される。同位置の別要素は開放層で参照可能である。EmbeddingGemma 2 では元の注意範囲を維持した。実行時には選んだトークン対とヘッドについて、境界前後の参照可否と遮断エッジの重みが0であることを確認した。 [MiniLM の実装](../../experiments/set_context.cpp#L74) · [Qwen3 の実装](../2026-11-11-other-model-boundaries/qwen3.cpp#L74)

最終層のトークン出力を使い、各要素の文字列本体に当たるトークンの平均を L2 正規化した。集合ベクトルは、これらの要素ベクトルを等重みで平均して再度 L2 正規化した。語義判定には集合全体のベクトルではなく、対象語の要素ベクトルを使った。Qwen3 の標準的な文章埋め込みは末尾 `[EOS]` を読むため、本測定の対象語トークン平均とは読み出す位置が異なる。 [Qwen3 Embedding 技術報告](https://arxiv.org/html/2506.05176v3)

## 評価方法

**語義判定。** たとえば `{ジャガー, ヒョウ, チーター}` と `{ジャガー, BMW, ベンツ}` から対象語のベクトル $u_A,u_B$ を取得した。同じモデルで「動物」と「車」の定義を使って参照ベクトル $r_A,r_B$ を作り、$m_A=\cos(u_A,r_A)-\cos(u_A,r_B)$、$m_B=\cos(u_B,r_B)-\cos(u_B,r_A)$ を求めた。両方が厳密に正なら1語正解とし、同点は不正解とした。文脈による移動量には $1-\cos(u_A,u_B)$ を記録した。これは語義判定と別の尺度である。

| 参照 | 作り方 |
|---|---|
| `definition`（主評価） | `ジャガー : 食肉目ネコ科の大型哺乳類` を一つの文字列として入力し、対象語のトークンだけを平均。Qwen3 は因果マスクで対象語から定義を参照できるように `定義 : 対象語` とした。末尾の対象語を抽出する際は、空白を含む接尾部のトークン列を使った。 |
| `legacy`（参照方式への感度） | `{ジャガー, 食肉目ネコ科の大型哺乳類}` を二要素として処理し、対象語だけを平均。参照を作るときの $k$ は各モデルの半分（MiniLM と E5 は6、EmbeddingGemma 2 は12、Qwen3 は14）。 |

どちらの方式でも各モデルの参照を一度作り、評価側の全 $k$ で共用した。参照も評価対象も同じモデルから得たため、正解数はこの手順での語義識別を表す。定義の意味に対応するベクトルを独立に与えた評価ではない。

**組み替え。** 元の集合、組み替え、言い換えのベクトルを $v,v_{\rm swap},v_{\rm para}$ とし、組み替え距離を $1-\cos(v,v_{\rm swap})$、順位のマージンを $\cos(v,v_{\rm para})-\cos(v,v_{\rm swap})$ とした。マージンが正なら、語を共有する組み替えより言い換えを上位に置いたことになる。この判定は元の集合を基準とする片方向の比較である。

**順序。** 2要素は2順列、3要素は全6順列を正規順に変換して実行し、集合および要素のベクトルの完全一致を検査した。元の投入順を物理バッチの同位置トークンの順に残す対照では、逆順とのコサイン距離も記録した。正規順での一致は、入力整列を含む手順についての結果である。

## 実測結果

| モデル | 語義正解数 $k=0$ | `definition` の最大値と $k$ | 最良条件の日英内訳 | 同じ $k$ での `legacy` | `legacy` の最大値と $k$ |
|---|---:|---|---|---:|---|
| MiniLM | 5/10 | 8/10、2〜6 | 3/5・5/5 | 9/10 | 9/10、2〜10 |
| E5 | 7/10 | 9/10、2 | 4/5・5/5 | 8/10 | 8/10、2〜3 |
| EmbeddingGemma 2 | 6/10 | 9/10、1〜2 | 4/5・5/5 | 3/10 | 7/10、12 |
| Qwen3 | 6/10 | 6/10、0 | 3/5・3/5 | 2/10 | 5/10、2〜4 |

同点の最大値が複数の境界にあるとき、内訳と `legacy` との同一境界比較には最小の $k$ を使用した。全層遮断では両方式とも全モデル0/10語だった。主評価の最良条件では、MiniLM は「マウス」「ライト」、E5 は「ライト」、EmbeddingGemma 2 は「カキ」、Qwen3 は「ジャガー」「カキ」「bank」「bat」の両方向判定を満たさなかった。 [MiniLM の全条件](../2026-11-10-minilm-set-context/summary.json) · [E5](../2026-11-11-other-model-boundaries/e5-common-summary.json) · [EmbeddingGemma 2](../2026-11-11-other-model-boundaries/gemma2-common-summary.json) · [Qwen3](../2026-11-11-other-model-boundaries/qwen3-common-summary.json)

| モデル | 組み替え距離の中央値 $k=0$ | $k=L/2$ | $k=L$ | 正規順で検査した入力順 |
|---|---:|---:|---:|---:|
| MiniLM | 0.00001950 | 0.00123097 | 0.00804439 | 2,496 |
| E5 | 0.00000123 | 0.00053115 | 0.00152007 | 2,496 |
| EmbeddingGemma 2 | 0.00015636 | 0.00331437 | 0.00437565 | 4,200 |
| Qwen3 | 0.00000423 | 0.00560963 | 0.00689799 | 4,872 |

距離は共通6例、入力順の検査数は各モデルの全入力（各集合の基準順を含む）。共通6例と全80境界を合わせた480比較で、組み替え距離は各例の物理的な逆順による数値差を上回った。言い換えを上位に置く比較は0/480で、MiniLM と E5 の追加2例を含めても0/532だった。MiniLM は本測定を2回実行し、境界ごとの JSONL 出力がバイト単位で一致した。他3モデルについては全境界の完走と順列チェックを確認した。

物理的な逆順を試した入力での集合ベクトルのコサイン距離は、モデルごとの最大値が MiniLM $1.13\times10^{-6}$、E5 $7.45\times10^{-7}$、EmbeddingGemma 2 $2.11\times10^{-4}$、Qwen3 $1.18\times10^{-4}$ だった。組み替えとの比較には、モデル全体の最大値ではなく、同じ例・同じ $k$ の逆順との差を使った。

| モデル | 記録した経過時間 | 計測範囲 |
|---|---:|---|
| MiniLM | 42.1秒 / 47.9秒 | 初回は測定プロセス全体。再実行はコンパイル・推論・集計・グラフ生成を含む。再実行の推論は約36秒。 |
| E5 | 44.1秒 | 準備・コンパイル約9秒、推論約35秒、集計約0.2秒。内訳はファイル時刻からの概算。 |
| EmbeddingGemma 2 | 452.7秒 | モデル読み込み、参照の作成、全境界の推論。 |
| Qwen3 | 762.7秒 | 修正後のモデル読み込み、参照の作成、全境界の推論。 |

MiniLM の時間は実行ログと出力ファイル時刻、他3モデルは [実行時刻の記録](../2026-11-11-other-model-boundaries/timing-journal.log) に基づく。EmbeddingGemma 2 と Qwen3 は途中までCPUを共有し、Qwen3 のCPU割り当て上限は途中で1コア相当から4コア相当へ変更した。Qwen3 の最初の試行は参照文の対象語が単独入力と異なるトークン列になったため失敗した。上表は修正後の成功試行の時間である。各境界の個別時間と、後半2モデルの準備・集計時間は未計測であり、この表はモデル間の速度比較には使えない。

## 論文で扱う際の制約

- **少数の作例と境界の選択。** 10語と8組の手作りの例で、境界の選択と評価に同じ語を使った。最高値はこの入力における探索結果である。日英の内訳も各5語なので、言語差の推定には独立した評価集合が必要になる。
- **参照と読み出しの交絡。** 参照の形式を替えるだけで語義正解数が変わった。特に Qwen3 では因果マスクのため定義の語順を変更し、対象語の位置とトークン化も他モデルと異なる。Qwen3 の因果性、学習法、参照方式の影響はこの測定だけでは分離できない。E5 の入力プレフィックス、各モデルの量子化形式、対象語のトークン平均も比較条件に含まれる。
- **順序不変性と結び付きの区別。** 正規順での完全一致は実装の性質であり、要素間マスクだけによる順序不変性を実証したものではない。組み替えを数値上区別することと、意味を保つ言い換えを上位に置けることも別である。後者の結果には言い換えの語彙差も含まれる。

## 再解析と再実行

| 対象 | 実装・入力 | 解析用データ |
|---|---|---|
| MiniLM（コミット `2a54bbd`） | [ランナー](../../experiments/set_context.py)、[C++](../../experiments/set_context.cpp)、[18例](../../fixtures/set-context-ja-en-v1.json) | [境界別の出力](evidence/minilm-raw.jsonl)、[集計](../2026-11-10-minilm-set-context/summary.json) |
| E5（`0711205`） | [ランナー](../2026-11-11-other-model-boundaries/set_context.py)、[C++](../2026-11-11-other-model-boundaries/set_context.cpp)、同じ18例 | [境界別の出力](../2026-11-11-other-model-boundaries/e5-raw.jsonl)、[全8例の集計](../2026-11-11-other-model-boundaries/e5-summary.json) |
| EmbeddingGemma 2・Qwen3（`0711205`） | [Gemma C++](../2026-11-11-other-model-boundaries/gemma2.cpp)、[Qwen C++](../2026-11-11-other-model-boundaries/qwen3.cpp)、[10語と共通6例](../2026-11-11-other-model-boundaries/fixture.json) | [Gemma の境界別出力](../2026-11-11-other-model-boundaries/gemma2-raw.jsonl)、[Qwen の境界別出力](../2026-11-11-other-model-boundaries/qwen3-raw.jsonl) |

JSONL は境界ごとに1行で、`k` が遮断する先頭層数。`sense[].definition.margins` と `sense[].legacy.margins` は文脈A・Bの正解マージン、`binding[].swap_distance` は組み替え距離、`binding[].ranking_margin` は言い換えと組み替えの順位差である。共通6例の距離中央値を求めるには、各行の `binding` を共通入力ファイルの `binding[].id` で絞る。ベクトル成分の代わりに評価値を保存しているため、新しい距離関数で成分を使う場合は再推論が必要になる。

[CPU ワーカーの準備手順](../../service/README.md#準備)に従って重みとパッチ済みランタイムを用意し、リポジトリのルートから実行する。再ビルドしたワーカーの成果物 ID が上記と異なる場合は、その ID と実行環境を新しい測定に記録する。

~~~bash
python3 experiments/fetch.py minilm e5 gemma2 qwen3
python3 experiments/set_context.py --out runs/phase1-minilm-repeat

phase_models=results/2026-11-11-other-model-boundaries
PYTHONPATH="$PWD/experiments" python3 "$phase_models/set_context.py" \
  --out runs/phase1-e5-repeat

phase_worker=$(realpath .assets/embedding-worker/current)
phase_patch=$(jq -r .patch_sha256 "$phase_worker/manifest.json")
phase_headers="$PWD/.assets/mcp-build/sources/$phase_patch"
phase_output=$(mktemp -d "$PWD/runs/phase1-gemma-qwen-repeat.XXXXXXXX")
for phase_model in gemma2 qwen3; do
  phase_gguf=$(jq -r --arg model "$phase_model" '.groups[$model][0].path' assets.lock.json)
  g++ -std=c++17 -O2 -I"$phase_headers/include" -I"$phase_headers/ggml/include" \
    -I"$phase_headers/vendor" "$phase_models/$phase_model.cpp" \
    "$phase_worker/libllama.so" -Wl,-rpath,"$phase_worker" -o "$phase_output/$phase_model"
  "$phase_output/$phase_model" ".assets/$phase_gguf" "$phase_models/fixture.json" \
    > "$phase_output/$phase_model.jsonl"
done
~~~

Gemma と Qwen の再実行例は順次実行する。本測定と異なりCPUを同時に共有しないため、経過時間の再現には別途スケジューリング条件をそろえる必要がある。予備実験の保存物は [SHA-256 一覧](evidence/SHA256SUMS) で確認できる。MCP の予備実験コードには当時の作業ツリーの絶対パスが含まれる。実験の再解析には保存済みの JSON を使え、別ホストでの再実行にはパスと対応するサービス実装の確認が必要になる。

## 関連研究

- Milbauer らの LAIT（ACL 2023）は、12層の T5 で独立に処理する先頭層数を0〜12まで変え、WiC を含む課題で比較した。境界ごとにタスク用に調整した分類モデルを評価している。 [LAIT、表2](https://aclanthology.org/2023.acl-long.571.pdf#page=6)
- MacAvaney らの PreTTR（SIGIR 2020）、Cao らの DeFormer（ACL 2020）、Vast らの MICE（2026）は、検索または質問応答で、二つの入力が相互参照を始める層を比較した。MICE には12層の MiniLM 系の再順位付けモデルを使った測定がある。 [PreTTR](https://arxiv.org/abs/2004.14255) · [DeFormer](https://aclanthology.org/2020.acl-main.411/) · [MICE v4](https://arxiv.org/abs/2602.16299v4)
- Egressy と Stühmer の Set-LLM（NeurIPS 2025）は、要素内位置を保ちながら集合の順序に依存しないモデルを構成した。図4では同じ位置付きトークンから異なる要素の結び付きを区別するために要素間マスクを使う。 [Set-LLM、図4](https://proceedings.neurips.cc/paper_files/paper/2025/file/5abdb0ae08d5a00fbdcdc187178276ac-Paper-Conference.pdf#page=5)
- Lee らの Set Transformer（ICML 2019）は、要素間のアテンションを経てから順序に依存しない出力へ集約する。Ethayarajh（EMNLP-IJCNLP 2019）は、異なる文脈に現れた同じ語の表現をコサイン類似度で比較した。 [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) · [文脈化された語表現の分析](https://aclanthology.org/D19-1006/)
- 組み替えの評価には、語彙を保って属性と対象の対応を変える SABER と、同義の言い換えと組み替えを比較する SugarCrepe++ のテキスト同士の評価が近い。本実験の順位マージンは元の集合を基準とする片方向で、SugarCrepe++ の指標は両方向を確認する。 [SABER](https://arxiv.org/abs/2610.22602) · [SugarCrepe++](https://arxiv.org/html/2406.11171#S3)
