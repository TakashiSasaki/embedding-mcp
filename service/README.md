# CPU 埋め込み MCP

MCP の入口は `127.0.0.1:18765/mcp` の Streamable HTTP です。Python がジョブとメディアを管理し、llama.cpp が埋め込みを計算します。ワーカーとの通信にはサービスユーザー専用の Unix ソケットを使います。

## 準備

Linux ARM64、Python 3.12、Git、CMake、C/C++ コンパイラ、make、curl、FFmpeg、uv が必要です。リポジトリのルートで実行します。

```bash
python3 experiments/fetch.py e5 minilm qwen3 gemma2 gemma2-media
uv sync --project service --locked --python /usr/bin/python3
python3 service/build_worker.py
service/.venv/bin/python service/main.py --version
service/.venv/bin/python service/main.py
```

`build_worker.py` は [b12110 の固定ソース](native/upstream.json) を SHA-256 で検証し、[パッチ](native/) を適用して CPU 向けにビルドします。実行ファイルとライブラリのハッシュをリリースに記録し、`.assets/embedding-worker/current` を更新します。実行中のサービスは起動時に選んだリリースを使います。更新はフロントを再起動して反映します。モデルの置き場所には実験と同じ `EMBED_ASSET_CACHE` を指定できます。

`systemd --user` で常駐させる場合は、手動起動を終了してからユニットを設置します。

```bash
install -Dm644 service/deploy/embedding-mcp.service ~/.config/systemd/user/embedding-mcp.service
systemctl --user daemon-reload
systemctl --user enable --now embedding-mcp.service
curl --fail http://127.0.0.1:18765/health
```

## ツール

| ツール | 用途 |
|---|---|
| `list_models` | モデル、次元数、トークン上限、層数、クエリヘッド数、対応モダリティ |
| `runtime_status` | ロード状態、PID、RSS、空きメモリ、最後のアンロード理由、キュー |
| `inspect_input` | 入力を展開した後の位置表と `layout_id` |
| `embed` | 1 入力のベクトル |
| `embed_batch` | 同じモデルで最大 16 入力。入力順に結果か項目エラーを返す |
| `submit_embedding_job` | 同じモデルで最大 1,000 入力の永続ジョブを登録 |
| `embedding_job_status` | 完了項目数と期限を確認 |
| `embedding_job_results` | 完了済みの結果をカーソルで取得。1 ページ最大 32 項目 |
| `cancel_embedding_job` | 処理中の項目が終わった時点で停止 |
| `register_media` | PNG、JPEG、PCM WAV、H.264 MP4 を登録し、内容の SHA-256 に基づく ID を取得 |

テキストは文字列で指定します。順序付き入力は `{"parts":[{"type":"text","text":"赤い画像"},{"type":"image","media_id":"media_…"}]}` です。メディアの `type` は `image`、`audio`、`video` のいずれかです。Gemma 2 がメディアに対応します。登録は 1 件 4 MiB、画像の各辺は 2,048 px、WAV は 16-bit PCM で最長 20 秒、MP4 は最大 1,920×1,080・60 fps・10 秒です。1 入力につき最大 4 パート、計 8 MiB までです。登録内容は再起動をまたいで保持し、合計上限は 512 MiB です。

`embed_batch` と `submit_embedding_job` の `inputs` には文字列、`parts` オブジェクト、または項目固有のマスクを含む `{"input":…,"mask":…}` を混在できます。ベクトルは `float` または little-endian float32 の `f32_base64` です。モデルのプーリングと L2 正規化を使います。E5 の `query: ` や `passage: ` などは呼び出し側で付けます。

### マスクの例

まず同じ `model` と `input` を `inspect_input` に渡します。取得した `layout_id` を以下の `mask` に入れます。例では 0 層目の 0 番クエリヘッドで、位置 0 から位置 1 への辺を閉じます。

```json
{
  "model": "minilm",
  "input": "埋め込みを比較します",
  "mask": {
    "layout_id": "<inspect_input の layout_id>",
    "rules": [{
      "action": "block",
      "layers": [0],
      "heads": [0],
      "query": {"kind": "range", "start": 0, "end": 1},
      "key": {"kind": "range", "start": 1, "end": 2}
    }]
  }
}
```

- `allow` はモデルが閉じていた辺も開きます。複数のルールが重なると最後のルールが適用されます。ルールなしではモデル既定のマスクを使います。
- 層とクエリヘッドは 0 始まりで、一覧または `"all"` を指定します。位置範囲は半開区間で、BOS/EOS などの制御トークンも位置に数えます。生の範囲を使う場合は `layout_id` が必須です。
- 範囲のほかに `{"kind":"all"}`、`{"kind":"modality","modality":"image"}`、`{"kind":"part","part":1}` で指定できます。連続するテキストパートは連結して一度にトークン化し、その範囲に各パート番号を付けます。メディア由来の制御トークンには `control` と元のパート番号を付けます。動画の展開後もフレームを元のパート番号で選択できます。
- マスクはメディア展開後の埋め込みバックボーンに適用します。全てのキーが閉じた行、範囲外の座標、パディング後に 256 MiB を超える追加マスクは項目エラーになります。

### ロードとジョブ

フロントの起動時点でワーカーは空です。入力の検査、同期埋め込み、またはジョブの最初の項目を処理するときにモデルをロードします。重みは初回ロード時にハッシュを検証します。ワーカーは 1 プロセス・1 スロット・CPU 4 スレッドで、同じモデルを使う間は再利用します。Gemma 2 のテキスト用ワーカーはメディアが必要になると projector 付きに切り替わり、そのワーカーはテキストにも使えます。

5 分間アイドルになるか、ホストの `MemAvailable` が 3 GiB 未満になった場合に、実行中の項目の境界でワーカーを終了します。メモリが回復するとバックグラウンドジョブが再開します。ステータス取得と結果取得はワーカーを起動しません。同期呼び出しはキュー待ち・ロードを含めて 80 秒です。非同期ジョブ処理中も項目の境界で同期呼び出しを優先します。

ジョブは `~/.local/state/embedding-mcp/embeddings.sqlite3` に保存します。同じキーと同じ入力を再送すると同じジョブ ID が返り、異なる入力では競合になります。最大 8 件の未完了ジョブと計 128 件のジョブを保持します。完了・取消・失敗から 24 時間は結果を再取得できます。カーソルは対象ジョブに結び付きます。再起動時は確定済みの結果を保持し、同一のエンジンとモデルで次の未完了項目から進めます。ジョブの途中でエンジンのバイナリや重みの固定情報が変わった場合は、そのジョブを `identity_changed` として終了します。

### 再検証

`python3 experiments/fetch.py runtime` で比較用の上流実行ファイルを用意します。`service/.venv/bin/python service/verify.py all --out runs/mcp-verification.json` で、固定した 4 モデルの出力比較、マスクの適用、1,000 件の再開、保持期限、アイドルとメモリ閾値を検証します。`native` と `service` で範囲を分けられます。

[初回の検証結果](../results/2026-11-09-embedding-mcp.json) に、配備版の比較と再起動試験、および先行ビルドでの 1,000 件の復旧を記録しています。

### 外部の接続経路

- **Caddy** — このホストでは `embedding-mcp.moukaeritai.work` から `oci.moukaeritai.work` への CNAME を用意します。[Caddy のサイト設定](deploy/embedding-mcp.caddy) を import するときに、既存 MCP の `@trusted_mcp` にある許可リストを引数として渡します。ホストのインポート行は `~/.config/embedding-mcp/caddy-import.caddy` に置けます。DNS が反映されたら、そのファイルを稼働中の Caddyfile に import し、検証して reload します。送信元の判定は Caddy の `client_ip` に任せます。
- **ChatGPT Secure MCP Tunnel** — 専用トンネルの ID を用意します。このホストの既存プロファイルが共有するランタイムキー参照は `file:$HOME/.config/tunnel-client/secrets/runtime-api-key` です。既存の `tunnel-client@.service` を使います。トンネルの転送先は `http://127.0.0.1:18765/mcp` です。インストール済みの CLI では次の形で専用プロファイルを作成できます。

```bash
tunnel-client init --profile embedding-mcp --tunnel-id "$EMBED_MCP_TUNNEL_ID" \
  --mcp-server-url http://127.0.0.1:18765/mcp \
  --control-plane-api-key-ref "$EMBED_MCP_RUNTIME_KEY_REF" \
  --health-listen-addr 127.0.0.1:0
systemctl --user enable --now tunnel-client@embedding-mcp.service
```

トンネル ID の作成に使う管理鍵はランタイムキーと別です。作成後は、ChatGPT のアプリ設定で同じ専用トンネルを選びます。

任意の Bearer 認証を有効にするには、サービスユーザーだけが読めるファイルにトークンを置き、`~/.config/embedding-mcp/service.env` の `EMBED_MCP_BEARER_FILE` に絶対パスを指定します。Caddy 経由のクライアントは `Authorization: Bearer …` を送り、トンネルはランタイムと発見用の両方に同じ Authorization ヘッダーの秘密参照を設定します。トンネル用参照ファイルには `Bearer ` を含む値を入れます。

Caddy の `/mcp` には、Gemini の POST を観測した `66.249.82.128/27` と、HEAD・GET を観測した `108.177.69.32/29` を追加しています。後者は Google が案内する `108.177.64.0/20` の一部です。[Google のフェッチャー IP 範囲](https://developers.google.com/static/crawling/ipranges/user-triggered-fetchers-google.json) · [Connected Apps の接続要件](https://developers.google.com/connected-apps/gpc/prerequisites)

Caddy は公開ルート `/` で [ランディングページ](web/index.html) を配信します。ページとアセットは `service/web/` にあり、配信前のビルドは不要です。

Caddy とトンネルの利用者は同じジョブとメディアの保管領域を共有します。
