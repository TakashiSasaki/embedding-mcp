# Embedding MCP の作業引き継ぎ

## 再開時に確認すること

1. `git status --short --branch`、`git log -5 --oneline`、`git remote -v` で作業ツリーと変更履歴を確認します。既存ホストの作業ツリーは `/data/cpu-embedding-experiments/main` にあり、`main` は GitHub リモートの `github/main` を追跡します。このホストで配置やブランチを変える場合は親ディレクトリと `/data/AGENTS.md` の方針を確認してください。
2. [ルート README](README.md) で実験とサービスを区別します。[設計文書](service/DESIGN.md) から変更箇所を探し、[サービス README](service/README.md) で起動・接続・ツールの契約を確認します。[初回の検証記録](results/2026-11-09-embedding-mcp.json) は当時の結果です。現在の稼働状況はサービスを照会して確認します。
3. CPU の実測には固定したアセットを使います。`assets.lock.json` とワーカーのマニフェストを基準にし、重み、生成したランタイム、実行ログ、稼働中の SQLite、認証情報はローカルに保管します。共有する再現条件と結果の要約は `results/` にコミットします。

## 変更の確認

- MCP のフロント、スケジューラー、保存処理を変更したら `service/.venv/bin/python service/verify.py service --items 32` を使います。サービスが起動済みでも隔離されたポートと状態を使用します。ビルドやアセットがなければ先に [準備手順](service/README.md#準備) に従います。
- llama.cpp のパッチやモデルの条件を変更したら、`python3 experiments/fetch.py runtime` で比較用の上流ランタイムを用意し、`service/.venv/bin/python service/verify.py native` を使います。両方を検証する場合は `all` を指定します。
- ドキュメントを更新したら、相対リンクとコマンドが現行のファイルや CLI に対応することを確認します。

## 稼働中のサービスを扱う場合

- まず `systemctl --user status embedding-mcp.service` と `curl --fail http://127.0.0.1:18765/health` でフロントを確認します。MCP の接続テストは [SDK の例](service/README.md#準備) を使い、ロードやジョブの状況は `runtime_status` で確認します。ログは `journalctl --user -u embedding-mcp.service` と、外部経路の場合は `~/caddy/logs/embedding-mcp-access.log` を使います。
- 変更の反映に再起動が必要なら、先に `runtime_status` で実行中の処理とキューを確認します。反映後はヘルス、`tools/list`、対象ツールの応答を確認し、利用者が通る経路でも検証します。Caddy を変更する場合は設定を検証してから reload します。
- 2026年11月9日時点で Caddy 経由の Gemini 接続と MiniLM バッチが確認済みです。SSE の修正後はホスト上の SDK で公開 HTTPS 経路を再検証しました。同日の確認では専用 ChatGPT トンネルのユーザーユニットは inactive でした。外部クライアントからの検証とホスト上の疎通結果を分けて報告します。
