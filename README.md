# ComfyUI Tagger Worker

5つのモデル(camie / pixai / WD14 / JoyTag / censor_detect)を統合し、`onnxruntime-openvino` EP経由でNPU/CPU推論を行う常駐型workerノード群です。

タギング(camie/pixai/WD14/JoyTag)とNSFWモザイク処理(censor_detect)を、ComfyUI本体とは独立したvenv内の常駐プロセスとして実行することで、ComfyUI側のライブラリ競合を避けつつ動作します。

## ⚠️ メモリ使用量について(必読)

本ツールは **ComfyUI本体とは別プロセス** としてバックグラウンドで動き続けます。そのため、ここで消費されるメモリは **ComfyUI自体のメモリ使用量に加算される** 点に注意してください。

実測値の一例(NPU推論、`ov_cache`未生成の初回ロード時):

| 状態 | プロセスメモリ |
|---|---|
| worker起動直後(モデル未ロード) | 約90MB |
| camieモデル1つロード後 | 約860MB |
| camie + JoyTag 2モデルロード後 | 約2.0GB |

上記はモデル2つの時点での数字です。デフォルト設定 (`MAX_LOADED_MODELS = 3`) では、タガー系モデルが最大3つまで同時に常駐します。さらにNSFWモザイク用の `censor_detect` モデルはこの上限とは別枠でカウントされるため、タガー3種 + censor_detectが全て常駐すると **合計で3〜4GB程度のメモリを消費する状態が定常化しうる** ことを想定しておいてください。

また、NPU/OpenVINO EPは初回推論時に `models/<model_id>/ov_cache/` 以下にコンパイル済みキャッシュをディスクに生成します。これはメモリではなくディスク容量ですが、モデルごとに数百MB単位で増える場合があるため、ストレージ容量にも余裕を持たせてください。

### メモリ使用量を抑えたい場合

`worker.py` 冒頭の定数を編集してください:

```python
IDLE_TIMEOUT      = 7200  # 未使用モデルを自動解放するまでの秒数(デフォルト2時間)
MAX_LOADED_MODELS = 3     # タガーモデルの同時ロード上限(censor_detect除く)
CLEANUP_INTERVAL  = 600   # 自動解放チェックの間隔(秒)
```

省メモリ環境では `MAX_LOADED_MODELS = 1〜2`、`IDLE_TIMEOUT` を短め(例: 600〜1800秒)に設定することを推奨します。モデルは使用時に自動でロードされる(Lazy Load)ため、上限を下げても機能自体は動作しますが、複数モデルを切り替えて使う場合はロード時間が増える点はトレードオフになります。

## ディレクトリ構成

```
ComfyUI/custom_nodes/ComfyUI_Tagger_Worker/
├── __init__.py
├── node_camie.py          # camie用ノード
├── node_anima.py          # pixai用ノード
├── node_wd14.py           # WD14用ノード
├── node_joytag.py         # JoyTag用ノード
├── node_mosaic.py         # NSFWモザイク用ノード
├── node_setup.py          # 常駐worker管理 + Setup Environmentノード
├── worker.py              # 常駐workerスクリプト本体
├── setup_venv.py          # 独立venv構築・モデルダウンロードスクリプト
├── requirements_worker.txt
└── README.md
```

`venv/`, `models/`, `worker.log`, `worker.pid.json`, `setup_complete.json` は初回セットアップ・実行時に自動生成されるため、リポジトリには含まれません。

## 動作要件

- Python 3.12で動作確認済みです。`onnxruntime-openvino==1.24.1`のwheelが提供されている他のバージョン(3.10〜3.11等)でも動作する可能性がありますが未検証です。ComfyUIを起動しているPython(venv構築時に`sys.executable`として使われます)が対応バージョンであることを確認してください。
- Windows環境を主な動作確認環境としています(`add_openvino_libs_to_path()`はWindows専用の処理です)。Linux/macOSでの動作は未検証です。

## インストール・セットアップ

1. 本リポジトリを `ComfyUI/custom_nodes/ComfyUI_Tagger_Worker` として配置します。
2. ComfyUIを起動し、**`Setup Tagger Worker Environment`** ノードを追加して `Queue Prompt` で実行します。
3. 自動的に独立したvenvが構築され、依存ライブラリ(`onnxruntime-openvino`, `openvino`, `dghs-imgutils`等)と以下5モデルがHuggingFaceからダウンロードされます。
   - **camie**: `Camais03/camie-tagger-v2`
   - **pixai**: `deepghs/pixai-tagger-v0.9-onnx`
   - **WD14**: `SmilingWolf/wd-vit-tagger-v3`
   - **JoyTag**: `fancyfeast/joytag`
   - **censor_detect**: `deepghs/anime_censor_detection`(censor_detect_v0.10_s)
4. セットアップ完了後 `setup_complete.json` が生成され、以降の起動時はスキップされます(モデルハッシュ検証つき)。

## タガーノード(camie / anima / WD14 / JoyTag 共通)

### 入力

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `image` | 入力画像 | - |
| `threshold` | 一般タグの検出閾値 | 0.35 |
| `threshold_character` | キャラクタータグの検出閾値 | 0.75 |
| `threshold_copyright` | 著作権タグの検出閾値 | 0.50 |
| `exclude_tags` | 除外タグ(カンマ区切り、大小文字/アンダースコア表記揺れ対応) | "" |
| `replace_underscores` | アンダースコア→スペース変換 | True |
| `device` | 推論デバイス(`NPU` / `CPU`) | NPU |

### 出力

| 出力 | 説明 |
|---|---|
| `all_tags` | 検出タグ全件(カンマ区切り) |
| `general_tags` | 一般タグ |
| `character_tags` | キャラクタータグ(対応モデルのみ) |
| `copyright_tags` | 著作権タグ(対応モデルのみ) |
| `artist_tags` | アーティストタグ(対応モデルのみ) |
| `rating_tags` | レーティングタグ(対応モデルのみ) |
| `actual_backend` | 実際に使用された推論バックエンド(例: `OpenVINOExecutionProvider`) |

### モデル別カテゴリ対応表

| カテゴリ | camie | pixai | WD14 | JoyTag |
|---|---|---|---|---|
| general | ✅ | ✅ | ✅ | ✅(全部) |
| character | ✅ | ✅ | ✅ | ❌ |
| copyright | ✅ | ✅ | ✅ | ❌ |
| artist | ✅ | ❌ | ✅ | ❌ |
| rating | ✅ | ❌ | ✅ | ❌ |
| meta | ✅ | ❌ | ✅ | ❌ |
| year | ✅ | ❌ | ❌ | ❌ |

## NSFW Mosaicノード

`censor_detect_v0.10_s`(YOLOv8ベース検出モデル)で画像内の該当箇所を検出し、モザイク/ぼかし/黒塗り処理を行います。

### 入力

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `image` | 入力画像 | - |
| `detect_nipple_f` | 検出対象クラスの有効/無効 | True |
| `detect_penis` | 同上 | True |
| `detect_pussy` | 同上 | True |
| `method` | 処理方式(`モザイク` / `ぼかし` / `黒塗り`) | モザイク |
| `mosaic_strength` | モザイク強度 | 20 |
| `blur_radius` | ぼかし半径 | 10 |
| `confidence` | 検出信頼度の閾値 | 0.35 |
| `device` | 推論デバイス(`NPU` / `CPU`) | NPU |

### 出力

| 出力 | 説明 |
|---|---|
| `image` | 処理後の画像 |
| `actual_backend` | 実際に使用された推論バックエンド |

`censor_detect`モデルはタガー系の同時ロード上限(`MAX_LOADED_MODELS`)とは別枠で管理されますが、`IDLE_TIMEOUT`による自動解放の対象にはなります。

## 動作の仕組み(実装メモ)

- **Lazy Load**: 起動時にモデルはロードされません。各ノードの初回呼び出し時に必要なモデルのみロードされます。
- **LRU上限制御**: タガー系モデルは`MAX_LOADED_MODELS`を超えると、最も使われていないものから自動解放されます。
- **アイドル自動解放**: `IDLE_TIMEOUT`秒間使用されなかったモデル(censor_detect含む)は`CLEANUP_INTERVAL`間隔のチェックで自動アンロードされます。
- **直列処理**: workerはリクエストを1件ずつ直列に処理します(`ThreadPoolExecutor(max_workers=1)`)。これはNPU/GPUへの同時アクセスによる不安定化を避けるための意図的な設計です。複数ノードを並列実行しても、内部的には順番に処理されるため、まとめて大量の画像を処理する場合は時間がかかる点をご了承ください。
- **localhost限定**: worker-node間通信は`127.0.0.1`のTCPソケット + トークン認証のみで、外部ネットワークへの公開を想定していません。同一マシン上の信頼された用途に限定してご利用ください。
- **予約プロトコル**: `worker.py`には`register`/`unregister`/`raw_scores`といった、現在のノード群からは呼び出されないaction/モードが実装されています。これは将来的な別クライアント(D-linerなど)との連携を見据えて予約されているもので、現状のComfyUIノード単体では使用されません。

## トラブルシューティング

- **Workerが起動しない**: `Setup Tagger Worker Environment`ノードを再実行してください。改善しない場合は`venv/`と`setup_complete.json`を削除して再セットアップしてください。
- **NPUが使われずCPUにフォールバックする**: `worker.log`に`[Worker WARNING] ... falling back to CPU`が出ていないか確認してください。ドライバやOpenVINO Runtimeのバージョン不整合が主な原因です。
- **プロセスが残留する**: 異常終了時、`worker.pid.json`が残ったままになることがあります。次回起動時に生存確認(ping)を行い、応答がなければ自動的に終了・削除されます。

## ライセンス

本プロジェクトはMITライセンスの下で提供されています。
各モデルのライセンス、およびONNX Runtime OpenVINO EPのライセンスについては、それぞれの開発元のライセンス条項に従ってください。
