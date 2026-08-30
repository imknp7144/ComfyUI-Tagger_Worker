# ComfyUI Tagger Worker

複数のタギングモデル・セグメンテーションモデル・NSFWモザイクモデルを統合し、ComfyUI本体とは独立したvenv内の常駐プロセスとして推論を行うworkerノード群です。

- **標準タガー(NPU優先)**: camie / anima(pixai) / WD14 / JoyTag
- **Ensembleタガー(iGPU優先)**: cl_v1 / cl_v2 / dtq_l16 / dtq_b16 / oppai_v11 / wd_eva02_l
- **セグメンテーション**: isnet-anime(前景/背景マスク)
- **NSFWモザイク**: censor_detect

いずれも`onnxruntime-openvino` EP経由で推論し、ComfyUI側のライブラリ競合(NPU/CUDAの共存不可等)を避けつつ、RTXのVRAMを消費しない実行経路を提供します。

## 対応モデル一覧

| モデル | ノード | 推論デバイス | 配布元 | 自動DL |
|---|---|---|---|---|
| camie | Camie Tagger (Worker) | NPU→iGPU→CPU | Camais03/camie-tagger-v2 | ✅ |
| anima(pixai) | Anima Tagger (Worker) | NPU→iGPU→CPU | deepghs/pixai-tagger-v0.9-onnx | ✅ |
| WD14 | WD14 Tagger (Worker) | NPU→iGPU→CPU | SmilingWolf/wd-vit-tagger-v3 | ✅ |
| JoyTag | JoyTag Tagger (Worker) | NPU→iGPU→CPU | fancyfeast/joytag | ✅ |
| isnet-anime | isnet-anime Segmentation (Worker) | NPU→iGPU→CPU | tomjackson2023/rembg (isnet-anime.onnx) | ✅ |
| censor_detect | NSFW Mosaic (Worker) | NPU→iGPU→CPU | deepghs/anime_censor_detection | ✅ |
| cl_v1 | CL Tagger v1 (Ensemble/iGPU) | GPU(iGPU)→CPU | cella110n/cl_tagger (cl_tagger_1_02) | ✅ |
| oppai_v11 | OppaiOracle v1.1 (Ensemble/iGPU) | GPU(iGPU)→CPU | Grio43/OppaiOracle (V1.1_onnx) | ✅ |
| wd_eva02_l | WD EVA02-Large v3 (Ensemble/iGPU) | GPU(iGPU)→CPU | SmilingWolf/wd-eva02-large-tagger-v3 | ✅ |
| cl_v2 | CL Tagger v2 (Ensemble/iGPU) | GPU(iGPU)→CPU | cella110n/cl_tagger_v2 | ⚠️手動(要利用規約同意) |
| dtq_l16 | DanbooruTagQuery L16 (Ensemble/iGPU) | GPU(iGPU)→CPU | realphongha/danbooru-tag-query | ⚠️手動(要利用規約同意) |
| dtq_b16 | DanbooruTagQuery B16 (Ensemble/iGPU) | GPU(iGPU)→CPU | realphongha/danbooru-tag-query | ⚠️手動(要利用規約同意) |

Ensembleタガーは、姉妹プロジェクト `ComfyUI_Tagger_Ensemble_Worker`(CUDA/timm実行、ComfyUI本体プロセス内で動作)で採用されていたモデル群のうち、ONNXバックエンドの6モデルのみを本Workerに移植したものです。これら6モデルは可変shape・padding_mask付き入力を持つため実機でNPU動作に対応できないことが確認されており、`device`選択肢は`GPU`(Intel iGPU専用。OpenVINOのGPUプラグインはNVIDIA GPUを認識しないためRTXのVRAMは消費しません)/`CPU`の2択に固定しています。

`at_eva02`・`at_convnext_huge`の2モデルはtorch/timmバックエンドでONNX化されておらず(GPL-3.0)、ONNX Runtime前提の本Workerには原理的に統合できないため対象外です。タグ・アップサンプラー系(TagForge等)も役割が異なるため対象外とし、別プロジェクトとして切り出す方針です。

## ⚠️ メモリ使用量について(必読)

本ツールは **ComfyUI本体とは別プロセス** としてバックグラウンドで動き続けます。そのため、ここで消費されるメモリは **ComfyUI自体のメモリ使用量に加算される** 点に注意してください。

実測値の一例(NPU推論、`ov_cache`未生成の初回ロード時):

| 状態 | プロセスメモリ |
|---|---|
| worker起動直後(モデル未ロード) | 約90MB |
| camieモデル1つロード後 | 約860MB |
| camie + JoyTag 2モデルロード後 | 約2.0GB |

上記は標準タガー2つの時点での数字です。デフォルト設定 (`MAX_LOADED_MODELS = 3`) では、タガー系モデル(標準タガー・Ensembleタガーとも同じLRUプールを共有)が最大3つまで同時に常駐します。さらに`censor_detect`と`isnet_anime`の2モデルはこの上限とは別枠でカウントされるため、タガー3種 + censor_detect + isnet_anime が全て常駐すると、それに応じてメモリ消費もさらに積み上がる点を想定しておいてください。Ensembleタガー(特にcl_v2の巨大分類ヘッド)はモデルによってメモリフットプリントが大きいため、複数のEnsembleタガーを併用する場合は特に注意してください。

また、NPU/iGPU/OpenVINO EPは初回推論時に `models/<model_id>/ov_cache/` 以下にコンパイル済みキャッシュをディスクに生成します。これはメモリではなくディスク容量ですが、モデルごとに数百MB単位で増える場合があるため、ストレージ容量にも余裕を持たせてください。

### メモリ使用量を抑えたい場合

`worker.py` 冒頭の定数を編集してください:

```python
IDLE_TIMEOUT      = 7200  # 未使用モデルを自動解放するまでの秒数(デフォルト2時間)
MAX_LOADED_MODELS = 3     # タガーモデルの同時ロード上限(censor_detect / isnet_anime除く)
CLEANUP_INTERVAL  = 600   # 自動解放チェックの間隔(秒)
```

省メモリ環境では `MAX_LOADED_MODELS = 1〜2`、`IDLE_TIMEOUT` を短め(例: 600〜1800秒)に設定することを推奨します。モデルは使用時に自動でロードされる(Lazy Load)ため、上限を下げても機能自体は動作しますが、複数モデルを切り替えて使う場合はロード時間が増える点はトレードオフになります。

## ディレクトリ構成

```
ComfyUI/custom_nodes/ComfyUI_Tagger_Worker/
├── __init__.py
├── node_camie.py           # camie用ノード
├── node_anima.py           # pixai用ノード
├── node_wd14.py            # WD14用ノード
├── node_joytag.py          # JoyTag用ノード
├── node_isnet.py           # isnet-anime(前景/背景セグメンテーション)用ノード
├── node_mosaic.py          # NSFWモザイク用ノード
├── node_cl_v1.py           # Ensemble: CL Tagger v1用ノード
├── node_cl_v2.py           # Ensemble: CL Tagger v2用ノード
├── node_dtq_l16.py         # Ensemble: DanbooruTagQuery L16用ノード
├── node_dtq_b16.py         # Ensemble: DanbooruTagQuery B16用ノード
├── node_oppai_v11.py       # Ensemble: OppaiOracle v1.1用ノード
├── node_wd_eva02_l.py      # Ensemble: WD EVA02-Large v3用ノード
├── node_setup.py           # 常駐worker管理 + Setup Environmentノード
├── worker.py               # 常駐workerスクリプト本体
├── ensemble_catalog.py     # Ensembleタガー6モデルのメタデータ・ライセンス・配布元カタログ
├── ensemble_preprocess.py  # Ensembleタガー6モデルの前処理仕様レジストリ
├── setup_venv.py           # 独立venv構築・モデルダウンロードスクリプト
├── requirements_worker.txt
└── README.md
```

`venv/`, `models/`, `worker.log`, `worker.pid.json`, `setup_complete.json` は初回セットアップ・実行時に自動生成されるため、リポジトリには含まれません。

## 動作要件

- Python 3.12で動作確認済みです。`onnxruntime-openvino==1.24.1`のwheelが提供されている他のバージョン(3.10〜3.11等)でも動作する可能性がありますが未検証です。ComfyUIを起動しているPython(venv構築時に`sys.executable`として使われます)が対応バージョンであることを確認してください。
- Windows環境を主な動作確認環境としています(`add_openvino_libs_to_path()`はWindows専用の処理です)。Linux/macOSでの動作は未検証です。
- Ensembleタガー(GPU=iGPU実行)を利用する場合、Intel iGPU向けOpenVINO GPUプラグインが有効な環境が必要です。NVIDIA GPU(RTX等)はOpenVINOのGPUプラグイン経由では認識されません。

## インストール・セットアップ

1. 本リポジトリを `ComfyUI/custom_nodes/ComfyUI_Tagger_Worker` として配置します。
2. ComfyUIを起動し、**`Setup Tagger Worker Environment`** ノードを追加して `Queue Prompt` で実行します。
3. 自動的に独立したvenvが構築され、依存ライブラリ(`onnxruntime-openvino`, `openvino`, `dghs-imgutils`等)と以下のモデルがHuggingFaceからダウンロードされます。

   **標準タガー・セグメンテーション・モザイク(全自動)**
   - **camie**: `Camais03/camie-tagger-v2`
   - **anima**: `deepghs/pixai-tagger-v0.9-onnx`
   - **WD14**: `SmilingWolf/wd-vit-tagger-v3`
   - **JoyTag**: `fancyfeast/joytag`
   - **isnet-anime**: `tomjackson2023/rembg`(`isnet-anime.onnx`)
   - **censor_detect**: `deepghs/anime_censor_detection`(censor_detect_v0.10_s)

   **Ensembleタガー・非gated(自動)**
   - **cl_v1**: `cella110n/cl_tagger`(`cl_tagger_1_02/`フォルダ、Apache-2.0)
   - **oppai_v11**: `Grio43/OppaiOracle`(`V1.1_onnx/`フォルダ、Apache-2.0)
   - **wd_eva02_l**: `SmilingWolf/wd-eva02-large-tagger-v3`(Apache-2.0)

   **Ensembleタガー・gated(手動配置が必要)**
   - **cl_v2**: `cella110n/cl_tagger_v2`(独自ライセンス、利用規約への同意が必要)。`model.onnx`と同じフォルダに`model.onnx.data`(ONNX外部データ)も配置してください。
   - **dtq_l16 / dtq_b16**: `realphongha/danbooru-tag-query`(DINOv3 Licenseのため配布元HFで利用規約への同意が必要)

   gatedモデルは自動ダウンロードされません。セットアップ実行時、`setup_venv.py`が未配置を検知すると配置先パスとダウンロード元をコンソールに案内するので、手動でダウンロード・配置してから再度セットアップノードを実行してください。

4. セットアップ完了後 `setup_complete.json` が生成され、以降の起動時はスキップされます(モデルハッシュ検証つき)。

## タガーノード(標準: camie / anima / WD14 / JoyTag)

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

| カテゴリ | camie | anima | WD14 | JoyTag |
|---|---|---|---|---|
| general | ✅ | ✅ | ✅ | ✅(全部) |
| character | ✅ | ✅ | ✅ | ❌ |
| copyright | ✅ | ✅ | ✅ | ❌ |
| artist | ✅ | ❌ | ✅ | ❌ |
| rating | ✅ | ❌ | ✅ | ❌ |
| meta | ✅ | ❌ | ✅ | ❌ |
| year | ✅ | ❌ | ❌ | ❌ |

## Ensembleタガーノード(cl_v1 / cl_v2 / dtq_l16 / dtq_b16 / oppai_v11 / wd_eva02_l)

姉妹プロジェクト`ComfyUI_Tagger_Ensemble_Worker`から移植した6モデルです。入出力パラメータの形は標準タガーノードと共通ですが、`device`選択肢が`GPU`(iGPU)/`CPU`の2択のみになっている点が異なります(NPU非対応が実機確認済みのため選択肢に含めていません)。

### 入力・出力

標準タガーノードと同一の構成です(`threshold` / `threshold_character` / `threshold_copyright` / `exclude_tags` / `replace_underscores` に加え、`device`が`GPU`/`CPU`)。出力も`all_tags` / `general_tags` / `character_tags` / `copyright_tags` / `artist_tags` / `rating_tags` / `actual_backend`の7種です。

### 既知の注意点

- **oppai_v11**: 配布元の`selected_tags.csv`がカテゴリ情報を持たない(全タグがcategory=0)ため、実質的に全タグがgeneral扱いとなり、`character_tags`/`copyright_tags`は常に空になります。これはモデル配布物側の制約でありコード側の不具合ではありません。
- **cl_v2**: `model.onnx`と`model.onnx.data`(ONNX外部データ)をセットで配置する必要があります。片方だけではロードに失敗します。
- 各モデルの前処理仕様(入力解像度・正規化・パディング色・BGR順序等)は`ensemble_preprocess.py`のレジストリで一元管理しています。モデルごとに前処理が大きく異なる(例: wd_eva02_lはBGR・0-255レンジのまま、dtq系はDINOv3独自の正規化定数)ため、新規モデルを追加する場合は個別のPreprocessSpecを追加してください。

## isnet-anime Segmentationノード

アニメ絵のキャラクター前景/背景セグメンテーションを行います。既存の`RembgByBiRefNet`等の差し替えとして、同じ配線パターンで利用できます。

### 入力

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `image` | 入力画像 | - |
| `device` | 推論デバイス(`NPU` / `CPU`) | NPU |

### 出力

| 出力 | 説明 |
|---|---|
| `mask` | 前景(キャラクター)側が1、背景側が0のマスク |
| `actual_backend` | 実際に使用された推論バックエンド |

背景だけを残したい場合は`InvertMask` → `MaskImage`等と組み合わせてください。

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

`censor_detect`・`isnet_anime`の2モデルはタガー系の同時ロード上限(`MAX_LOADED_MODELS`)とは別枠で管理されますが、`IDLE_TIMEOUT`による自動解放の対象にはなります。

## 動作の仕組み(実装メモ)

- **Lazy Load**: 起動時にモデルはロードされません。各ノードの初回呼び出し時に必要なモデルのみロードされます。
- **デバイスフォールバックチェーン**: `device`に`NPU`を指定した場合、内部では`NPU → iGPU(OpenVINO "GPU") → 素のCPUExecutionProvider`の順で試行し、利用できないデバイスは自動的にスキップされます。`GPU`(Ensembleタガー等)は`iGPU → CPU`、`CPU`を明示した場合はOpenVINO EPを介さず直接CPUで実行します。各段階の失敗は`worker.log`にWARNINGとして記録されます。
- **LRU上限制御**: タガー系モデル(標準・Ensemble問わず同一プール)は`MAX_LOADED_MODELS`を超えると、最も使われていないものから自動解放されます。`censor_detect`・`isnet_anime`はこのLRUプールとは別枠でカウントされます。
- **アイドル自動解放**: `IDLE_TIMEOUT`秒間使用されなかったモデル(censor_detect/isnet_anime含む)は`CLEANUP_INTERVAL`間隔のチェックで自動アンロードされます。
- **直列処理**: workerはリクエストを1件ずつ直列に処理します(`ThreadPoolExecutor(max_workers=1)`)。これはNPU/iGPUへの同時アクセスによる不安定化を避けるための意図的な設計です。複数ノードを並列実行しても、内部的には順番に処理されるため、まとめて大量の画像を処理する場合は時間がかかる点をご了承ください。
- **localhost限定**: worker-node間通信は`127.0.0.1`のTCPソケット + トークン認証のみで、外部ネットワークへの公開を想定していません。同一マシン上の信頼された用途に限定してご利用ください。
- **raw_scoresモード**: `tag`アクションには、閾値フィルタリング前の生スコアを辞書形式で返す`raw_scores`モードが実装されています。現行のノード群からは呼び出されませんが、将来的な別クライアント(D-linerのimage_sorter等)との連携を見据えたプロトコルです。`register`/`unregister`アクションによるクライアント参照カウント管理(全クライアントが切断すると自動シャットダウン)も同様の連携を見据えて実装済みです。

## トラブルシューティング

- **Workerが起動しない**: `Setup Tagger Worker Environment`ノードを再実行してください。改善しない場合は`venv/`と`setup_complete.json`を削除して再セットアップしてください。
- **NPUが使われずiGPU/CPUにフォールバックする**: `worker.log`に`[Worker WARNING] ... falling back to ...`が出ていないか確認してください。ドライバやOpenVINO Runtimeのバージョン不整合が主な原因です。デバイスフォールバックチェーンにより自動的に次点のデバイスへ切り替わるため、処理自体は継続されます。
- **Ensembleタガーでgatedモデルのロードに失敗する**: `cl_v2` / `dtq_l16` / `dtq_b16`は自動ダウンロードされません。配布元(HuggingFace)で利用規約に同意した上で、`ensemble_catalog.py`に記載のファイル名で`models/<model_id>/`配下に手動配置してください。`cl_v2`は`model.onnx.data`の配置漏れに注意してください。
- **oppai_v11でキャラクター/著作権タグが出ない**: モデル配布物のCSVにカテゴリ情報が無いための既知の制約です(上記「Ensembleタガーノード」の既知の注意点を参照)。
- **プロセスが残留する**: 異常終了時、`worker.pid.json`が残ったままになることがあります。次回起動時に生存確認(ping)を行い、応答がなければ自動的に終了・削除されます。

## ライセンス

本プロジェクトはMITライセンスの下で提供されています。
各モデルのライセンス、およびONNX Runtime OpenVINO EPのライセンスについては、それぞれの開発元のライセンス条項に従ってください。特に以下のモデルは独自ライセンス・利用規約への同意が必要です。

- **cl_v2**: CL Tagger v2 Model License v1.0(独自ライセンス。再配布禁止・自己使用/条件付きサーブのみ許諾)
- **dtq_l16 / dtq_b16**: DINOv3 License(Meta社の独自ライセンス。バックボーンがDINOv3のため配布元HFで利用規約への同意が必要)