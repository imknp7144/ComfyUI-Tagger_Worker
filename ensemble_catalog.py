"""
ensemble_catalog.py

_MyEXT_tagger_ensemble_worker (Heavyタガー、ComfyUI本体プロセス内でCUDA/timm実行) 由来の
6モデルを、このworker.py(専用venv、onnxruntime-openvino)経由でiGPU実行できるように
するためのカタログ定義。

【背景】これら6モデルは実機で「可変shape・padding_mask付き入力を持つため NPUでの動作に
対応できない」ことが確認されている(tagger_ensemble_worker側の引き継ぎ資料を参照)。
一方でCUDA(RTXのVRAM)を使わせたくないというユーザー要望があるため、OpenVINOのGPUプラグイン
(Intel iGPU専用、NVIDIA GPUは認識しない)経由での実行先として、この _MyEXT_ComfyUI_Tagger_Worker
に統合する。そのため、これらのモデルの device 選択肢は "NPU" を含めず "GPU"(iGPU) / "CPU" の
2択のみとする(node_*.py 側で固定)。

除外したモデル: at_eva02 / at_convnext_huge (GPL-3.0、torch/timmバックエンドでONNX化されて
いないため、ONNX Runtime前提のこのworkerには原理的に統合できない。将来ONNX変換すれば対象に
できるが、本カタログの対象外とする)。

gatedモデル(cl_v2, dtq_l16, dtq_b16)は、tagger_ensemble_worker側と同じ方針を踏襲し、
利用規約への同意が必要な配布元のため自動ダウンロードは行わない。setup_venv.py は
「配置されているか確認し、無ければ配置先とダウンロード元を案内する」だけに留める。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnsembleModelEntry:
    model_id: str
    gated: bool
    license: str
    source_url: str
    tags_filename: str                       # models/<model_id>/ からの相対ファイル名
    category_filename: Optional[str] = None  # タグ名とカテゴリが別ファイルのモデル用(dtq系)
    apply_sigmoid: bool = True
    notes: Optional[str] = None              # 配置時の追加注意事項(外部データファイル等)
    # 非gatedモデルの自動ダウンロード用(huggingface_hub repo)。gatedモデルはNoneのまま。
    repo_id: Optional[str] = None
    repo_model_filename: Optional[str] = None
    repo_tags_filename: Optional[str] = None


# --------------------------------------------------------------------------
# モデルカタログ (2026-08時点、tagger_ensemble_worker側の実機検証結果に基づく)
# --------------------------------------------------------------------------
ENSEMBLE_MODEL_CATALOG = [
    EnsembleModelEntry(
        model_id="cl_v2", gated=True,
        license="CL Tagger v2 Model License v1.0(独自ライセンス。再配布禁止・自己使用/条件付きサーブのみ許諾)",
        source_url="https://huggingface.co/cella110n/cl_tagger_v2",
        tags_filename="model_vocabulary.json",
        notes="model.onnx と同じフォルダに model.onnx.data (ONNX外部データ) も配置すること。",
    ),
    EnsembleModelEntry(
        model_id="cl_v1", gated=False, license="Apache-2.0",
        source_url="https://huggingface.co/cella110n/cl_tagger (cl_tagger_1_02/ フォルダ、最新版)",
        tags_filename="tag_mapping.json",
        repo_id="cella110n/cl_tagger",
        repo_model_filename="cl_tagger_1_02/model.onnx",
        repo_tags_filename="cl_tagger_1_02/tag_mapping.json",
    ),
    EnsembleModelEntry(
        model_id="dtq_l16", gated=True,
        license="DINOv3 License(Meta社の独自ライセンス。バックボーンがDINOv3のため配布元HFで利用規約への同意が必要)",
        source_url="https://huggingface.co/realphongha/danbooru-tag-query"
                    " (models/DanbooruTagQuery_l16_448x448/ フォルダ)",
        tags_filename="tag_to_id.json",
        category_filename="tag_category.json",
    ),
    EnsembleModelEntry(
        model_id="dtq_b16", gated=True,
        license="DINOv3 License(Meta社の独自ライセンス。バックボーンがDINOv3のため配布元HFで利用規約への同意が必要)",
        source_url="https://huggingface.co/realphongha/danbooru-tag-query"
                    " (models/DanbooruTagQuery_b16_448x448/ フォルダ)",
        tags_filename="tag_to_id.json",
        category_filename="tag_category.json",
    ),
    EnsembleModelEntry(
        model_id="oppai_v11", gated=False, license="Apache-2.0",
        source_url="https://huggingface.co/Grio43/OppaiOracle (V1.1_onnx/ フォルダ)",
        tags_filename="selected_tags.csv",
        # 【実機確認済み】モデル自身が既にsigmoid適用済みの確率を出力している。
        # apply_sigmoid=True のままだと二重適用となり出力レンジが圧縮され、
        # どのタグも閾値を超えられずtag出力が常に空になる不具合がtagger_ensemble_worker側で
        # 実機確認されている。
        apply_sigmoid=False,
        repo_id="Grio43/OppaiOracle",
        repo_model_filename="V1.1_onnx/model.onnx",
        repo_tags_filename="V1.1_onnx/selected_tags.csv",
    ),
    EnsembleModelEntry(
        model_id="wd_eva02_l", gated=False, license="Apache-2.0",
        source_url="https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3",
        tags_filename="selected_tags.csv",
        repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
        repo_model_filename="model.onnx",
        repo_tags_filename="selected_tags.csv",
    ),
]

ENSEMBLE_MODEL_IDS = {e.model_id for e in ENSEMBLE_MODEL_CATALOG}
_ENSEMBLE_BY_ID = {e.model_id: e for e in ENSEMBLE_MODEL_CATALOG}


def get_ensemble_entry(model_id: str) -> EnsembleModelEntry:
    if model_id not in _ENSEMBLE_BY_ID:
        raise KeyError(f"Unknown ensemble model_id: {model_id}")
    return _ENSEMBLE_BY_ID[model_id]
