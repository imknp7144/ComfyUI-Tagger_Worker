"""
worker.py  –  ComfyUI Tagger + NSFW Mosaic 統合常駐推論サーバー

  action: "tag"    … camie / anima / wd14 / joytag タギング
  action: "mosaic" … censor_detect_v0.10_s による NSFW モザイク処理

推論バックエンド（両 action 共通）:
  onnxruntime-openvino EP  NPU → CPU フォールバック

セッション管理:
  - 起動時プリロードなし（Lazy Load）
  - SessionEntry による last_used 管理
  - IDLE_TIMEOUT 秒未使用セッションの自動解放
  - MAX_LOADED_MODELS による LRU 上限制御（censor_detect 除く）
  - タグ辞書も Lazy Load
"""

import gc
import os
import sys
import socket
import json
import secrets
import glob
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import re

# onnxruntime より先に OpenVINO Win ライブラリを PATH に追加
if sys.platform == "win32":
    try:
        import onnxruntime.tools.add_openvino_win_libs as _ov_libs
        _ov_libs.add_openvino_libs_to_path()
    except Exception as e:
        print(f"[Worker WARNING] Failed to add OpenVINO Win libs to path: {e}", flush=True)

import onnxruntime as ort

# stdout を UTF-8 に固定（Windows cp932 対策）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# パス定義
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(BASE_DIR, "worker.pid.json")

# 起動時に残留一時ファイルを削除
for _pat in ["comfyui_tagger_in_*.bmp", "comfyui_nsfw_in_*.bmp", "comfyui_nsfw_out_*.bmp"]:
    for _f in glob.glob(os.path.join(tempfile.gettempdir(), _pat)):
        try:
            os.unlink(_f)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# クライアント接続数管理
# ---------------------------------------------------------------------------
_client_count: int = 0
_client_count_lock = threading.Lock()

# ---------------------------------------------------------------------------
# セッション管理定数
# ---------------------------------------------------------------------------
IDLE_TIMEOUT      = 7200  # 秒: 未使用セッションの自動解放まで（2時間）
MAX_LOADED_MODELS = 3      # タガーモデル同時ロード上限（censor_detect 除く）
CLEANUP_INTERVAL  = 600    # 秒: アイドルチェック間隔（10分）

# ---------------------------------------------------------------------------
# SessionEntry
# ---------------------------------------------------------------------------
@dataclass
class SessionEntry:
    session:        object   # ort.InferenceSession
    device:         str
    actual_backend: str
    loaded_at:      float = field(default_factory=time.time)
    last_used:      float = field(default_factory=time.time)

    def touch(self):
        self.last_used = time.time()

    def idle_seconds(self) -> float:
        return time.time() - self.last_used


# ---------------------------------------------------------------------------
# セッションキャッシュ（Lazy Load）
# key: (model_id, device) → SessionEntry
# ---------------------------------------------------------------------------
_session_cache: dict = {}
_session_lock  = threading.Lock()

# ---------------------------------------------------------------------------
# タグ辞書キャッシュ（Lazy Load）
# key: model_id → (tags_list, tag_to_category_dict)
# ---------------------------------------------------------------------------
_tag_cache: dict = {}
_tag_lock  = threading.Lock()

# ---------------------------------------------------------------------------
# メモリ使用量取得ユーティリティ
# ---------------------------------------------------------------------------
def _get_process_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# セッション生成（共通ユーティリティ）
# ---------------------------------------------------------------------------
def _create_ort_session(onnx_path: str, cache_dir: str, device: str) -> tuple:
    """OpenVINO EP → CPU の順でセッションを生成。(session, actual_backend) を返す。"""
    os.makedirs(cache_dir, exist_ok=True)
    providers = [
        ("OpenVINOExecutionProvider", {
            "device_type": device,
            "cache_dir":   cache_dir,
        }),
        "CPUExecutionProvider",
    ]
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        actual  = session.get_providers()[0]
        print(f"[Worker] Session created with provider: {actual}", flush=True)
        return session, actual
    except Exception as e:
        if device != "CPU":
            print(f"[Worker WARNING] {device} provider failed ({e}), falling back to CPU.", flush=True)
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            actual  = session.get_providers()[0]
            print(f"[Worker] Session created with fallback provider: {actual}", flush=True)
            return session, actual
        raise


def _model_onnx_path(model_id: str) -> str:
    if model_id == "censor_detect":
        return os.path.join(BASE_DIR, "models", "censor_detect_v0.10_s", "model.onnx")
    return os.path.join(BASE_DIR, "models", model_id, "model.onnx")


def _model_cache_dir(model_id: str) -> str:
    subdir = "censor_detect_v0.10_s" if model_id == "censor_detect" else model_id
    return os.path.join(BASE_DIR, "models", subdir, "ov_cache")

# ---------------------------------------------------------------------------
# LRU 解放: _session_lock 保持状態で呼ぶこと
# ---------------------------------------------------------------------------
def _evict_lru_if_needed():
    tagger_keys = [(mid, dev) for (mid, dev) in _session_cache if mid != "censor_detect"]
    if len(tagger_keys) <= MAX_LOADED_MODELS:
        return
    oldest_key = min(tagger_keys, key=lambda k: _session_cache[k].last_used)
    entry = _session_cache.pop(oldest_key)
    model_id, device = oldest_key
    idle = entry.idle_seconds()
    mem_before = _get_process_memory_mb()
    try:
        del entry.session
        entry.session = None
    except Exception:
        pass
    gc.collect()
    mem_after = _get_process_memory_mb()
    print(
        f"[Worker] Unload model (LRU evict):\n"
        f"  Model:        {model_id}\n"
        f"  Device:       {device}\n"
        f"  Idle:         {idle:.0f}s\n"
        f"  Memory Freed: {mem_before - mem_after:.1f} MB",
        flush=True,
    )

# ---------------------------------------------------------------------------
# get_session: Lazy Load + LRU + ログ
# ---------------------------------------------------------------------------
def get_session(model_id: str, device: str) -> tuple:
    key = (model_id, device)
    with _session_lock:
        if key in _session_cache:
            entry = _session_cache[key]
            entry.touch()
            return entry.session, entry.actual_backend

        # --- 新規ロード ---
        onnx_path = _model_onnx_path(model_id)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"Model ONNX not found: {onnx_path}\n"
                "Setup Worker Environment ノードを実行してください。"
            )
        cache_dir = _model_cache_dir(model_id)

        try:
            model_size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        except Exception:
            model_size_mb = 0.0

        mem_before = _get_process_memory_mb()
        t0 = time.time()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[Worker] Loading model:\n"
            f"  Model:  {model_id}\n"
            f"  Device: {device}\n"
            f"  Time:   {now_str}\n"
            f"  Memory: {mem_before:.2f} MB",
            flush=True,
        )

        session, actual = _create_ort_session(onnx_path, cache_dir, device)
        elapsed = time.time() - t0
        mem_after = _get_process_memory_mb()

        print(
            f"[Worker] Model loaded:\n"
            f"  Model:        {model_id}\n"
            f"  Device:       {device}\n"
            f"  Load Time:    {elapsed:.2f}s\n"
            f"  Memory After: {mem_after:.2f} MB (Diff: {mem_after - mem_before:.2f} MB)",
            flush=True,
        )

        entry = SessionEntry(session=session, device=device, actual_backend=actual)
        _session_cache[key] = entry

        if model_id != "censor_detect":
            _evict_lru_if_needed()

        return entry.session, entry.actual_backend

# ---------------------------------------------------------------------------
# Idle Session Unload: バックグラウンド定期実行
# ---------------------------------------------------------------------------
def _cleanup_sessions():
    with _session_lock:
        to_remove = []
        for key, entry in _session_cache.items():
            if entry.idle_seconds() > IDLE_TIMEOUT:
                to_remove.append(key)
        for key in to_remove:
            entry = _session_cache.pop(key)
            model_id, device = key
            idle = entry.idle_seconds()
            mem_before = _get_process_memory_mb()
            try:
                del entry.session
                entry.session = None
            except Exception:
                pass
            gc.collect()
            mem_after = _get_process_memory_mb()
            print(
                f"[Worker] Unload model (idle):\n"
                f"  Model:        {model_id}\n"
                f"  Device:       {device}\n"
                f"  Idle:         {idle:.0f}s\n"
                f"  Memory Freed: {mem_before - mem_after:.1f} MB",
                flush=True,
            )


def _cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL)
        try:
            _cleanup_sessions()
        except Exception as e:
            print(f"[Worker WARNING] cleanup_sessions error: {e}", flush=True)

# ---------------------------------------------------------------------------
# タグ辞書: Lazy Load
# ---------------------------------------------------------------------------
def _get_tags(model_id: str) -> tuple:
    """(tags_list, tag_to_category_dict) を返す。初回のみファイルを読み込む。"""
    with _tag_lock:
        if model_id in _tag_cache:
            return _tag_cache[model_id]
        result = _load_tags_from_file(model_id)
        _tag_cache[model_id] = result
        return result


def _load_tags_from_file(model_id: str) -> tuple:
    import csv

    if model_id == "camie":
        meta_path = os.path.join(BASE_DIR, "models", "camie", "camie-tagger-v2-metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} が見つかりません。")
        print(f"[Worker] Loading camie tags from {meta_path}...", flush=True)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tag_mapping = meta["dataset_info"]["tag_mapping"]
        if "idx_to_tag" in tag_mapping:
            idx_map = {int(k): v for k, v in tag_mapping["idx_to_tag"].items()}
            tags = [idx_map[i] for i in range(len(idx_map))]
        else:
            tags = [
                tag for tag, _ in sorted(tag_mapping["tag_to_idx"].items(), key=lambda x: x[1])
            ]
        tag_to_category = tag_mapping.get("tag_to_category", {})
        print(f"[Worker] Loaded {len(tags)} camie tags.", flush=True)
        return tags, tag_to_category

    elif model_id == "anima":
        csv_path = os.path.join(BASE_DIR, "models", "anima", "selected_tags.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} が見つかりません。")
        print(f"[Worker] Loading anima tags from {csv_path}...", flush=True)
        _CAT_MAP = {"0": "general", "4": "character", "3": "copyright", "1": "artist"}
        tag_names: list = []
        tag_categories: dict = {}
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return [], {}
            header_lower = [h.strip().lower() for h in header]
            name_col = header_lower.index("name")     if "name"     in header_lower else 2
            cat_col  = header_lower.index("category") if "category" in header_lower else 3
            idx_col  = header_lower.index("id")       if "id"       in header_lower else 0
            all_rows = list(reader)
            def _row_idx(r):
                try:
                    return int(r[idx_col])
                except (ValueError, IndexError):
                    return 999999
            for cells in sorted(all_rows, key=_row_idx):
                if len(cells) <= max(name_col, cat_col):
                    tag_names.append("")
                    continue
                name = cells[name_col].strip()
                cat  = _CAT_MAP.get(cells[cat_col].strip(), "general")
                tag_names.append(name)
                if name:
                    tag_categories[name] = cat
        print(f"[Worker] Loaded {len(tag_names)} anima tags.", flush=True)
        return tag_names, tag_categories

    elif model_id == "wd14":
        csv_path = os.path.join(BASE_DIR, "models", "wd14", "selected_tags.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} が見つかりません。")
        print(f"[Worker] Loading WD14 tags from {csv_path}...", flush=True)
        _CAT_MAP = {
            "0": "general", "1": "artist", "3": "copyright",
            "4": "character", "5": "meta", "9": "rating",
        }
        tag_names: list = []
        tag_categories: dict = {}
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    name = row[1].strip()
                    cat  = _CAT_MAP.get(row[2].strip(), "general")
                    tag_names.append(name)
                    tag_categories[name] = cat
        print(f"[Worker] Loaded {len(tag_names)} WD14 tags.", flush=True)
        return tag_names, tag_categories

    elif model_id == "joytag":
        txt_path = os.path.join(BASE_DIR, "models", "joytag", "top_tags.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"{txt_path} が見つかりません。")
        print(f"[Worker] Loading JoyTag tags from {txt_path}...", flush=True)
        with open(txt_path, "r", encoding="utf-8") as f:
            tags = [line.strip() for line in f if line.strip()]
        print(f"[Worker] Loaded {len(tags)} JoyTag tags.", flush=True)
        return tags, {}

    else:
        raise ValueError(f"Unknown model for tag loading: {model_id}")

# ---------------------------------------------------------------------------
# タガー: 前処理
# ---------------------------------------------------------------------------
_KEEP_UNDERSCORE_RE = re.compile(
    r"^(score_\d+|score_\d+_up|masterpiece|best_quality|"
    r"good_quality|normal_quality|low_quality|worst_quality)$"
)
_DANBOORU_TO_GELBOORU = {
    "hair_ornament":  "hair ornament",
    "hair_ribbon":    "hair ribbon",
    "hair_flower":    "hair flower",
    "holding_hands":  "holding hands",
    "sailor_collar":  "sailor collar",
    "school_uniform": "school uniform",
    "thigh_highs":    "thighhighs",
    "knee_highs":     "kneehighs",
    "ahoge":          "ahoge",
}

def normalize_anima_tag(tag: str) -> str:
    mapped = _DANBOORU_TO_GELBOORU.get(tag, tag)
    if _KEEP_UNDERSCORE_RE.match(mapped):
        return mapped
    return mapped.replace("_", " ")


def preprocess_camie(image_pil: Image.Image) -> np.ndarray:
    """512×512、ImageNet正規化、NCHW [1,3,512,512]"""
    image = image_pil.convert("RGB").resize((512, 512), Image.LANCZOS)
    arr   = np.array(image, dtype=np.float32) / 255.0
    mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr   = (arr - mean) / std
    arr   = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0)   # [1, 3, 512, 512]


def preprocess_anima(image_pil: Image.Image) -> np.ndarray:
    """448×448 letterbox（黒パディング）、ImageNet正規化、NCHW [1,3,448,448]"""
    image  = image_pil.convert("RGB")
    w, h   = image.size
    scale  = 448 / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    image  = image.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (448, 448), (0, 0, 0))
    canvas.paste(image, ((448 - nw) // 2, (448 - nh) // 2))
    arr    = np.array(canvas, dtype=np.float32) / 255.0
    mean   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std    = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr    = (arr - mean) / std
    arr    = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0)   # [1, 3, 448, 448]


def preprocess_wd14(image_pil: Image.Image) -> np.ndarray:
    """448×448、RGB→BGR、[0,255] float32、NHWC [1,448,448,3]"""
    image = image_pil.convert("RGB").resize((448, 448), Image.LANCZOS)
    arr   = np.array(image, dtype=np.float32)
    arr   = arr[:, :, ::-1]         # RGB → BGR
    return np.expand_dims(arr, 0)   # [1, 448, 448, 3]


def preprocess_joytag(image_pil: Image.Image) -> np.ndarray:
    """448×448、[0,1] float32、NCHW [1,3,448,448]"""
    image = image_pil.convert("RGB").resize((448, 448), Image.LANCZOS)
    arr   = np.array(image, dtype=np.float32) / 255.0
    arr   = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0)   # [1, 3, 448, 448]

# ---------------------------------------------------------------------------
# censor_detect: 前処理 / NMS / 後処理 / モザイク処理
# ---------------------------------------------------------------------------
CENSOR_CLASS_NAMES = {0: "nipple_f", 1: "penis", 2: "pussy"}


def preprocess_censor(image_pil: Image.Image):
    """YOLOv8標準前処理（左上寄せ右下パディング）。scale と元サイズを返す。"""
    orig_w, orig_h = image_pil.size
    scale          = min(640 / orig_w, 640 / orig_h)
    new_w, new_h   = int(orig_w * scale), int(orig_h * scale)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    resized = image_pil.resize((new_w, new_h), resample)
    padded  = Image.new("RGB", (640, 640), (114, 114, 114))
    padded.paste(resized, (0, 0))
    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)    # [1, 3, 640, 640]
    return arr, scale, orig_w, orig_h


def _iou(a, b) -> float:
    inter = (
        max(0, min(a[2], b[2]) - max(a[0], b[0])) *
        max(0, min(a[3], b[3]) - max(a[1], b[1]))
    )
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(detections: list, iou_threshold: float = 0.45) -> list:
    detections = sorted(detections, key=lambda x: x[5], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if _iou(best, d) < iou_threshold]
    return kept


def postprocess_censor(
    output: np.ndarray,
    confidence: float,
    target_classes: list,
    scale: float,
    orig_w: int,
    orig_h: int,
) -> list:
    """
    output: (1, 7, 8400) → 転置して (8400, 7)
    先頭4列: cx/cy/w/h  残り3列: クラススコア
    戻り値: [(x1, y1, x2, y2, class_name, score), ...]
    """
    preds      = output[0].T
    boxes_xywh = preds[:, :4]
    scores     = np.max(preds[:, 4:], axis=1)
    class_ids  = np.argmax(preds[:, 4:], axis=1)
    mask       = scores > confidence
    boxes_xywh = boxes_xywh[mask]
    scores     = scores[mask]
    class_ids  = class_ids[mask]
    if len(boxes_xywh) == 0:
        return []
    x1 = np.clip((boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) / scale, 0, orig_w)
    y1 = np.clip((boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) / scale, 0, orig_h)
    x2 = np.clip((boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) / scale, 0, orig_w)
    y2 = np.clip((boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) / scale, 0, orig_h)
    results = [
        (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
         CENSOR_CLASS_NAMES[class_ids[i]], float(scores[i]))
        for i in range(len(x1))
        if CENSOR_CLASS_NAMES.get(class_ids[i]) in target_classes
    ]
    return _nms(results)


def apply_mosaic(
    image_pil: Image.Image,
    detections: list,
    method: str,
    mosaic_strength: int,
    blur_radius: int,
) -> Image.Image:
    from imgutils.operate.censor_ import censor_areas
    areas = [(d[0], d[1], d[2], d[3]) for d in detections]
    if not areas:
        return image_pil
    if method == "モザイク":
        return censor_areas(image_pil, "pixelate", areas=areas, radius=mosaic_strength)
    elif method == "ぼかし":
        return censor_areas(image_pil, "blur", areas=areas, radius=blur_radius * 2 + 1)
    else:  # 黒塗り
        return censor_areas(image_pil, "color", areas=areas, color=(0, 0, 0))

# ---------------------------------------------------------------------------
# リクエストハンドラ
# ---------------------------------------------------------------------------
def handle_request(conn: socket.socket, token: str):
    try:
        conn.settimeout(60.0)
        request_data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            request_data += chunk

        if not request_data:
            return

        req    = json.loads(request_data.decode("utf-8"))
        action = req.get("action")

        # トークン検証
        if req.get("token") != token:
            conn.sendall(
                json.dumps({"status": "error", "message": "Invalid token"}).encode("utf-8")
            )
            return

        # --- ping ---
        if action == "ping":
            conn.sendall(json.dumps({"status": "alive"}).encode("utf-8"))
            return

        # --- register ---
        if action == "register":
            global _client_count
            with _client_count_lock:
                _client_count += 1
                count = _client_count
            conn.sendall(json.dumps({"status": "ok", "client_count": count}).encode("utf-8"))
            print(f"[Worker] Client registered. Total: {count}", flush=True)
            return

        # --- unregister ---
        if action == "unregister":
            with _client_count_lock:
                _client_count = max(0, _client_count - 1)
                count = _client_count
            conn.sendall(json.dumps({"status": "ok", "client_count": count}).encode("utf-8"))
            print(f"[Worker] Client unregistered. Total: {count}", flush=True)
            if count <= 0:
                print("[Worker] No clients remaining. Shutting down in 3s...", flush=True)
                threading.Timer(3.0, lambda: os._exit(0)).start()
            return

        # --- shutdown ---
        if action == "shutdown":
            conn.sendall(json.dumps({"status": "ok"}).encode("utf-8"))
            print("[Worker] Shutdown requested. Exiting in 1s...", flush=True)
            threading.Timer(1.0, lambda: os._exit(0)).start()
            return

        # ================================================================
        # action: "tag"  –  タギング推論
        # ================================================================
        if action == "tag":
            model_id            = req.get("model")
            input_path          = req.get("input_path")
            device_req          = req.get("device", "NPU")
            threshold           = req.get("threshold", 0.35)
            threshold_character = req.get("threshold_character", 0.75)
            threshold_copyright = req.get("threshold_copyright", 0.50)
            exclude_tags        = req.get("exclude_tags", "")
            replace_underscores = req.get("replace_underscores", True)
            raw_scores_mode     = req.get("raw_scores", False)

            SUPPORTED = {"camie", "anima", "wd14", "joytag"}
            if model_id not in SUPPORTED:
                raise ValueError(f"Unknown model: {model_id}")

            # タグ辞書 Lazy Load
            tags, tag_to_category = _get_tags(model_id)
            if not tags:
                raise RuntimeError(f"Tags for {model_id} failed to load.")

            # 画像読み込み
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input image not found: {input_path}")
            image_pil = Image.open(input_path).convert("RGB")

            # セッション取得（Lazy Load）
            session, actual_backend = get_session(model_id, device_req)

            # 前処理・推論
            input_name = session.get_inputs()[0].name

            if model_id == "camie":
                arr          = preprocess_camie(image_pil)
                output_names = [o.name for o in session.get_outputs()]
                refined      = next(
                    (n for n in output_names if n == "refined_predictions"), output_names[0]
                )
                raw   = session.run([refined], {input_name: arr})[0][0]
                probs = 1 / (1 + np.exp(-np.clip(raw, -88, 88)))

            elif model_id == "anima":
                arr          = preprocess_anima(image_pil)
                output_names = [o.name for o in session.get_outputs()]
                preferred    = ("prediction", "logits")
                chosen       = next((n for n in preferred if n in output_names), output_names[0])
                raw   = session.run([chosen], {input_name: arr})[0][0]
                probs = raw if chosen == "prediction" else 1 / (1 + np.exp(-np.clip(raw, -88, 88)))

            elif model_id == "wd14":
                arr   = preprocess_wd14(image_pil)
                probs = session.run(None, {input_name: arr})[0][0]

            elif model_id == "joytag":
                arr   = preprocess_joytag(image_pil)
                probs = session.run(None, {input_name: arr})[0][0]

            # 推論後に last_used 更新
            with _session_lock:
                entry = _session_cache.get((model_id, device_req))
                if entry:
                    entry.touch()

            # タグフィルタリング
            exclude_set = {t.strip().lower().replace("_", " ") for t in exclude_tags.split(",") if t.strip()}
            thresholds  = {
                "general":   threshold,
                "character": threshold_character,
                "copyright": threshold_copyright,
                "artist":    threshold,
                "meta":      threshold,
                "rating":    threshold,
                "year":      threshold,
            }
            categorized: dict = {c: [] for c in thresholds}

            for idx, prob in enumerate(probs):
                if idx >= len(tags):
                    break
                tag_name = tags[idx]
                if not tag_name:
                    continue
                cat           = tag_to_category.get(tag_name, "general")
                cat_threshold = thresholds.get(cat, threshold)
                if prob < cat_threshold:
                    continue
                if tag_name.lower().replace("_", " ") in exclude_set:
                    continue
                if model_id == "anima":
                    display_tag = normalize_anima_tag(tag_name) if replace_underscores else tag_name
                else:
                    display_tag = tag_name.replace("_", " ") if replace_underscores else tag_name
                categorized[cat].append((display_tag, float(prob)))

            # raw_scores モード（image_sorter 向け）
            if raw_scores_mode:
                raw_dict = {}
                for items in categorized.values():
                    for display_tag, prob in items:
                        raw_dict[display_tag.replace(" ", "_")] = prob
                conn.sendall(json.dumps({
                    "status":         "ok",
                    "model":          model_id,
                    "actual_backend": actual_backend,
                    "raw_scores":     raw_dict,
                }).encode("utf-8"))
                return

            # 通常レスポンス構築
            cat_order     = ["general", "character", "copyright", "artist", "meta", "rating", "year"]
            all_tags_list = []
            formatted: dict = {}
            for cat in cat_order:
                items          = sorted(categorized[cat], key=lambda x: x[1], reverse=True)
                tag_strings    = [x[0] for x in items]
                formatted[cat] = ", ".join(tag_strings)
                all_tags_list.extend(tag_strings)

            conn.sendall(json.dumps({
                "status":         "ok",
                "model":          model_id,
                "actual_backend": actual_backend,
                "all_tags":       ", ".join(all_tags_list),
                "general_tags":   formatted["general"],
                "character_tags": formatted["character"],
                "copyright_tags": formatted["copyright"],
                "artist_tags":    formatted["artist"],
                "meta_tags":      formatted["meta"],
                "rating_tags":    formatted["rating"],
                "year_tags":      formatted["year"],
            }).encode("utf-8"))
            return

        # ================================================================
        # action: "mosaic"  –  NSFW モザイク処理
        # ================================================================
        if action == "mosaic":
            input_path      = req.get("input_path")
            output_path     = req.get("output_path")
            device_req      = req.get("device", "NPU")
            confidence      = req.get("confidence", 0.35)
            method          = req.get("method", "モザイク")
            mosaic_strength = req.get("mosaic_strength", 20)
            blur_radius     = req.get("blur_radius", 10)

            target_classes = []
            if req.get("detect_nipple_f", True): target_classes.append("nipple_f")
            if req.get("detect_penis",    True): target_classes.append("penis")
            if req.get("detect_pussy",    True): target_classes.append("pussy")

            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input image not found: {input_path}")
            image_pil = Image.open(input_path).convert("RGB")

            session, actual_backend = get_session("censor_detect", device_req)
            arr, scale, orig_w, orig_h = preprocess_censor(image_pil)
            output     = session.run(None, {session.get_inputs()[0].name: arr})[0]
            detections = postprocess_censor(output, confidence, target_classes, scale, orig_w, orig_h)

            # 推論後に last_used 更新
            with _session_lock:
                entry = _session_cache.get(("censor_detect", device_req))
                if entry:
                    entry.touch()

            processed = apply_mosaic(image_pil, detections, method, mosaic_strength, blur_radius)
            processed.save(output_path, format="BMP")

            conn.sendall(json.dumps({
                "status":         "ok",
                "actual_backend": actual_backend,
                "detections":     len(detections),
            }).encode("utf-8"))
            return

        # 未知の action
        conn.sendall(
            json.dumps({"status": "error", "message": f"Unknown action: {action}"}).encode("utf-8")
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        try:
            conn.sendall(
                json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
            )
        except Exception as send_err:
            print(f"[Worker] Failed to send error response: {send_err}", flush=True)
    finally:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------
def main():
    # バックグラウンドで Idle Cleanup を起動
    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    cleanup_thread.start()

    # TCP サーバー起動（起動時にモデルはロードしない）
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(8)
    token = secrets.token_hex(4)

    with open(PID_FILE, "w") as f:
        json.dump({"pid": os.getpid(), "port": port, "token": token}, f)

    # 親プロセスが ready JSON をパースしてポート・トークンを取得する
    print(json.dumps({"status": "ready", "port": port, "token": token}), flush=True)

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            conn, _ = server_sock.accept()
            executor.submit(handle_request, conn, token)
    except KeyboardInterrupt:
        pass
    finally:
        server_sock.close()
        try:
            os.unlink(PID_FILE)
        except Exception:
            pass


if __name__ == "__main__":
    main()
