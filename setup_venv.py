import os
import sys
import subprocess
import json
import hashlib

import ensemble_catalog

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VENV_DIR   = os.path.join(BASE_DIR, "venv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REQ_PATH   = os.path.join(BASE_DIR, "requirements_worker.txt")
SETUP_COMPLETE_PATH = os.path.join(BASE_DIR, "setup_complete.json")
DEVICE_CAPS_PATH    = os.path.join(BASE_DIR, "device_capabilities.json")

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"

def print_green(text):  print(f"{GREEN}{text}{RESET}")
def print_yellow(text): print(f"{YELLOW}{text}{RESET}")
def print_red(text):    print(f"{RED}{text}{RESET}")


def run_command(cmd, cwd=None):
    process = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode


def calculate_sha256(filepath):
    if not os.path.exists(filepath):
        return ""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def probe_device_capabilities(venv_python):
    """
    venv内のopenvinoでCore().available_devicesを取得し、
    NPUが実機に存在するかどうかをdevice_capabilities.jsonへ書き出す。
    ノード側(node_setup.py)はこのファイルを見て、NPUが無ければ
    deviceドロップダウンの選択肢からNPUを外す。
    判定自体に失敗した場合は"unknown"として記録し、
    ノード側では安全側(NPUを選択肢に残す)にフォールバックさせる。
    """
    print_green("[Setup] OpenVINO対応デバイス(NPU有無)を確認しています...")
    probe_script = r"""
import json
try:
    from openvino import Core
    available = Core().available_devices
    ok = True
except Exception as e:
    available = []
    ok = False
    print(f"[Setup] OpenVINO device query failed: {e}")

has_npu = ok and any(d == "NPU" or str(d).startswith("NPU") for d in available)
result = {"available_devices": available, "has_npu": has_npu, "probe_ok": ok}
with open(r"__CAPS_PATH__", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"[Setup] Detected OpenVINO devices: {available} (has_npu={has_npu})")
"""
    script = probe_script.replace("__CAPS_PATH__", DEVICE_CAPS_PATH)
    ret = subprocess.call([venv_python, "-c", script])
    if ret != 0:
        print_yellow(
            "[Setup] デバイス判定に失敗しました。NPUオプションは既定で選択肢に残ります"
            "(実際にNPUが無ければworker側の自動フォールバックでiGPU/CPUに切り替わります)。"
        )
        try:
            with open(DEVICE_CAPS_PATH, "w", encoding="utf-8") as f:
                json.dump({"available_devices": [], "has_npu": True, "probe_ok": False}, f, indent=2)
        except Exception:
            pass


def ensure_ensemble_models(venv_python):
    """
    tagger_ensemble_worker由来の6タガー(iGPU/OpenVINO経由)のモデルファイルを確認する。
    - 非gated(cl_v1, oppai_v11, wd_eva02_l): 未配置なら自動ダウンロードする
    - gated(cl_v2, dtq_l16, dtq_b16): 利用規約への同意が必要なため自動ダウンロードせず、
      未配置の場合は配布元と配置先パスを案内するだけに留める(セットアップ全体は失敗させない)
    """
    print_green("[Setup] Tagger Ensemble Worker由来のタガー(iGPU用、6種)を確認しています...")

    to_download = []
    for entry in ensemble_catalog.ENSEMBLE_MODEL_CATALOG:
        model_dir     = os.path.join(MODELS_DIR, entry.model_id)
        tags_path     = os.path.join(model_dir, entry.tags_filename)
        model_path    = os.path.join(model_dir, "model.onnx")
        category_path = os.path.join(model_dir, entry.category_filename) if entry.category_filename else None
        required       = [model_path, tags_path] + ([category_path] if category_path else [])

        if all(os.path.exists(p) for p in required):
            print_green(f"[Setup]   [OK] {entry.model_id}: 配置済みです。")
            continue

        if entry.gated:
            missing_desc = ", ".join(required)
            notes_desc   = f" ({entry.notes})" if entry.notes else ""
            print_yellow(
                f"[Setup]   [ACTION REQUIRED] {entry.model_id}: 利用規約への同意が必要な配布元のため、"
                f"自動ダウンロードは行いません。{entry.source_url} から手動でダウンロードし、"
                f"次のパスに配置してください: {missing_desc}{notes_desc}"
            )
            continue

        to_download.append((entry, model_dir, model_path, tags_path))

    for entry, model_dir, model_path, tags_path in to_download:
        print_green(f"[Setup]   {entry.model_id} を自動ダウンロードしています...")
        os.makedirs(model_dir, exist_ok=True)
        downloader_script = f"""
import shutil
from huggingface_hub import hf_hub_download
try:
    p1 = hf_hub_download(repo_id={entry.repo_id!r}, filename={entry.repo_model_filename!r})
    shutil.copy(p1, r"{model_path}")
    p2 = hf_hub_download(repo_id={entry.repo_id!r}, filename={entry.repo_tags_filename!r})
    shutil.copy(p2, r"{tags_path}")
    print("[Setup] Downloaded {entry.model_id}")
except Exception as e:
    print(f"[Setup ERROR] Failed to download {entry.model_id}: {{e}}")
    raise
"""
        ret = run_command([venv_python, "-c", downloader_script])
        if ret != 0:
            print_yellow(
                f"[Setup]   [ERROR] {entry.model_id} の自動ダウンロードに失敗しました。"
                f"{entry.source_url} から手動配置することもできます: {model_path}"
            )


def main():
    print_green("[Setup] ComfyUI Tagger Worker セットアップを開始します。")

    # モデルファイルパス定義
    camie_onnx    = os.path.join(MODELS_DIR, "camie",  "model.onnx")
    camie_meta    = os.path.join(MODELS_DIR, "camie",  "camie-tagger-v2-metadata.json")
    anima_onnx    = os.path.join(MODELS_DIR, "anima",  "model.onnx")
    anima_tags    = os.path.join(MODELS_DIR, "anima",  "selected_tags.csv")
    wd14_onnx     = os.path.join(MODELS_DIR, "wd14",   "model.onnx")
    wd14_tags     = os.path.join(MODELS_DIR, "wd14",   "selected_tags.csv")
    joytag_onnx   = os.path.join(MODELS_DIR, "joytag", "model.onnx")
    joytag_tags   = os.path.join(MODELS_DIR, "joytag", "top_tags.txt")
    censor_onnx   = os.path.join(MODELS_DIR, "censor_detect_v0.10_s", "model.onnx")
    isnet_onnx    = os.path.join(MODELS_DIR, "isnet_anime", "model.onnx")

    all_files = [
        camie_onnx, camie_meta,
        anima_onnx, anima_tags,
        wd14_onnx,  wd14_tags,
        joytag_onnx, joytag_tags,
        censor_onnx,
        isnet_onnx,
    ]
    all_files_exist = all(os.path.exists(f) for f in all_files)

    if sys.platform == "win32":
        venv_python_check = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        venv_python_check = os.path.join(VENV_DIR, "bin", "python")

    # スキップ判定
    if os.path.exists(SETUP_COMPLETE_PATH) and all_files_exist and os.path.exists(venv_python_check):
        try:
            with open(SETUP_COMPLETE_PATH, "r") as f:
                data = json.load(f)
            stored_hashes = data.get("hashes", {})
            current_hashes = {
                "camie_onnx":   calculate_sha256(camie_onnx),
                "anima_onnx":   calculate_sha256(anima_onnx),
                "wd14_onnx":    calculate_sha256(wd14_onnx),
                "joytag_onnx":  calculate_sha256(joytag_onnx),
                "censor_onnx":  calculate_sha256(censor_onnx),
                "isnet_onnx":   calculate_sha256(isnet_onnx),
            }
            if not all(stored_hashes.get(k) == current_hashes[k] for k in current_hashes):
                print_yellow("[Setup] モデルファイルの変更または破損を検知しました。再ダウンロードを実行します。")
            else:
                check_ret = subprocess.call(
                    [venv_python_check, "-c",
                     "import numpy, onnxruntime, PIL, huggingface_hub, imgutils"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if check_ret == 0:
                    print_green(
                        f"[Setup] すでにセットアップ済みで、モデルファイルの整合性も確認できました"
                        f" (Schema: {data.get('schema')})。処理をスキップします。"
                    )
                    if not os.path.exists(DEVICE_CAPS_PATH):
                        probe_device_capabilities(venv_python_check)
                    ensure_ensemble_models(venv_python_check)
                    print("[Setup] 再セットアップを行いたい場合は setup_complete.json と venv/ を削除してください。")
                    sys.exit(0)
                else:
                    print_yellow("[Setup] venv内のパッケージが不完全です。再インストールを実行します。")
        except Exception as e:
            print_yellow(f"[Setup] 既存セットアップ検証中にエラー: {e}。再セットアップします。")

    try:
        os.makedirs(MODELS_DIR, exist_ok=True)

        # venv 作成
        python_exe = sys.executable
        print_green(f"[Setup] Python venvを作成しています... ({python_exe})")
        if not os.path.exists(VENV_DIR):
            ret = run_command([python_exe, "-m", "venv", "venv"], cwd=BASE_DIR)
            if ret != 0:
                raise RuntimeError("venvの作成に失敗しました。")
            print_green("[Setup] venvの作成が完了しました。")
        else:
            print("[Setup] venvは既に存在します。")

        if sys.platform == "win32":
            venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(VENV_DIR, "bin", "python")

        if not os.path.exists(venv_python):
            raise RuntimeError(f"venv内のpythonバイナリが見つかりません: {venv_python}")

        # pip アップグレード
        print_green("[Setup] pipをアップグレードしています...")
        run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

        # 依存パッケージインストール
        print_green("[Setup] requirements_worker.txt をインストールしています...")
        if os.path.exists(REQ_PATH):
            ret = run_command([venv_python, "-m", "pip", "install", "-r", REQ_PATH])
        else:
            print_yellow(f"[Setup] {REQ_PATH} が見つかりません。パッケージを直接インストールします。")
            packages = [
                "onnxruntime-openvino==1.24.1",
                "openvino==2025.4.1",
                "Pillow",
                "numpy",
                "huggingface_hub",
                "psutil",
                "dghs-imgutils>=0.19.0",
            ]
            ret = run_command([venv_python, "-m", "pip", "install"] + packages)
        if ret != 0:
            raise RuntimeError("依存パッケージのインストールに失敗しました。")
        print_green("[Setup] 依存ライブラリのインストールが完了しました。")

        # モデルダウンロード
        print_green("[Setup] HuggingFaceからモデルをダウンロードしています...")
        downloader_script = r"""
import os, shutil
from huggingface_hub import hf_hub_download

models = [
    {"repo": "Camais03/camie-tagger-v2",          "file": "camie-tagger-v2.onnx",              "dest": r"__CAMIE_ONNX__"},
    {"repo": "Camais03/camie-tagger-v2",          "file": "camie-tagger-v2-metadata.json",      "dest": r"__CAMIE_META__"},
    {"repo": "deepghs/pixai-tagger-v0.9-onnx",   "file": "model.onnx",                         "dest": r"__ANIMA_ONNX__"},
    {"repo": "deepghs/pixai-tagger-v0.9-onnx",   "file": "selected_tags.csv",                  "dest": r"__ANIMA_TAGS__"},
    {"repo": "SmilingWolf/wd-vit-tagger-v3",      "file": "model.onnx",                         "dest": r"__WD14_ONNX__"},
    {"repo": "SmilingWolf/wd-vit-tagger-v3",      "file": "selected_tags.csv",                  "dest": r"__WD14_TAGS__"},
    {"repo": "fancyfeast/joytag",                  "file": "model.onnx",                         "dest": r"__JOYTAG_ONNX__"},
    {"repo": "fancyfeast/joytag",                  "file": "top_tags.txt",                       "dest": r"__JOYTAG_TAGS__"},
    {"repo": "deepghs/anime_censor_detection",     "file": "censor_detect_v0.10_s/model.onnx",  "dest": r"__CENSOR_ONNX__"},
    {"repo": "tomjackson2023/rembg",               "file": "isnet-anime.onnx",                   "dest": r"__ISNET_ONNX__"},
]

for m in models:
    print(f"[Setup] Downloading {m['repo']}/{m['file']}...")
    os.makedirs(os.path.dirname(m['dest']), exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=m['repo'],
            filename=m['file'],
        )
        shutil.copy(path, m['dest'])
        print(f"[Setup] Saved to {m['dest']}")
    except Exception as e:
        print(f"[Setup ERROR] Failed: {m['repo']}/{m['file']}: {e}")
        raise
"""
        script = (downloader_script
            .replace("__CAMIE_ONNX__",   camie_onnx)
            .replace("__CAMIE_META__",   camie_meta)
            .replace("__ANIMA_ONNX__",   anima_onnx)
            .replace("__ANIMA_TAGS__",   anima_tags)
            .replace("__WD14_ONNX__",    wd14_onnx)
            .replace("__WD14_TAGS__",    wd14_tags)
            .replace("__JOYTAG_ONNX__",  joytag_onnx)
            .replace("__JOYTAG_TAGS__",  joytag_tags)
            .replace("__CENSOR_ONNX__",  censor_onnx)
            .replace("__ISNET_ONNX__",   isnet_onnx)
        )
        ret = run_command([venv_python, "-c", script])
        if ret != 0:
            raise RuntimeError("モデルファイルのダウンロード中にエラーが発生しました。")

        missing = [f for f in all_files if not os.path.exists(f)]
        if missing:
            raise RuntimeError("download incomplete: " + ", ".join(missing))

        # setup_complete.json 書き出し
        print_green("[Setup] モデルファイルのハッシュ値を計算して記録しています...")
        hashes = {
            "camie_onnx":   calculate_sha256(camie_onnx),
            "anima_onnx":   calculate_sha256(anima_onnx),
            "wd14_onnx":    calculate_sha256(wd14_onnx),
            "joytag_onnx":  calculate_sha256(joytag_onnx),
            "censor_onnx":  calculate_sha256(censor_onnx),
            "isnet_onnx":   calculate_sha256(isnet_onnx),
        }
        with open(SETUP_COMPLETE_PATH, "w") as f:
            json.dump({
                "schema":     3,
                "models":     ["camie", "anima", "wd14", "joytag", "censor_detect", "isnet_anime"],
                "ort_openvino": "1.24.1",
                "openvino":   "2025.4.1",
                "hashes":     hashes,
            }, f, indent=2)

        # NPU有無を判定してdevice_capabilities.jsonへ書き出す
        probe_device_capabilities(venv_python)

        # tagger_ensemble_worker由来6モデルの確認/自動ダウンロード
        ensure_ensemble_models(venv_python)

        print_green("[Setup] すべてのセットアップが完了しました！")

    except Exception as e:
        print_red(f"\n[Setup ERROR] セットアップに失敗しました: {e}")
        print_yellow(f"[Setup ERROR]   1. {VENV_DIR} を削除")
        print_yellow(f"[Setup ERROR]   2. {MODELS_DIR} を削除")
        print_yellow("[Setup ERROR]   3. Setup Worker Environment ノードを再実行")
        sys.exit(1)


if __name__ == "__main__":
    main()
