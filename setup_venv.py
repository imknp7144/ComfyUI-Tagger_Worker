import os
import sys
import subprocess
import json
import hashlib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VENV_DIR   = os.path.join(BASE_DIR, "venv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REQ_PATH   = os.path.join(BASE_DIR, "requirements_worker.txt")
SETUP_COMPLETE_PATH = os.path.join(BASE_DIR, "setup_complete.json")

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

    all_files = [
        camie_onnx, camie_meta,
        anima_onnx, anima_tags,
        wd14_onnx,  wd14_tags,
        joytag_onnx, joytag_tags,
        censor_onnx,
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
        }
        with open(SETUP_COMPLETE_PATH, "w") as f:
            json.dump({
                "schema":     2,
                "models":     ["camie", "anima", "wd14", "joytag", "censor_detect"],
                "ort_openvino": "1.24.1",
                "openvino":   "2025.4.1",
                "hashes":     hashes,
            }, f, indent=2)

        print_green("[Setup] すべてのセットアップが完了しました！")

    except Exception as e:
        print_red(f"\n[Setup ERROR] セットアップに失敗しました: {e}")
        print_yellow(f"[Setup ERROR]   1. {VENV_DIR} を削除")
        print_yellow(f"[Setup ERROR]   2. {MODELS_DIR} を削除")
        print_yellow("[Setup ERROR]   3. Setup Worker Environment ノードを再実行")
        sys.exit(1)


if __name__ == "__main__":
    main()
