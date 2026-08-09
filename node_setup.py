import os
import sys
import json
import socket
import subprocess
import tempfile
import atexit
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(BASE_DIR, "worker.pid.json")

_worker_proc  = None
_worker_port  = None
_worker_token = None

# ---------------------------------------------------------------------------
# ゾンビ Worker 検出 & 排除
# ---------------------------------------------------------------------------
def _check_and_kill_zombie():
    global _worker_port, _worker_token
    if not os.path.exists(PID_FILE):
        return

    try:
        with open(PID_FILE) as f:
            info = json.load(f)
        pid   = info["pid"]
        port  = info["port"]
        token = info["token"]
    except Exception:
        try:
            os.unlink(PID_FILE)
        except Exception:
            pass
        return

    # ping で生存確認
    alive = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        sock.connect(("127.0.0.1", port))
        sock.sendall(json.dumps({"action": "ping", "token": token}).encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        raw = b""
        while True:
            chunk = sock.recv(1024)
            if not chunk:
                break
            raw += chunk
        sock.close()
        if json.loads(raw.decode("utf-8")).get("status") == "alive":
            alive = True
    except Exception:
        pass

    if alive:
        _worker_port  = port
        _worker_token = token
        return

    # 応答なし → ゾンビを終了
    _worker_port  = None
    _worker_token = None
    try:
        import psutil
        if psutil.pid_exists(pid):
            psutil.Process(pid).terminate()
            print(f"\033[33m[TaggerWorker] 残留workerプロセス(pid={pid})を終了しました\033[0m")
    except Exception:
        pass
    try:
        os.unlink(PID_FILE)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Worker 起動
# ---------------------------------------------------------------------------
def _start_worker():
    global _worker_proc, _worker_port, _worker_token
    _check_and_kill_zombie()
    if _worker_port is not None:
        return

    if sys.platform == "win32":
        venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(BASE_DIR, "venv", "bin", "python")

    worker_script = os.path.join(BASE_DIR, "worker.py")

    if not os.path.exists(venv_python):
        raise RuntimeError(
            "Worker venv が見つかりません。「Setup Tagger Worker Environment」ノードを実行してください。"
        )

    check_ret = subprocess.call(
        [venv_python, "-c", "import numpy, onnxruntime, PIL, huggingface_hub, imgutils"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if check_ret != 0:
        raise RuntimeError(
            "Worker venv のパッケージが不完全です(numpy等が見つかりません)。"
            "setup_complete.json と venv/ を削除してから"
            "「Setup Tagger Worker Environment」ノードを再実行してください。"
        )

    LOG_FILE = os.path.join(BASE_DIR, "worker.log")
    print("\033[32m[TaggerWorker] 常駐workerプロセスを起動しています...\033[0m")

    log_fp = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    _worker_proc = subprocess.Popen(
        [venv_python, "-u", worker_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    ready       = False
    error_lines = []
    for line in _worker_proc.stdout:
        stripped = line.strip()
        print(f"[TaggerWorker Log] {stripped}")
        log_fp.write(line)
        log_fp.flush()
        error_lines.append(stripped)
        if "ready" in line:
            try:
                info = json.loads(stripped)
                if info.get("status") == "ready":
                    _worker_port  = info["port"]
                    _worker_token = info["token"]
                    ready = True
                    break
            except Exception:
                pass

    if not ready or _worker_port is None:
        try:
            remaining = _worker_proc.stdout.read()
            if remaining:
                for ln in remaining.splitlines():
                    print(f"[TaggerWorker Log] {ln.strip()}")
                    log_fp.write(ln + "\n")
                    error_lines.append(ln.strip())
        except Exception:
            pass
        log_fp.close()
        exit_code = _worker_proc.poll()
        tail = "\n".join(error_lines[-20:]) if error_lines else "(出力なし)"
        raise RuntimeError(
            f"Workerプロセスの起動に失敗しました (exit={exit_code})。\n"
            f"--- Worker出力 (末尾20行) ---\n{tail}\n"
            "--- ここまで ---\n"
            "venvが壊れている場合は setup_complete.json と venv/ を削除して"
            "「Setup Tagger Worker Environment」ノードを再実行してください。"
        )

    print(f"\033[32m[TaggerWorker] Workerプロセスとの接続完了 (Port={_worker_port})\033[0m")

    import threading
    def _drain_stdout():
        try:
            for line in _worker_proc.stdout:
                stripped = line.strip()
                if stripped:
                    print(f"[TaggerWorker Log] {stripped}")
                log_fp.write(line)
                log_fp.flush()
        except Exception:
            pass
        finally:
            try:
                log_fp.close()
            except Exception:
                pass
    threading.Thread(target=_drain_stdout, daemon=True).start()


@atexit.register
def _shutdown_worker():
    global _worker_proc
    if _worker_proc and _worker_proc.poll() is None:
        _worker_proc.terminate()
        try:
            _worker_proc.wait(timeout=2.0)
        except Exception:
            pass
    try:
        os.unlink(PID_FILE)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 共通ソケット送受信ヘルパー
# ---------------------------------------------------------------------------
def _reset_worker():
    """Worker 状態をリセット（再起動準備）。"""
    global _worker_proc, _worker_port, _worker_token
    if _worker_proc and _worker_proc.poll() is None:
        try:
            _worker_proc.terminate()
            _worker_proc.wait(timeout=3.0)
        except Exception:
            pass
    if os.path.exists(PID_FILE):
        try:
            os.unlink(PID_FILE)
        except Exception:
            pass
    _worker_proc  = None
    _worker_port  = None
    _worker_token = None


def _log_worker_tail():
    log_path = os.path.join(BASE_DIR, "worker.log")
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
            return "".join(lf.readlines()[-30:])
    except Exception:
        return "(ログ読み取り失敗)"


def _send_recv(params: dict, timeout: float = 120.0, _retry: bool = True) -> dict:
    """Worker に JSON リクエストを送り、JSON レスポンスを返す。
    空レスポンス / ConnectionRefused の場合は Worker を再起動して1回リトライする。
    """
    global _worker_port, _worker_token
    response_data = b""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(("127.0.0.1", _worker_port))
        sock.sendall(json.dumps(params).encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_data += chunk
        sock.close()
    except (ConnectionRefusedError, OSError) as e:
        print(f"\033[33m[TaggerWorker] Worker 接続エラー: {e}\033[0m")
        response_data = b""

    if not response_data:
        if _retry:
            print("\033[33m[TaggerWorker] Worker が無応答です。再起動してリトライします...\033[0m")
            print(f"[TaggerWorker] worker.log 末尾:\n{_log_worker_tail()}")
            _reset_worker()
            _start_worker()
            params["token"] = _worker_token
            return _send_recv(params, timeout, _retry=False)
        raise RuntimeError(
            "Worker process disconnected without response.\n"
            f"worker.log (末尾30行):\n{_log_worker_tail()}"
        )

    res = json.loads(response_data.decode("utf-8"))
    if res.get("status") != "ok":
        raise RuntimeError(res.get("message", "Unknown error in worker."))
    return res


# ---------------------------------------------------------------------------
# タギングリクエスト
# ---------------------------------------------------------------------------
def send_tag_request(
    model_id, image, threshold, threshold_character, threshold_copyright,
    exclude_tags, replace_underscores, device,
):
    _start_worker()

    results  = []
    img_np   = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    for i in range(img_np.shape[0]):
        pil_img    = Image.fromarray(img_np[i])
        fd, input_path = tempfile.mkstemp(prefix="comfyui_tagger_in_", suffix=".bmp")
        os.close(fd)
        try:
            pil_img.save(input_path, format="BMP")
            res = _send_recv({
                "action":              "tag",
                "token":               _worker_token,
                "model":               model_id,
                "input_path":          input_path,
                "threshold":           threshold,
                "threshold_character": threshold_character,
                "threshold_copyright": threshold_copyright,
                "exclude_tags":        exclude_tags,
                "replace_underscores": replace_underscores,
                "device":              device,
            })
            results.append(res)
        finally:
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
            except Exception:
                pass

    return results


# ---------------------------------------------------------------------------
# モザイクリクエスト
# ---------------------------------------------------------------------------
def send_mosaic_request(
    image, device,
    detect_nipple_f, detect_penis, detect_pussy,
    method, mosaic_strength, blur_radius, confidence,
):
    """
    ComfyUI IMAGE テンソル (B, H, W, C) を受け取り、
    モザイク処理済み IMAGE テンソルと actual_backend を返す。
    """
    import torch
    _start_worker()

    output_images   = []
    actual_backends_list = []
    img_np = (image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    for i in range(img_np.shape[0]):
        pil_img     = Image.fromarray(img_np[i])
        fd_in, input_path   = tempfile.mkstemp(prefix="comfyui_nsfw_in_",  suffix=".bmp")
        fd_out, output_path = tempfile.mkstemp(prefix="comfyui_nsfw_out_", suffix=".bmp")
        os.close(fd_in)
        os.close(fd_out)
        try:
            pil_img.save(input_path, format="BMP")
            res = _send_recv({
                "action":          "mosaic",
                "token":           _worker_token,
                "input_path":      input_path,
                "output_path":     output_path,
                "device":          device,
                "detect_nipple_f": detect_nipple_f,
                "detect_penis":    detect_penis,
                "detect_pussy":    detect_pussy,
                "method":          method,
                "mosaic_strength": mosaic_strength,
                "blur_radius":     blur_radius,
                "confidence":      confidence,
            })
            out_np = (
                np.array(Image.open(output_path).convert("RGB")).astype(np.float32) / 255.0
            )
            output_images.append(torch.from_numpy(out_np))
            actual_backends_list.append(res.get("actual_backend", "unknown"))
        finally:
            for path in (input_path, output_path):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass

    actual_backend = actual_backends_list[0] if actual_backends_list else "unknown"
    return torch.stack(output_images, dim=0), actual_backend


# ---------------------------------------------------------------------------
# Setup ノード
# ---------------------------------------------------------------------------
class SetupTaggerWorkerEnvironmentNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION     = "setup"
    OUTPUT_NODE  = True
    CATEGORY     = "Image/Tagger"

    def setup(self):
        setup_script = os.path.join(BASE_DIR, "setup_venv.py")
        print("\033[32m[TaggerWorker] 環境のセットアップを開始します...\033[0m")
        ret = subprocess.call([sys.executable, setup_script], cwd=BASE_DIR)
        if ret == 0:
            print("\033[32m[TaggerWorker] 環境のセットアップが正常に完了しました。\033[0m")
        else:
            print("\033[31m[TaggerWorker] 環境のセットアップに失敗しました。コンソール出力を確認してください。\033[0m")
        return ()
