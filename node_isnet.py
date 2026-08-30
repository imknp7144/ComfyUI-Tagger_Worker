from .node_setup import send_isnet_request


class IsnetAnimeWorkerNode:
    """
    isnet-anime によるアニメキャラクター前景/背景セグメンテーション。
    出力される mask は前景(キャラクター)側が1、背景側が0。
    背景だけを残したい場合は InvertMask -> MaskImage 等と組み合わせる
    (既存の RembgByBiRefNet 差し替えとして同じ配線パターンで使える)。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "device": (["NPU", "CPU"], {"default": "NPU"}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "actual_backend")
    FUNCTION     = "apply"
    CATEGORY     = "Image/Tagger"

    def apply(self, image, device):
        mask, actual_backend = send_isnet_request(image=image, device=device)
        return (mask, actual_backend)
