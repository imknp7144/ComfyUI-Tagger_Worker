from .node_setup import send_mosaic_request


class NSFWMosaicWorkerNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":           ("IMAGE",),
                "detect_nipple_f": ("BOOLEAN", {"default": True}),
                "detect_penis":    ("BOOLEAN", {"default": True}),
                "detect_pussy":    ("BOOLEAN", {"default": True}),
                "method":          (["モザイク", "ぼかし", "黒塗り"], {"default": "モザイク"}),
                "mosaic_strength": ("INT",   {"default": 20,   "min": 1,   "max": 50,  "step": 1}),
                "blur_radius":     ("INT",   {"default": 10,   "min": 1,   "max": 20,  "step": 1}),
                "confidence":      ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "device":          (["NPU", "CPU"], {"default": "NPU"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("image", "actual_backend")
    FUNCTION      = "apply"
    CATEGORY      = "Image/Tagger"

    def apply(
        self, image,
        detect_nipple_f, detect_penis, detect_pussy,
        method, mosaic_strength, blur_radius, confidence, device,
    ):
        output_image, actual_backend = send_mosaic_request(
            image           = image,
            device          = device,
            detect_nipple_f = detect_nipple_f,
            detect_penis    = detect_penis,
            detect_pussy    = detect_pussy,
            method          = method,
            mosaic_strength = mosaic_strength,
            blur_radius     = blur_radius,
            confidence      = confidence,
        )
        return (output_image, actual_backend)
