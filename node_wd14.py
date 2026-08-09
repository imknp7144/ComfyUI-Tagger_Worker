from .node_setup import send_tag_request

class WD14TaggerWorkerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold_character": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold_copyright": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exclude_tags": ("STRING", {"default": "", "multiline": True}),
                "replace_underscores": ("BOOLEAN", {"default": True}),
                "device": (["NPU", "CPU"], {"default": "NPU"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("all_tags", "general_tags", "character_tags", "copyright_tags", "artist_tags", "rating_tags", "actual_backend")
    FUNCTION = "tag"
    CATEGORY = "Image/Tagger"

    def tag(self, image, threshold, threshold_character, threshold_copyright, exclude_tags, replace_underscores, device):
        results = send_tag_request(
            model_id="wd14",
            image=image,
            threshold=threshold,
            threshold_character=threshold_character,
            threshold_copyright=threshold_copyright,
            exclude_tags=exclude_tags,
            replace_underscores=replace_underscores,
            device=device
        )
        
        all_tags_list = []
        general_tags_list = []
        character_tags_list = []
        copyright_tags_list = []
        artist_tags_list = []
        rating_tags_list = []
        
        actual_backend = results[0].get("actual_backend", "unknown") if results else "unknown"
        
        for res in results:
            all_tags_list.append(res.get("all_tags", ""))
            general_tags_list.append(res.get("general_tags", ""))
            character_tags_list.append(res.get("character_tags", ""))
            copyright_tags_list.append(res.get("copyright_tags", ""))
            artist_tags_list.append(res.get("artist_tags", ""))
            rating_tags_list.append(res.get("rating_tags", ""))
            
        def _unwrap(lst):
            return lst[0] if len(lst) == 1 else "\n---\n".join(lst)
            
        return (
            _unwrap(all_tags_list),
            _unwrap(general_tags_list),
            _unwrap(character_tags_list),
            _unwrap(copyright_tags_list),
            _unwrap(artist_tags_list),
            _unwrap(rating_tags_list),
            actual_backend
        )
