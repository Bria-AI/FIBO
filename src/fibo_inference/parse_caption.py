
import math

import ujson
from boltons.iterutils import remap


def clean_json(caption):
    caption['pickascore']=1.0
    caption['aesthetic_score']=10.0    
    caption = prepare_clean_caption(caption)
    return caption

def parse_aesthetic_score(record: dict) -> str:
    ae = record["aesthetic_score"]
    if ae < 5.5:
        return "very low"
    elif ae < 6:
        return "low"
    elif ae < 7:
        return "medium"
    elif ae < 7.6:
        return "high"
    else:
        return "very high"

def parse_pickascore(record: dict) -> str:
    ps = record["pickascore"]
    if ps < 0.78:
        return "very low"
    elif ps < 0.82:
        return "low"
    elif ps < 0.87:
        return "medium"
    elif ps < 0.91:
        return "high"
    else:
        return "very high"

def prepare_clean_caption(record: dict) -> str:
    def keep(p, k, v):
        is_none = v is None
        is_empty_string = isinstance(v, str) and v == ""
        is_empty_dict = isinstance(v, dict) and not v
        is_empty_list = isinstance(v, list) and not v
        is_nan = isinstance(v, float) and math.isnan(v)   
        if is_none or is_empty_string or is_empty_list or is_empty_dict or is_nan:
            return False
        return True

    try:
        
        scores = {}
        if 'pickascore' in record:
            scores["preference_score"] = parse_pickascore(record)
        if 'aesthetic_score' in record:
            scores["aesthetic_score"] = parse_aesthetic_score(record)        


        # Create structured caption dict of original values
        fields=[
            "short_description",
            "objects",
            "background_setting",
            "lighting",
            "aesthetics",
            "photographic_characteristics",
            "style_medium",
            "text_render",
            "context",
            "artistic_style",
            ]

        original_caption_dict = {f:record[f] for f in fields if f in record}

        # filter empty values recursivly (i.e. None, "", {}, [], float("nan"))
        clean_caption_dict = remap(original_caption_dict, visit=keep)

        # Set aesthetics scores
        if "aesthetics" not in clean_caption_dict:
            if len(scores)>0:
                clean_caption_dict["aesthetics"] = scores
        else:
            clean_caption_dict["aesthetics"].update(scores)

        # Dumps clean structured caption as minimal json string (i.e. no newlines\whitespaces seps)
        clean_caption_str = ujson.dumps(clean_caption_dict, escape_forward_slashes=False)
        return clean_caption_str
    except Exception as ex:
        print("Error: ", ex)
        raise ex