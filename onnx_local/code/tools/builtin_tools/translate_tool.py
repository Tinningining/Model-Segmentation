"""
翻译工具 - 调用 MyMemory 翻译 API（免费，无需 API Key）
"""
import json

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


def execute(text: str, from_lang: str = "zh", to_lang: str = "en") -> dict:
    """
    翻译文本（调用 MyMemory API）

    Args:
        text: 要翻译的文本
        from_lang: 源语言 (zh/en/ja/ko/fr/de/es 等)
        to_lang: 目标语言

    Returns:
        翻译结果字典
    """
    if from_lang == to_lang:
        return {"text": text, "translated": text, "from": from_lang, "to": to_lang}

    # 语言代码映射（MyMemory 使用 ISO 639-1）
    lang_map = {
        "zh": "zh-CN", "en": "en-GB", "ja": "ja-JP",
        "ko": "ko-KR", "fr": "fr-FR", "de": "de-DE", "es": "es-ES",
    }
    src = lang_map.get(from_lang, from_lang)
    tgt = lang_map.get(to_lang, to_lang)

    if not HAS_URLLIB:
        return {"text": text, "error": "urllib not available", "from": from_lang, "to": to_lang}

    try:
        encoded_text = urllib.request.quote(text)
        url = "https://api.mymemory.translated.net/get?q=%s&langpair=%s|%s" % (
            encoded_text, src, tgt
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        response_data = data.get("responseData", {})
        translated = response_data.get("translatedText", "")

        if not translated or "MYMEMORY WARNING" in translated.upper():
            return {"text": text, "translated": translated or text,
                    "from": from_lang, "to": to_lang, "warning": "翻译质量可能不佳"}

        return {
            "text": text,
            "translated": translated,
            "from": from_lang,
            "to": to_lang,
            "source": "mymemory",
        }
    except Exception as e:
        print("[translate_tool] API failed (%s), using fallback dict" % str(e))
        return _fallback(text, from_lang, to_lang)


# 兜底词典
_DICT_ZH_EN = {
    "你好": "Hello", "谢谢": "Thank you", "再见": "Goodbye",
    "早上好": "Good morning", "晚上好": "Good evening",
    "天气": "weather", "北京": "Beijing", "上海": "Shanghai",
    "今天": "today", "明天": "tomorrow", "温度": "temperature",
    "我": "I", "你": "you", "是": "am/is/are",
    "什么": "what", "怎么": "how", "好的": "OK",
}
_DICT_EN_ZH = {v.lower(): k for k, v in _DICT_ZH_EN.items()}


def _fallback(text: str, from_lang: str, to_lang: str) -> dict:
    """字典兜底翻译"""
    if from_lang == "zh" and to_lang == "en":
        # 先尝试整句匹配
        if text in _DICT_ZH_EN:
            translated = _DICT_ZH_EN[text]
        else:
            # 逐词替换
            translated = text
            for zh, en in sorted(_DICT_ZH_EN.items(), key=lambda x: -len(x[0])):
                translated = translated.replace(zh, en)
    elif from_lang == "en" and to_lang == "zh":
        words = text.split()
        parts = []
        for w in words:
            clean = w.lower().strip(".,!?;:")
            parts.append(_DICT_EN_ZH.get(clean, w))
        translated = "".join(parts)
    else:
        translated = text

    return {"text": text, "translated": translated,
            "from": from_lang, "to": to_lang, "source": "fallback_dict"}


TOOL_CONFIG = {
    'name': 'translate',
    'module_path': 'tools.builtin_tools.translate_tool',
    'memory_size': 20,
    'description': '翻译文本，支持中英日韩法德西等语言互译',
    'parameters': {
        'text': {'type': 'string', 'required': True, 'description': '要翻译的文本'},
        'from_lang': {'type': 'string', 'required': False, 'default': 'zh',
                      'description': '源语言代码，如：zh, en, ja, ko, fr, de, es'},
        'to_lang': {'type': 'string', 'required': False, 'default': 'en',
                    'description': '目标语言代码，如：zh, en, ja, ko, fr, de, es'},
    }
}
