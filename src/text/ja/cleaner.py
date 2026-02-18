from jaconv import jaconv

REMOVE_CHARACTERS = ["「", "」", "（", "）"]

def japanese_text_cleaner(text: str) -> str:
    text = jaconv.normalize(text)
    for char in REMOVE_CHARACTERS:
        text = text.replace(char, "")
    return text
    