_vowel = "a i u e o"
_unvoiced_vowel = "A I U E O"
_consonant = "b d f g h j k m n p r s t v w y z by dy gy hy ky my ny py ry ty ch sh ts"
_other = "cl N"
_pause = "pau"

_prosody = "_ ^ $ ? # ] ["

PHONEME_JP = (_vowel.split() + _unvoiced_vowel.split() + _consonant.split() + _other.split() + _pause.split())
PROSODY_JP = _prosody.split()

PHONEME_MAP_JP = {s: i + 1 for i, s in enumerate(PHONEME_JP)}
PROSODY_MAP_JP = {s: i + 1 for i, s in enumerate(PROSODY_JP)}
