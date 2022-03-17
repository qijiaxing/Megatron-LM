# JQ: split Chinese doc into sentences by regex
# modified from https://stackoverflow.com/questions/27441191/splitting-chinese-document-into-sentences
import re
from zhconv import convert

def zng(para):
    """
      Split a paragraph into sentences by ending symbols
    """
    for sent in re.findall(u'[^!?。！？\!\?\n]+[!?。\!\?]?[”’" 」 ）]*', para, flags=re.U):
        yield sent

def has_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)

def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])

def stringpartQ2B(ustring):
    """把字符串中数字和字母全角转半角"""
    return "".join([Q2B(uchar) if is_Qnumber(uchar) or is_Qalphabet(uchar) else uchar for uchar in ustring])

punct_dict = {
  '–':'-',   # En Dash (/u2012) -> hyphen
  '—':'-',   # Em Dash (/u2014) -> hyphen
  '▼':'',    # Black Down-Pointing Triangle (/u25BC)
  '℃':'摄氏度',    # Degree Celsius (/u2103)
  '①':'一', '②':'二', '③': '三', '④':'四', '⑤':'五',
}
punct_table = str.maketrans(punct_dict)
# txt.translate(punct_table)

def translate_punct(ustring):
  return ustring.translate(punct_table)

def to_zh_cn(string):
    """
      JQ: Convert string to Simplified Chinese
    """
    return convert(string, "zh-cn")
