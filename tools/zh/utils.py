# JQ: split Chinese doc into sentences by regex
# modified from https://stackoverflow.com/questions/27441191/splitting-chinese-document-into-sentences
import re
def zng(para):
    """
      Split a paragraph into sentences by ending symbols
    """
    for sent in re.findall(u'[^!?。！？\!\?\n]+[!?。\!\?]?[”’" 」 ）]*', para, flags=re.U):
        yield sent

def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False
