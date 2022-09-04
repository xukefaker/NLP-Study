'''
v1.0:字符过滤器
'''


def is_chinese(ch):
    # 判断一个unicode是否是中文字符
    return '\u4e00' <= ch <= '\u9fff'


def is_alpha_lower(ch):
    # 判断一个unicode是否是小写字母
    return '\u0061' <= ch <= '\u007a'


def is_alpha_upper(ch):
    # 判断一个unicode是否是大写字母
    return '\u0041' <= ch <= '\u005a'


def is_alpha(ch):
    # 判断一个unicode是否是字母（可以是大写也可以是小写）
    return is_alpha_lower(ch) or is_alpha_upper(ch)


def is_number(ch):
    # 判断一个unicode是否是数字  此函数用str.isdigit()代替也可
    return '\u0030' <= ch <= '\u0039'


def is_chinese_string(words):
    # 判断一个字符串是否仅包含中文
    for ch in words:
        if not is_chinese(ch):
            return False
    return True


def is_alpha_string(words):
    # 判断一个字符串是否仅包含字母
    for ch in words:
        if not is_alpha(ch):
            return False
    return True


def remove_non_chinese(words):
    # 去除一个list中非中文的item
    for word in words:
        if not is_chinese_string(word):
            words.remove(word)


def remove_space(words):
    # 去除一个list中的''和'\n'
    while '' in words:
        words.remove('')
    while '\n' in words:
        words.remove('\n')


