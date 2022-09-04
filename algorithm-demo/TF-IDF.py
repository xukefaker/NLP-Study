import json
import jieba
import utilities.token_filter as tf

#通过json文件来控制一些重要的参数，比如news的数量
with open("parameters.json") as f:
    parameters = json.load(f)
news = []
news_in_words = []
news_num = parameters["news_num"]

for i in range(news_num):
    file = open("../resource/news-"+str(i), mode="r")
    news.append(file.read())
    word_list_temp = jieba.lcut_for_search(news[i]) # 先把非中文的item干掉
    tf.remove_non_chinese(word_list_temp)
    tf.remove_space(word_list_temp)
    news_in_words.append(word_list_temp)
    file.close()

print(news_in_words[1])







