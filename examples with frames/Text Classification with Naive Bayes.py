import random
import jieba 
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score
import re,string 

def remove_punctuation(text):
    # 使用正则表达式匹配英文标点、中文标点和特殊符号
    pattern = r'[^\w]|_'
    # 将匹配到的字符替换为空字符
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def text_to_words(file_path):
    # 定义一个空列表，用于存储句子
    sentences_arr = []
    # 定义一个空列表，用于存储标签
    lab_arr = []

    # 打开文件，以只读模式读取，编码格式为utf-8
    with open(file_path,'r',encoding='utf-8') as f:
        # 遍历文件的每一行
        for line in f.readlines():
            # 将每一行按照'_!_'分割，并将分割后的第二部分添加到标签列表中
            lab_arr.append(line.split('_!_')[1])
            # 将每一行按照'_!_'分割，并将分割后的最后一部分去除空格和标点符号，然后使用jieba分词，将分词结果添加到句子列表中
            sentence= line.split('_!_')[-1].strip()
            #去除标点符号
            sentence=remove_punctuation(sentence)
            sentence=jieba.lcut(sentence,cut_all=False)
            sentences_arr.append(sentence)
    # 返回句子列表和标签列表
    return sentences_arr,lab_arr

#加载停用词表
def load_stopwords(file_path):
    stopwords=[line.strip() for line in open(file_path,'r',encoding='utf-8').readlines()]
    return stopwords

#词频统计
def get_dict(sentences_arr,stopwords):
    # 定义一个空字典，用于存储词频
    word_dict = {}
    # 遍历句子列表
    for sentence in sentences_arr:
        # 遍历句子中的每个词
        for word in sentence:
            if word != '' and word.isalpha():
                # 如果词不在停用词列表中，则将其添加到字典中，并将词频加1
                if word not in stopwords:
                    word_dict[word] = word_dict.get(word, 1) + 1
    #按词频排序
    word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    # 返回词频字典
    return word_dict

#构建词表，过滤掉频率低于word_num的词
def get_feature_words(word_dict,word_num):
    '''
    从词典中选取N个特征词，形成特征词列表
    '''
    n=0
    feature_words=[]
    for word in word_dict:
        if n<word_num:
            feature_words.append(word[0])
        n+=1
    return feature_words

#文本特征表示
def get_text_features(train_data_list,test_data_list,feature_words):
    #根据特征词，将文本转化为特征向量
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    #将训练集和测试集转化为特征向量
    train_feature_list= [text_features(text, feature_words) for text in train_data_list]
    test_feature_list= [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list,test_feature_list
