import util
import numpy as np
from pickle import load
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences
from PIL import Image as pil_image
from matplotlib import pyplot as plt



def word_for_id(integer, tokenizer):
    """
    将一个整数转换为英文单词
    :param integer: 一个代表英文的整数
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :return: 输入整数对应的英文单词
    """
    for word, index in tokenizer.word_index.items(): # items()函数返回可遍历的（键，值）元组数据
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, photo_feature, max_length=40):
    """
    根据输入的图像特征产生图像的标题
    :param model: 预先训练好的图像标题生成神经网络模型
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :param photo_feature:输入的图像特征, 为VGG16网络修改版产生的特征
    :param max_length: 训练数据中最长的图像标题的长度
    :return: 产生的图像的标题
    """

    in_text = 'startseq' # 保存字符串结果
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])
        sequence = pad_sequences(sequence, max_length)[0] # 输入的数字序列
        output = model.predict([photo_feature, sequence.reshape(1, max_length)])
        integer = np.argmax(output)
        word = word_for_id(integer, tokenizer)
        if word is None:
            break
        in_text = in_text + ' ' + word
        if word == 'endseq':
            break
    return in_text

def generate_caption_run():
    # 加载测试集
    filename = './Flickr_8k.testImages.txt'
    test = util.load_ids(filename)

    # 图像特征
    test_features = util.load_photo_features('./features.pkl', test)
    print('Photos: test=%d' % len(test_features))

    # 加载模型
    model = load_model('./model_8.h5')

    tokenizer = load(open('./tokenizer.pkl', 'rb'))

    pic = pil_image.open('../Flicker8k_Dataset/280706862_14c30d734a.jpg', 'r')
    plt.imshow(pic)
    plt.show()
    caption = generate_caption(model, tokenizer, test_features['280706862_14c30d734a'], 40)
    caption = caption.split(' ')[1:-1]
    caption = ' '.join(caption)
    return caption




def evaluate_model(model, captions, photo_features, tokenizer, max_length = 40):
    """计算训练好的神经网络产生的标题的质量,根据4个BLEU分数来评估

    Args:
        model:　训练好的产生标题的神经网络
        captions: dict, 测试数据集, key为文件名(不带.jpg后缀), value为图像标题list
        photo_features: dict, key为文件名(不带.jpg后缀), value为图像特征
        tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
        max_length：训练集中的标题的最大长度

    Returns:
        tuple:
            第一个元素为权重为(1.0, 0, 0, 0)的ＢＬＥＵ分数
            第二个元素为权重为(0.5, 0.5, 0, 0)的ＢＬＥＵ分数
            第三个元素为权重为(0.3, 0.3, 0.3, 0)的ＢＬＥＵ分数
            第四个元素为权重为(0.25, 0.25, 0.25, 0.25)的ＢＬＥＵ分数

    """
    actual, predicted = list(), list()

    num = 0
    for key, caption_list in captions.items():
        # 生成描述
        yhat = generate_caption(model, tokenizer, photo_features[key], max_length)

        # 存储真实值和预测值
        references = [d.split() for d in caption_list]
        actual.append(references)
        predicted.append(yhat.split())

        num += 1
        print('第%d张测试图像标题生成！' % num)

    # 计算BLEU得分
    bleu1 = corpus_bleu(actual, predicted, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    print('BLEU_1: %f' % bleu1)
    print('BLEU_2: %f' % bleu2)
    print('BLEU_3: %f' % bleu3)
    print('BLEU_4: %f' % bleu4)

    return bleu1, bleu2, bleu3, bleu4


# evaluate_model(model, captions, photo_features, tokenizer, max_length = 40):
def evaluate_model_run():

    # 加载模型
    model = load_model('./model_8.h5')

    # 加载测试数据
    test_fig_name = util.load_ids('./Flickr_8k.testImages.txt')
    captions = util.load_clean_captions('./descriptions.txt', test_fig_name)

    # 加载图像特征
    photo_features = util.load_photo_features('./features.pkl', test_fig_name)


    # 加载tokenizer
    tokenizer = load(open('./tokenizer.pkl', 'rb'))

    bleu1, bleu2, bleu3, bleu4 = evaluate_model(model, captions, photo_features, tokenizer, max_length=40)


if __name__ == '__main__':
    # result = generate_caption_run()
    # print('result is：' + result)

    evaluate_model_run()