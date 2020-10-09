from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
from pickle import load, dump
import numpy as np
import util # 注意task3和task4中的util的函数名称不一致，task3中按照视频上修改了名称

def create_tokenizer():
    """
    根据训练数据集中图像名，和其对应的标题，生成一个tokenizer
    Returns:生成的tokenizer

    """

    train_image_names = util.load_image_names('./Flickr_8k.trainImages.txt')
    train_descriptions = util.load_clean_captions('./descriptions.txt', train_image_names)  # train_decriptions的key为每一个训练图像名称（去除.jpg后缀），值为一个list，里面包含每一张图片的5个描述
    lines = util.to_list(train_descriptions)  # lines为一维的list

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    dump(tokenizer, open('./tokenizer.pkl', 'wb'))

    return tokenizer

def create_input_data(tokenizer, max_length, descriptions, photos_features, vocab_size):
    """
    从输入的图片标题list和图片特征构造LSTM的一组输入

    Args:
    :param tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
    :param max_length: 训练数据集中最长的标题的长度
    :param descriptions: dict, key 为图像的名(不带.jpg后缀), value 为list, 包含一个图像的几个不同的描述
    :param photos_features:  dict, key 为图像的名(不带.jpg后缀), value 为numpy array 图像的特征
    :param vocab_size: 训练集中表的单词数量
    :return: tuple:
            第一个元素为 numpy array, 元素为图像的特征, 它本身也是 numpy.array
            第二个元素为 numpy array, 元素为图像标题的前缀, 它自身也是 numpy.array
            第三个元素为 numpy array, 元素为图像标题的下一个单词(根据图像特征和标题的前缀产生) 也为numpy.array

    Examples:
        from pickle import load
        tokenizer = load(open('tokenizer.pkl', 'rb'))
        max_length = 6
        descriptions = {'1235345':['startseq one bird on tree endseq', "startseq red bird on tree endseq"],
                        '1234546':['startseq one boy play water endseq', "startseq one boy run across water endseq"]}
        photo_features = {'1235345':[ 0.434,  0.534,  0.212,  0.98 ],
                          '1234546':[ 0.534,  0.634,  0.712,  0.28 ]}
        vocab_size = 7378
        print(create_input_data(tokenizer, max_length, descriptions, photo_features, vocab_size))
(array([[ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ]]),
array([[  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  59],
       [  0,   0,   0,   2,  59, 254],
       [  0,   0,   2,  59, 254,   6],
       [  0,   2,  59, 254,   6, 134],
       [  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  26],
       [  0,   0,   0,   2,  26, 254],
       [  0,   0,   2,  26, 254,   6],
       [  0,   2,  26, 254,   6, 134],
       [  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  59],
       [  0,   0,   0,   2,  59,  16],
       [  0,   0,   2,  59,  16,  82],
       [  0,   2,  59,  16,  82,  24],
       [  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  59],
       [  0,   0,   0,   2,  59,  16],
       [  0,   0,   2,  59,  16, 165],
       [  0,   2,  59,  16, 165, 127],
       [  2,  59,  16, 165, 127,  24]]),
array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ...,
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]]))
    """
    input1 = list()
    input2 = list()
    output = list()


    for key in descriptions.keys():
        seq_list = tokenizer.texts_to_sequences(descriptions[key]) # 将字符串列表利用tokenizer转换为为数字序列
        for seq in seq_list:
            for i in range(1, len(seq)):
                inseq = seq[:i]
                outseq = seq[i]

                input1.append(photos_features[key])

                # 填充inseq，使其长度为max_length
                inseq = pad_sequences([inseq], maxlen=max_length)[0]
                input2.append(inseq)

                # 将outseq转化为one-hot编码
                outseq = to_categorical([outseq], num_classes=vocab_size)[0]
                output.append(outseq)
    input1 = np.array(input1)
    input2 = np.array(input2)
    output = np.array(output)

    return input1, input2, output

# 测试create_tokenizer()
tolenizer = create_tokenizer()
print(tolenizer.word_index)

# 测试create_input_data
# from pickle import load
# tokenizer = load(open('tokenizer.pkl', 'rb'))
# max_length = 6
# descriptions = {'1235345':['startseq one bird on tree endseq', "startseq red bird on tree endseq"],
#                 '1234546':['startseq one boy play water endseq', "startseq one boy run across water endseq"]}
# photo_features = {'1235345':[ 0.434,  0.534,  0.212,  0.98 ],
#                   '1234546':[ 0.534,  0.634,  0.712,  0.28 ]}
# vocab_size = 7378
# print(create_input_data(tokenizer, max_length, descriptions, photo_features, vocab_size))