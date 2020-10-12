from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from pickle import load
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import util # 注意task3和task4中的util的函数名称不一致，task3中按照视频上修改了名称，task4中保持不变
from numpy import array, concatenate


def create_batches(desc_list, photo_features, tokenizer, max_len, vocab_size=7378):
    """从输入的图片标题list和图片特征构造LSTM的一组输入

    Args:
        desc_list: 某一个图像对应的一组标题(一个list)
        photo_features: 某一个图像对应的特征
        tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
        max_len: 训练数据集中最长的标题的长度
        vocab_size: 训练集中的单词个数, 默认为7378

    Returns:
        tuple:
            第一个元素为list, list的元素为图像的特征
            第二个元素为list, list的元素为图像标题的前缀
            第三个元素为list, list的元素为图像标题的下一个单词(根据图像特征和标题的前缀产生)

    Examples:
        #>>> from pickle import load
        #>>> tokenizer = load(open('tokenizer.pkl', 'rb'))
        #>>> desc_list = ['startseq one dog on desk endseq', "startseq red bird on tree endseq"]
        #>>> photo_features = [0.434, 0.534, 0.212, 0.98]
        #>>> print(create_batches(desc_list, photo_features, tokenizer, 6, 7378))
            (array([[ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ]]),
            array([[   0,    0,    0,    0,    0,    2],
                   [   0,    0,    0,    0,    2,   59],
                   [   0,    0,    0,    2,   59,    9],
                   [   0,    0,    2,   59,    9,    6],
                   [   0,    2,   59,    9,    6, 1545],
                   [   0,    0,    0,    0,    0,    2],
                   [   0,    0,    0,    0,    2,   26],
                   [   0,    0,    0,    2,   26,  254],
                   [   0,    0,    2,   26,  254,    6],
                   [   0,    2,   26,  254,    6,  134]]),
            array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   ...,
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.]]))

    """
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo_features)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(captions, photo_features, tokenizer, max_len):
    """创建一个训练数据生成器, 用于传入模型训练函数的第一个参数model.fit_generator(generator,...)

    Args:
        captions: dict, key为图像名(不包含.jpg后缀), value为list, 图像的几个训练标题
        photo_features: dict, key为图像名(不包含.jpg后缀), value为图像的特征
        tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
        max_len: 训练集中的标题最长长度

    Returns:
        generator, 使用yield [[list, 元素为图像特征, list, 元素为输入的图像标题前缀], list, 元素为预期的输出图像标题的下一个单词]

    """
    # loop for ever over images
    while 1:
        flag = 0
        num = 1
        for key, desc_list in captions.items():
            # retrieve the photo feature
            photo_feature = photo_features[key][0] # photo_features[key]为二维numpy数组（由vgg网络执行model.predict()时产生的二维numpy数组）
            if flag == 0:
                in_img, in_seq, out_word = create_batches(desc_list, photo_feature, tokenizer, max_len)
                flag = 1
            else:
                in_img_new, in_seq_new, out_word_new = create_batches(desc_list, photo_feature, tokenizer, max_len)

                in_img = concatenate((in_img, in_img_new), axis=0)

                in_seq = concatenate((in_seq, in_seq_new), axis=0)

                out_word = concatenate((out_word, out_word_new), axis=0)
                num += 1
                if num == 25:
                    flag = 0
                    num = 1
                    yield [[in_img, in_seq], out_word]


def caption_model(vocab_size, max_len):
    """创建一个新的用于给图片生成标题的网络模型

    Args:
        vocab_size: 训练集中标题单词个数
        max_len: 训练集中的标题最长长度

    Returns:
        用于给图像生成标题的网络模型

    """
    input1 = Input(shape=(4096,))

    dropout_1 = Dropout(rate=0.5)(input1)

    dense_1 = Dense(256, activation='relu')(dropout_1)

    input2 = Input(shape=(max_len,))

    embed_1 = Embedding(input_dim=vocab_size, output_dim=256)(input2) # input_dim：词汇表大小， output_dim：词向量的维度
    lstm_1 = LSTM(units=256, activation='relu')(embed_1) # units:输出空间的维度

    add_1 = add([dense_1, lstm_1])
    dense_2 = Dense(units=256, activation='relu')(add_1)

    outputs = Dense(units=vocab_size, activation='softmax')(dense_2)

    model = Model(inputs=[input1, input2], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model




def train():
    # load training dataset (6K)
    filename = './Flickr_8k.trainImages.txt'
    train = util.load_ids(filename) # 加载训练图片的名称，组成一个列表
    print('Dataset: %d' % len(train))
    train_captions = util.load_clean_captions('./descriptions.txt', train) # 为图像标题首尾分别加上'startseq ' 和 ' endseq', 返回图像描述的字典
    print('Captions: train number=%d' % len(train_captions))
    # photo features
    train_features = util.load_photo_features('./features.pkl', train) # 加载提取的图像特征
    print('Photos: train=%d' % len(train_features))
    # prepare tokenizer
    tokenizer = load(open('tokenizer.pkl', 'rb')) # 加载tokenizer
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    max_len = util.get_max_length(train_captions)  # 计算标题的最大长度
    print('Description Length: %d' % max_len)

    # define the model
    model = caption_model(vocab_size, max_len)
    # train the model, run epochs manually and save after each epoch
    epochs = 20
    steps = len(train_captions) / 25
    for i in range(epochs):
        print('第%d个epoch开始：' % (i+1))
        # create the data generator
        generator = data_generator(train_captions, train_features, tokenizer, max_len)
        # fit for one epoch
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1) # 通过generator采用yield一次一次产生数据
        # save model
        model.save('model_' + str(i) + '.h5')

train()