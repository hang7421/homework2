from keras.models import model_from_json  # 从json文件生成网络模型
from PIL import Image as pil_image  # 打开图片文件
from keras import backend as K  # keras的后端
import numpy as np
from pickle import dump
from os import listdir
from keras.models import Model
import keras


def load_vgg16_model():
    """从当前目录下面的 vgg16_exported.json 和 vgg16_exported.h5 两个文件中导入 VGG16 网络并返回创建的网络模型
    # Returns
        创建的网络模型 model
    """
    json_file = open('./vgg16_exported.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights('./vgg16_exported.h5')

    return model



def preprocess_input(x):
    """预处理图像用于网络输入, 将图像由RGB格式转为BGR格式.
       将图像的每一个图像通道减去其均值

    # Arguments
        x: numpy 数组, 4维.
        data_format: Data format of the image array.

    # Returns
        Preprocessed Numpy array.
    """

    # PIL读取的是RGB,opencv读取的是BGR
    # x[..., ::-1] <==> x[:,:,:, ::-1]
    x = x[..., ::-1]   # 'RGB'->'BGR', https://www.scivision.co/numpy-image-bgr-to-rgb/

    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]  # x的第四个维度是BGR三层，索引为0，1，2
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x



def load_img_as_np_array(path, target_size):
    """从给定文件加载图像,转换图像大小为给定target_size,返回32位浮点数numpy数组.

    # Arguments
        path: 图像文件路径
        target_size: 元组(图像高度, 图像宽度).

    # Returns
        A PIL Image instance.
    """
    img = pil_image.open(path)
    img = img.resize(target_size, pil_image.NEAREST)   # 调整图片的尺寸大小
    return np.asarray(img, dtype=K.floatx())  # K.floatx()方法返回keras后端支持的浮点型类型

def extract_features(directory):
    """提取给定文件夹中所有图像的特征, 将提取的特征保存在文件features.pkl中,
       提取的特征保存在一个dict中, key为文件名(不带.jpg后缀), value为特征值[np.array]

    Args:
        directory: 包含jpg文件的文件夹

    Returns:
        None
    """
    model = load_vgg16_model()
    # 去掉模型的最后一层
    model.layers.pop()

    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    features = dict()

    num = 0
    for fn in listdir(directory):
        id = fn.split('.')[0]  # 去掉文件后缀
        fn = directory + '/' + fn
        arr = load_img_as_np_array(fn, target_size=(224, 224))
        # print(arr.shape)

        # 改变数组的形状，增加一个维度（批处理输入的维度）

        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        # print(arr.shape)

        # 预处理图像作为VGG模型的输入
        arr = preprocess_input(arr)

        # 计算特征
        feature = model.predict(arr, verbose=0)  # 二维numpy数组

        features[id] = feature

        num += 1

        if num % 10 == 0:
            print('第%d张图提取完毕！' % num)

    return features



if __name__ == '__main__':
    # 提取所有图像的特征，保存在一个文件中, 大约一小时的时间，最后的文件大小为127M
    directory = './Flicker8k_Dataset'
    features = extract_features(directory)
    print(features)
    print('提取特征的文件个数：%d' % len(features))
    print(keras.backend.image_data_format())
    #保存特征到文件
    dump(features, open('./features.pkl', 'wb'))



