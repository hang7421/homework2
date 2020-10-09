from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

# from keras.models import Sequential
# from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


def generate_vgg16():
    """
    搭建VGG16网络结构
    :return: VGG16网络
    """
    input_shape = (224, 224, 3)
    # Sequential（）产生序列结构的网络，里面传入一个表示层的列表
    model = Sequential([
        Conv2D(64, (3,3), input_shape=input_shape, padding='same', activation='relu'),  # 网络的第一层需要设定输入的shape，之后的每一层输入的shape是前一层输出的shape
        Conv2D(64, (3,3), padding='same', activation='relu'),  # Conv2D需要指定filters,kernel_size,padding以及activation
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),  # 卷积层和全连接层之间相连之前，需要将卷积层的输出拉平
        Dense(units=4096, activation='relu'),
        Dense(units=4096, activation='relu'),
        Dense(units=1000, activation='softmax') # 分类任务最后使用softmax作为激活函数


    ])

    return model

if __name__ == '__main__':
    model = generate_vgg16()
    model.summary() # 输出网络的各层信息（各层输出的shape，各层的参数个数）