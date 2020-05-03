import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


class Normalize(keras.layers.Layer):
    # L2 归一化
    def __init__(self, scale, **kwargs):
        '''
        :param scale: 一个常数，缩放比例，最终经过l2_normalize后会乘上
        :param kwargs: ...
        '''
        self.scale = scale
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = input_shape[-1]
        self.gamma = self.add_weight('{}_gamma'.format(self.name), 1, tf.float32,
                                     keras.initializers.ones, trainable=True)

    def call(self, x):
        output = keras.backend.l2_normalize(x, -1)
        output *= self.scale * self.gamma
        return output


class PrioirBox(keras.layers.Layer):
    # 为每一个特征分配初始模板框
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):
        '''
        :param img_size: 图片的尺寸(不是特征图) (w, h)
        :param min_size: box的最小尺寸(单位像素)
        :param max_size: box的最大尺寸(单位像素)
        :param aspect_ration: 一个有关box宽高比的列表
        :param flip: 是否交换宽高比
        :param variances: [x, y, w, h] 的 var 列表，这个与计算最终box的尺寸有关
        :param clip: 是限制box的坐标于[0, 1]间
        :param kwargs:
        '''
        self.waxis = 2
        self.haxis = 1
        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min size must be positive')
        self.min_size = min_size
        self.max_size = max_size

        self.aspect_ratios = [1.0]  # 宽高比
        if max_size:
            if max_size < min_size:
                raise Exception('max size must be greater than min size')
            self.aspect_ratios.append(1.0)  # 这里之后会处理
        if aspect_ratios:
            for ar in aspect_ratios:  # 将宽高比加到self中的里面
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:  # 翻转一下宽高比
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = clip
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        num_prioris_ = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        layer_heigh = input_shape[self.haxis]
        num_boxes = num_prioris_ * layer_heigh * layer_width
        return (input_shape[0], num_boxes, 8)
        # 8是由4 + 4 (box参数与variances参数构成)

    def call(self, inputs, **kwargs):
        '''
        :param inputs: inputs主要是取shape，返回和inputs没有关系
        :param kwargs:
        :return: (samples, num_boxes, 8) 的一个tensor
        '''
        input_shape = keras.backend.int_shape(inputs)
        layer_width = input_shape[self.waxis]
        layer_heigh = input_shape[self.haxis]

        img_width = self.img_size[0]
        img_heigh = self.img_size[1]

        # 定义模板框的形状
        box_width = []
        box_heigh = []

        for ar in self.aspect_ratios:
            if ar == 1 and len(box_width) == 0:
                # aspect_ratios 的第一个元素
                box_width.append(self.min_size)
                box_heigh.append(self.min_size)
                # 以 min_size 作为模板框的尺寸
            elif ar == 1 and len(box_width) > 0:
                # aspect_ratios 的第二个元素，此时一定定义了max_size
                box_width.append(np.sqrt(self.min_size * self.max_size))
                box_heigh.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                # 其余的情况根据公式计算
                box_width.append(self.min_size * np.sqrt(ar))
                box_heigh.append(self.min_size / np.sqrt(ar))

        # 转为半宽、半高
        box_width = 0.5 * np.array(box_width)
        box_heigh = 0.5 * np.array(box_heigh)

        # 接下来定义box的中心坐标
        step_x = img_width / layer_width  # 两个相邻box的宽间距
        step_y = img_heigh / layer_heigh

        # box 中心的横纵坐标
        linx = np.linspace(step_x * 0.5, img_width - step_x * 0.5, layer_width)
        liny = np.linspace(step_y * 0.5, img_heigh - step_y * 0.5, layer_heigh)

        # 以此来构建网格
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # 然后定义模板框的 xmin, ymin, xmax, ymax
        num_priors_ = len(self.aspect_ratios)
        # 将x、y坐标进行拼接，成为一个数组
        prior_boxes = np.concatenate([centers_x, centers_y], axis=1)
        # 此时每个 prior_boxes 仅仅存在两个坐标，而box应当有4个，故重复了1次
        prior_boxes = np.tile(prior_boxes, [1, 2 * num_priors_])

        # [feat_num, 4 * num_priors_]
        prior_boxes[:, ::4] -= box_width  # x_min
        prior_boxes[:, 1::4] -= box_heigh
        prior_boxes[:, 2::4] += box_width
        prior_boxes[:, 3::4] += box_heigh
        prior_boxes[:, ::2] /= img_width  # 归一化，使其与img_size产生依赖
        prior_boxes[:, 1::2] /= img_heigh
        prior_boxes = prior_boxes.reshape(-1, 4)

        # [feat_num * num_priors_, 4]
        if self.clip: # 限制取值
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        # 定义 variances
        num_boxes = len(prior_boxes)  # 取出 prior_boxes 第零维的元素个数

        if len(self.variances) == 1:
            variances = np.ones([num_boxes, 4]) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, [num_boxes, 1])
        else:
            raise Exception("Must provide one or four variances.")

        prior_boxes = np.concatenate([prior_boxes, variances], axis=1)
        # 将boxes 与 variances 进行拼接
        prior_boxes_tensor = tf.expand_dims(prior_boxes, axis=0)
        # 转为tensor
        pattern = [tf.shape(inputs)[0], 1, 1]  # 扩展到batch上
        # [batch_size, num_boxes, 8]
        prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)
        prior_boxes_tensor = tf.cast(prior_boxes_tensor, tf.float32)
        return prior_boxes_tensor


