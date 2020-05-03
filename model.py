import tensorflow as tf
import tensorflow.keras as keras
import layers


def get_SSD(input_shape, num_class=21):
    '''
    :param input_shape: 训练输入尺寸，(300, 300, 3)
    :param num_class: box 的类别数
    :return: [batch, box_num, 4 + num_class + 8]
    '''

    layer = {}
    img_size = (input_shape[1], input_shape[0])
    input_tensor = keras.Input(shape=input_shape)
    layer['input'] = input_tensor
    # 由于PriorBox层需要用到每个特征图的长宽，因此输入的大小必须事先给定

    layer['conv1_1'] = keras.layers.Conv2D(64, 3, 1, 'same', activation='relu', name='conv1_1')(layer['input'])
    layer['conv1_2'] = keras.layers.Conv2D(64, 3, 1, 'same', activation='relu', name='conv1_2')(layer['conv1_1'])
    layer['pool1'] = keras.layers.MaxPooling2D(strides=2, padding='same', name='pool1')(layer['conv1_2'])

    # Block 2
    layer['conv2_1'] = keras.layers.Conv2D(128, 3, 1, 'same', activation='relu', name='conv2_1')(layer['pool1'])
    layer['conv2_2'] = keras.layers.Conv2D(128, 3, 1, 'same', activation='relu', name='conv2_2')(layer['conv2_1'])
    layer['pool2'] = keras.layers.MaxPooling2D(strides=2, padding='same', name='pool2')(layer['conv2_2'])

    # Block 3
    layer['conv3_1'] = keras.layers.Conv2D(256, 3, 1, 'same', activation='relu', name='conv3_1')(layer['pool2'])
    layer['conv3_2'] = keras.layers.Conv2D(256, 3, 1, 'same', activation='relu', name='conv3_2')(layer['conv3_1'])
    layer['conv3_3'] = keras.layers.Conv2D(256, 3, 1, 'same', activation='relu', name='conv3_3')(layer['conv3_2'])
    layer['pool3'] = keras.layers.MaxPooling2D(strides=2, padding='same', name='pool3')(layer['conv3_3'])

    # Block 4
    layer['conv4_1'] = keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv4_1')(layer['pool3'])
    layer['conv4_2'] = keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv4_2')(layer['conv4_1'])
    layer['conv4_3'] = keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv4_3')(layer['conv4_2'])
    layer['pool4'] = keras.layers.MaxPooling2D(strides=2, padding='same', name='pool4')(layer['conv4_3'])

    # Block 5
    layer['conv5_1'] = keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv5_1')(layer['pool4'])
    layer['conv5_2'] = keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv5_2')(layer['conv5_1'])
    layer['conv5_3'] = keras.layers.Conv2D(512, 3, 1, 'same', activation='relu', name='conv5_3')(layer['conv5_2'])
    layer['pool5'] = keras.layers.MaxPooling2D((3, 3), strides=1, padding='same', name='pool5')(layer['conv5_3'])

    # 读取之后会用到的层的输出
    layer['b4_conv3'] = layer['conv4_3']
    layer['b5_pool'] = layer['pool5']
    # Middle 这里是代替原来VGG后面的全连接层
    layer['m_conv1'] = keras.layers.Conv2D(1024, 3, 1, 'same', activation='relu',
                                           dilation_rate=(6, 6), name='m_conv1')(layer['b5_pool'])
    layer['m_conv2'] = keras.layers.Conv2D(1024, 1, 1, 'same', name='m_conv2')(layer['m_conv1'])

    # Block 6
    layer['b6_conv1'] = keras.layers.Conv2D(256, 1, 1, 'same', activation='relu', name='b6_conv1')(layer['m_conv2'])
    layer['b6_conv2'] = keras.layers.Conv2D(512, 3, 2, 'same', activation='relu', name='b6_conv2')(layer['b6_conv1'])

    # Block 7
    layer['b7_conv1'] = keras.layers.Conv2D(128, 1, 1, 'same', activation='relu', name='b7_conv1')(layer['b6_conv2'])
    layer['b7_padding'] = keras.layers.ZeroPadding2D(name='b7_padding')(layer['b7_conv1'])
    layer['b7_conv2'] = keras.layers.Conv2D(256, 3, 2, 'same', activation='relu', name='b7_conv2')(layer['b7_padding'])

    # Block 8
    layer['b8_conv1'] = keras.layers.Conv2D(128, 1, 1, 'same', activation='relu', name='b8_conv1')(layer['b7_conv2'])
    layer['b8_conv2'] = keras.layers.Conv2D(256, 3, 2, 'same', activation='relu', name='b8_conv2')(layer['b8_conv1'])

    # Last pooling
    layer['b8_pool'] = keras.layers.GlobalAveragePooling2D(name='b8_pool')(layer['b8_conv2'])

    # 预测部分
    # b4_conv3 的部分
    layer['b4_conv3_norm'] = layers.Normalize(20, name='b4_conv3_norm')(layer['b4_conv3'])
    num_priors = 3  # 每个特征对应 3 个模板框
    layer['b4_conv3_norm_loc'] = keras.layers.Conv2D(num_priors * 4, 3, 1, 'same', name='b4_conv3_norm_loc')(
        layer['b4_conv3_norm'])
    # 预测位置，深度为3个模板框，每个框4个参数
    layer['b4_conv3_norm_loc_flat'] = keras.layers.Flatten(name='b4_conv3_norm_loc_flat')(layer['b4_conv3_norm_loc'])
    # 展平是为了之后的拼接
    name = 'b4_conv3_norm_conf'
    if (num_class != 21):
        name += '_{}'.format(num_class)
    layer['b4_conv3_norm_conf'] = keras.layers.Conv2D(num_priors * num_class, 3, 1, 'same', name=name)(
        layer['b4_conv3_norm'])
    # 类别的置信度预测
    layer['b4_conv3_norm_conf_flat'] = keras.layers.Flatten(name='b4_conv3_norm_conf_flat')(layer['b4_conv3_norm_conf'])
    priorbox = layers.PrioirBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2],
                                name='b4_conv3_norm_priorbox')
    layer['b4_conv3_norm_priorbox'] = priorbox(layer['b4_conv3_norm'])
    print('b4_conv3_norm_loc shape', layer['b4_conv3_norm_loc'].shape)

    # m_conv2 的部分
    num_priors = 6
    layer['m_conv2_loc'] = keras.layers.Conv2D(num_priors * 4, 3, 1, 'same', name='m_conv2_loc')(
        layer['m_conv2'])
    layer['m_conv2_loc_flat'] = keras.layers.Flatten(name='m_conv2_loc_flat')(layer['m_conv2_loc'])
    name = 'm_conv2_conf'
    if (num_class != 21):
        name += '_{}'.format(num_class)
    layer['m_conv2_conf'] = keras.layers.Conv2D(num_priors * num_class, 3, 1, 'same', name=name)(
        layer['m_conv2'])
    layer['m_conv2_conf_flat'] = keras.layers.Flatten(name='m_conv2_conf_flat')(layer['m_conv2_conf'])
    priorbox = layers.PrioirBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2],
                                name='m_conv2_priorbox')
    layer['m_conv2_priorbox'] = priorbox(layer['m_conv2'])
    print('m_conv2_loc shape', layer['m_conv2_loc'].shape)

    # b6_conv2 的部分
    num_priors = 6
    layer['b6_conv2_loc'] = keras.layers.Conv2D(num_priors * 4, 3, 1, 'same', name='b6_conv2_loc')(
        layer['b6_conv2'])
    layer['b6_conv2_loc_flat'] = keras.layers.Flatten(name='b6_conv2_loc_flat')(layer['b6_conv2_loc'])
    name = 'b6_conv2_conf'
    if (num_class != 21):
        name += '_{}'.format(num_class)
    layer['b6_conv2_conf'] = keras.layers.Conv2D(num_priors * num_class, 3, 1, 'same', name=name)(
        layer['b6_conv2'])
    layer['b6_conv2_conf_flat'] = keras.layers.Flatten(name='b6_conv2_conf_flat')(layer['b6_conv2_conf'])
    priorbox = layers.PrioirBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2],
                                name='b6_conv2_priorbox')
    layer['b6_conv2_priorbox'] = priorbox(layer['b6_conv2'])
    print('b6_conv2_loc shape', layer['b6_conv2_loc'].shape)

    # b7_conv2 的部分
    num_priors = 6
    layer['b7_conv2_loc'] = keras.layers.Conv2D(num_priors * 4, 3, 1, 'same', name='b7_conv2_loc')(
        layer['b7_conv2'])
    layer['b7_conv2_loc_flat'] = keras.layers.Flatten(name='b7_conv2_loc_flat')(layer['b7_conv2_loc'])
    name = 'b7_conv2_conf'
    if (num_class != 21):
        name += '_{}'.format(num_class)
    layer['b7_conv2_conf'] = keras.layers.Conv2D(num_priors * num_class, 3, 1, 'same', name=name)(
        layer['b7_conv2'])
    layer['b7_conv2_conf_flat'] = keras.layers.Flatten(name='b7_conv2_conf_flat')(layer['b7_conv2_conf'])
    priorbox = layers.PrioirBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2],
                                name='b7_conv2_priorbox')
    layer['b7_conv2_priorbox'] = priorbox(layer['b7_conv2'])
    print('b7_conv2_loc shape', layer['b7_conv2_loc'].shape)

    # b8_conv2 的部分
    num_priors = 6
    layer['b8_conv2_loc'] = keras.layers.Conv2D(num_priors * 4, 3, 1, 'same', name='b8_conv2_loc')(
        layer['b8_conv2'])
    layer['b8_conv2_loc_flat'] = keras.layers.Flatten(name='b8_conv2_loc_flat')(layer['b8_conv2_loc'])
    name = 'b8_conv2_conf'
    if (num_class != 21):
        name += '_{}'.format(num_class)
    layer['b8_conv2_conf'] = keras.layers.Conv2D(num_priors * num_class, 3, 1, 'same', name=name)(
        layer['b8_conv2'])
    layer['b8_conv2_conf_flat'] = keras.layers.Flatten(name='b8_conv2_conf_flat')(layer['b8_conv2_conf'])
    priorbox = layers.PrioirBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2],
                                name='b8_conv2_priorbox')
    layer['b8_conv2_priorbox'] = priorbox(layer['b8_conv2'])
    print('b8_conv2_loc shape', layer['b8_conv2_loc'].shape)

    # b8_pool  的部分
    num_priors = 6
    layer['b8_pool_loc_flat'] = keras.layers.Dense(num_priors * 4, name='b8_pool_loc_flat')(
        layer['b8_pool'])
    name = 'b8_pool_conf_flat'
    if (num_class != 21):
        name += '_{}'.format(num_class)
    layer['b8_pool_conf_flat'] = keras.layers.Dense(num_priors * num_class, name=name)(
        layer['b8_pool'])
    priorbox = layers.PrioirBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2],
                                name='b8_pool_priorbox')
    layer['b8_pool_reshaped'] = keras.layers.Reshape((1, 1, 256), name='pool6_reshaped')(layer['b8_pool'])
    layer['b8_pool_priorbox'] = priorbox(layer['b8_pool_reshaped'])
    print('b8_pool_priorbox shape', layer['b8_pool_priorbox'].shape)

    # 把前面的输出整合起来
    layer['mbox_loc'] = tf.concat([layer['b4_conv3_norm_loc_flat'],
                                   layer['m_conv2_loc_flat'],
                                   layer['b6_conv2_loc_flat'],
                                   layer['b7_conv2_loc_flat'],
                                   layer['b8_conv2_loc_flat'],
                                   layer['b8_pool_loc_flat']],
                                  axis=1, name='mbox_loc')
    layer['mbox_conf'] = tf.concat([layer['b4_conv3_norm_conf_flat'],
                                    layer['m_conv2_conf_flat'],
                                    layer['b6_conv2_conf_flat'],
                                    layer['b7_conv2_conf_flat'],
                                    layer['b8_conv2_conf_flat'],
                                    layer['b8_pool_conf_flat']],
                                   axis=1, name='mbox_conf')
    layer['mbox_priorbox'] = tf.concat([layer['b4_conv3_norm_priorbox'],
                                        layer['m_conv2_priorbox'],
                                        layer['b6_conv2_priorbox'],
                                        layer['b7_conv2_priorbox'],
                                        layer['b8_conv2_priorbox'],
                                        layer['b8_pool_priorbox']],
                                       axis=1, name='mbox_priorbox')

    num_boxes = keras.backend.int_shape(layer['mbox_loc'])[-1] // 4
    #  拼接完后最终 Reshape 回去
    layer['mbox_loc'] = keras.layers.Reshape((num_boxes, 4), name='mbox_loc_final')(layer['mbox_loc'])
    layer['mbox_conf'] = keras.layers.Reshape((num_boxes, num_class), name='mbox_conf_logits')(layer['mbox_conf'])
    # Softmax 激活函数
    layer['mbox_conf'] = keras.layers.Softmax(name='mbox_conf_final')(layer['mbox_conf'])

    # print(layer['mbox_loc'].shape, layer['mbox_conf'].shape, layer['mbox_priorbox'].shape)
    layer['predictions'] = tf.concat([layer['mbox_loc'],
                                      layer['mbox_conf'],
                                      layer['mbox_priorbox']],
                                     axis=2, name='predictions')
    print(layer['predictions'].shape)
    model = keras.models.Model(inputs=input_tensor, outputs=layer['predictions'])
    # keras.utils.plot_model(model,'model.png')
    return model

# get_SSD((300, 300, 3))

class MultiboxLoss():
    def __init__(self, num_class, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        '''
        :param num_class: 类别数
        :param alpha: L1-smooth损失的系数
        :param neg_pos_ratio:最大负正样本比
        :param background_label_id: 背景所在的位置
        :param negatives_for_hard: 如果没有负样本时负样本的个数
                (这里挺迷的，源代码写的是没有正样本时负样本的个数，
                但是代码里又是在没有负样本时添加，我觉得应该是没有负样本，
                这样可以均衡正负样本数)
        '''
        self.num_class = num_class
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported!')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard

    def smooth_l1_loss(self, y_true, y_pred):
        '''
        计算平滑L1loss
        :param y_true: Ground truth bounding boxes,
                tensor of shape (?, num_boxes, 4).
        :param y_pred: Predicted bounding boxes,
                tensor of shape (?, num_boxes, 4).
        :return: L1-smooth loss, tensor of shape (?, num_boxes).
        '''
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss)
        return tf.reduce_sum(l1_loss, axis=-1)

    def compute_loss(self, y_true, y_pred):
        '''
        :param y_true:
            Ground truth targets,
            tensor of shape (?, num_boxes, 4 + num_classes + 8),
            priors in ground truth are fictitious,
            y_true[:, :, -8] has 1 if prior should be penalized
                or in other words is assigned to some ground truth box,
            y_true[:, :, -7:] are all 0.
        :param y_pred: Predicted logists,
            tensor of shape (?, num_boxes, 4 + num_classes + 8).
        :return: Loss for prediction, tensor of shape (?,).
        '''
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.cast(tf.shape(y_true)[1], tf.float32)

        # 所有模板框的loss, 之后会对正负样本进行单独的处理
        conf_loss = keras.losses.categorical_crossentropy(y_true[:, :, 4:-8],
                                                          y_pred[:, :, 4:-8])
        loc_loss = self.smooth_l1_loss(y_true[:, :, :4],
                                       y_pred[:, :, :4])
        # loc_loss = [batch, num_priors]

        # 正样本的loss
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)  # 每个batch中正样本的个数

        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8], axis=1)
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8], axis=1)

        # 负样本的loss, 对于负样本只计算置信度的loss，忽略location
        # 先计算出每个批次中的负样本的个数
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos,
                             num_boxes - num_pos)

        pos_num_neg_mask = tf.greater(num_neg, 0)  # 每个batch中是否存在负样本
        has_min = tf.cast(tf.reduce_any(pos_num_neg_mask), dtype=tf.float32)  # 所有批次中是否存在负样本
        num_neg = tf.concat(axis=0, values=[num_neg,
                                            [(1 - has_min) * self.negatives_for_hard]])
        # 如果没有负样本则添加一个批次，其中包含negatives_for_hard这么多的负样本，其实还是为了均衡正负样本数

        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg,
                                                      tf.greater(num_neg, 0)))
        # 对于存在负样本的batch，求出其中最小的负样本个数，作为每个批次的夫样本个数

        num_neg_batch = tf.cast(num_neg_batch, tf.int32)

        # 除去background的预测值, 这里要依据其进行排序选出top_k
        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_class - 1
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end], axis=2)

        # 返回了负样本中置信度非背景最高的前num_neg_batch个priorbox的相对下标
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]), k=num_neg_batch)

        # 这里要将indices转为绝对坐标，即到数组头部的距离
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, [1, num_neg_batch])  # [batch_size, num_neg_batch]

        # 绝对坐标
        full_indeces = (tf.reshape(batch_idx, (-1,)) * tf.cast(num_boxes, dtype=tf.int32) +
                        tf.reshape(indices, (-1,)))
        # tf.reshape(batch_idx, (-1, ) * tf.cast(num_boxes)) 是求出每个batch的起始位置

        neg_conf_loss = tf.gather(tf.reshape(conf_loss, (-1,)),
                                  full_indeces)
        # tf.gather 作用是从原数组中挑选出所有给定下标的元素，
        # 这里就是挑选出top_k负样本中对应的loss部分
        neg_conf_loss = tf.reshape(neg_conf_loss, (batch_size, num_neg_batch))
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=-1)

        total_loss = pos_conf_loss + neg_conf_loss
        total_loss /= (num_pos + tf.cast(num_neg_batch, dtype=tf.float32))
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
        # 防止除 0
        total_loss += (self.alpha * pos_loc_loss) / num_pos
        return total_loss
