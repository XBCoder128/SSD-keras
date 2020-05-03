import tensorflow as tf
import pickle
import layers
import numpy as np
import matplotlib.pyplot as plt


class BBoxUtility():
    # 一个工具类，用于对边界框和模板框做一些操作

    def __init__(self, num_class, priors=None, overlap_threshold=0.5,
                 nms_threshold=0.45, top_k=400):
        '''
        :param num_class: 类别数
        :param priors: 模板框
        :param overlap_threshold: 将box分配给某个模板框的阈值
        :param nms_threshold: 非极大值抑制的阈值
        :param top_k: nms后留下的box数目
        '''
        self.num_class = num_class
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k

    @property
    def nms_threshold(self):
        return self._nms_threshold

    @nms_threshold.setter
    def nms_threshold(self, value):
        self._nms_threshold = value

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value

    def iou(self, box):
        '''
        计算box与所有模板框的IOU
        :param box: numpy tensor of shape (4,).
        :return: Intersection over union,
                numpy tensor of shape (num_priors).
                返回这个框和所有模板框的交并比
        '''
        # compute intersection 交集
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])  # 相交矩形的左上角
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])  # 相交矩形的右下角
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)  # 相交矩形的宽高
        inter = inter_wh[:, 0] * inter_wh[:, 1]  # 面积

        # compute union 并集
        # 用容斥原理求并集面积
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.priors[:, 2] - self.priors[:, 0])
        area_gt *= (self.priors[:, 3] - self.priors[:, 1])
        union = area_gt + area_pred - inter

        iou = inter / union  # 面积比
        return iou

    def encode_box(self, box, return_iou=True):
        '''
        将box编码为可以用于训练的数据，只对指定(IOU>0?)的先验框编码
        Encode box for training, do it only for assigned priors.
        :param box: numpy tensor of shape (4,).
        :param return_iou: 是否再返回中加入IOU
        :return: numpy tensor of shape (num_priors, 4 + int(return_iou)).
        '''
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():  # 如果没有一个是超过阈值的
            assign_mask[iou.argmax()] = True  # 就把里面IOU最大的设置为真

        if return_iou:
            # 将IOU放在最后一列
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        assigned_priors = self.priors[assign_mask]  # 取出满足条件的模板框

        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:4])
        assigned_priors_wh = assigned_priors[:, 2:4] - assigned_priors[:, :2]  # 注意模板框的尺寸是[..., 8]

        # 根据var将box转换为与模板框相关的系数
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]  # 根据
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]

        return encoded_box.ravel()  # 展平

    def assign_boxes(self, boxes):
        '''
        为训练分配模板框
        :param boxes: numpy tensor of shape (num_boxes, 4 + num_classes)
            num_classes without background. 此处的numclass不包含background

        :return: Tensor with assigned boxes
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                    如果assignment[:, -8]等于1则该prior应该被训练
                assignment[:, -7:] are all 0. See loss for more details.
                assignment[:, -7]全都是0
        '''
        assignment = np.zeros((self.num_priors, 4 + self.num_class + 8))
        assignment[:, 4] = 1.0  # 先全认为是负样本，后面慢慢分配
        if len(boxes) == 0:
            return assignment

        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape([-1, self.num_priors, 5])  # (num_boxes, num_priors, 5)

        # encoded_boxes[:, :, -1]是二维，再argmax变成1维
        # 这个就是求与每个模板框最匹配的box的位置
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)  # 求位置上的值

        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]  # 求出存在交集的

        assign_num = len(best_iou_idx)  # 能够分配的模板框个数
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]  # 没有交集的模板框就不要了

        # 将第encoded_boxes[best_iou_idx[i], np.arange(assign_num)[i], :4]放到xx[bestxx[i]]里面
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                           np.arange(assign_num),
                                           :4]
        assignment[:, 4][best_iou_mask] = 0  # 这部分就不是背景了
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1  # ground truth


        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        '''
        反推回box真正的尺寸
        :param mbox_loc: 预测位置的np数组
        :param mbox_priorbox: prior boxes
        :param variances: variances数组
        :return: 转换后的模板框
        '''
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_heigh = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x

        decode_bbox_center_y = mbox_loc[:, 1] * prior_heigh * variances[:, 1]
        decode_bbox_center_y += prior_center_y

        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width

        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_heigh

        # 由 [cx, xy, w, h] 转为 [xmin, ymin, xmax, ymax]
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 拼接起来
        decode_bbox = np.concatenate([decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]
                                      ], axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=200,
                      confidence_shreshold=0.01):
        '''
            在预测结果上做非极大抑制
        :param predictions: 预测值
        :param background_label_id: 背景的id
        :param keep_top_k: 做完非极大抑制后，剩余的box数目
        :param confidence_shreshold: 置信度阈值
        :return: List of predictions for every picture.
                具有[label, confidence, xmin, ymin, xmax, ymax]的格式的列表
        '''
        mbox_loc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mbox_prioribox = predictions[:, :, -8:-4]
        mbox_conf = predictions[:, :, 4:-8]
        results = []

        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i],
                                             mbox_prioribox[i],
                                             variances[i])
            for c in range(self.num_class):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_shreshold

                # 该类别中大于置信度阈值的box个数
                if len(c_confs[c_confs_m] > 0):
                    # 取出box和score，准备做nms
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    idx = tf.image.non_max_suppression(boxes_to_process,
                                                       confs_to_process,
                                                       self._top_k,
                                                       self._nms_threshold)
                    good_boxes = tf.gather(boxes_to_process, idx, axis=0)
                    confs = tf.gather(confs_to_process, idx, axis=0)[:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes), axis=1) # -1 也行
                    results[-1].extend(c_pred)  # 加入结果

            if len(results[-1]) > 0: # 因为每次添加都会在results的最后一个
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1] # 对置信度排序
                results[-1] = results[-1][argsort] # argsort 是排好序的位置
                results[-1] = results[-1][:keep_top_k] # 取出top_k个
        return results

def write_priorboxes(file_name):
    # 将default box写出到文件
    results = []
    img_size = (300, 300)

    x = tf.ones([1, 38, 38, 1])
    box = layers.PrioirBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2])(x)
    results.append(box)

    x = tf.ones([1, 19, 19, 1])
    box = layers.PrioirBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])(x)
    results.append(box)

    x = tf.ones([1, 10, 10, 1])
    box = layers.PrioirBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])(x)
    results.append(box)

    x = tf.ones([1, 6, 6, 1])
    box = layers.PrioirBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])(x)
    results.append(box)

    x = tf.ones([1, 3, 3, 1])
    box = layers.PrioirBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])(x)
    results.append(box)

    x = tf.ones([1, 1, 1, 1])
    box = layers.PrioirBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])(x)
    results.append(box)


    results = tf.concat(results, axis=1)
    results = tf.reshape(results, [-1, 8]).numpy()
    print(results)
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)

# write_priorboxes('prior_boxes_ssd300.pkl')
# 训练之前执行一下，生成default boxes

def draw(images, results, num_class, confidence=0.4):
    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= confidence]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, num_class)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            #         label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

        plt.show()