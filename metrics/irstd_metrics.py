import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])


class mIoU():
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        # preds: [B, 1, H, W] or [B, H, W]
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels,1)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class SamplewiseSigmoidMetric():
    def __init__(self, nclass=1, score_thresh=0.5):
        super(SamplewiseSigmoidMetric, self).__init__()
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        inter_arr, union_arr = samplewise_intersection_union(preds, labels, self.nclass, self.score_thresh)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        iou = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        niou = iou.mean() if self.total_union.size else 0.0
        return iou, float(niou)

    def reset(self):
        self.total_inter = np.array([])
        self.total_union = np.array([])


class PD_FA():
    def __init__(self):
        super(PD_FA, self).__init__()
        self.reset()

    def update(self, preds, labels):
        # preds/labels: [B, 1, H, W]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        batch_size = preds.shape[0]
        for b in range(batch_size):
            pred_map = preds[b]
            label_map = labels[b]
            if pred_map.ndim == 3:
                pred_map = pred_map[0]
            if label_map.ndim == 3:
                label_map = label_map[0]

            pred_map = (pred_map > 0).astype(np.int64)
            label_map = label_map.astype(np.int64)

            image = measure.label(pred_map, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(label_map, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target += len(coord_label)
            self.all_pixel += (pred_map.shape[0] * pred_map.shape[1])

            matched_indices = set()

            # 目标重心距离 < 3 像素判为命中。
            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    if m in matched_indices:
                        continue
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    if distance < 3:
                        self.PD += 1
                        matched_indices.add(m)
                        break

            # 未命中的预测连通域计入虚警像素。
            for m in range(len(coord_image)):
                if m not in matched_indices:
                    self.dismatch_pixel += coord_image[m].area

    def get(self):
        Final_FA = self.dismatch_pixel / (self.all_pixel + 1e-10)
        Final_PD = self.PD / (self.target + 1e-10)
        return Final_PD, Final_FA

    def reset(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0



def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


def samplewise_intersection_union(output, target, nclass, score_thresh):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())
    num_sample = intersection.shape[0]
    area_inter_arr = np.zeros(num_sample)
    area_union_arr = np.zeros(num_sample)

    for b in range(num_sample):
        area_inter, _ = np.histogram(intersection[b].detach().cpu().numpy(), bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict[b].detach().cpu().numpy(), bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target[b].detach().cpu().numpy(), bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        area_inter_arr[b] = area_inter.item()
        area_union_arr[b] = area_union.item()
        assert (area_inter <= area_union).all(), "Error: Intersection area should be smaller than Union area"

    return area_inter_arr, area_union_arr
