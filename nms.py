### 1. sort results by conf
### 2. caculate the iou with the hight conf socre obj
### 3. del the hight iou objs
### 4. recur 1-3

## Fast R-CNN

import numpy as np 

# det: 检测的 boxes及对应的socres;
# thresh: 设定的阈值

def nms(dets, thresh):
     # boxes 
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] 

    keep = []
    while order.size() > 0:
        i = order[0]
        keep.append(i)

        # caculate iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2-xx1 + 1)
        h = np.maximum(0.0, yy2-yy1 + 1)
        inter = w * h
        ovr = inter/(areas[i] + areas[order[1:]]-inter)
        
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

# Soft-NMS: Improving Object Detection With One Line of Code
def soft_nms(dets, sigma=0.5, Nt=0.5, method=2, threshold=0.1):
    box_len = len(dets) 
    for i in range(box_len):
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]
        max_pos = i
        max_scores = ts

        # get max boxes
        pos = i + 1
        #print(dets[pos, 4])
        while pos < box_len:
            if max_scores < dets[pos, 4]:
                max_scores = dets[pos, 4]
                max_pos = pos
            pos += 1

        # add max box as a detection
        dets[i, :] = dets[max_pos, :]

        # swap i-th box with position of max box
        dets[max_pos, 0] = tmpx1
        dets[max_pos, 1] = tmpy1
        dets[max_pos, 2] = tmpx2
        dets[max_pos, 3] = tmpy2
        dets[max_pos, 4] = ts

        # give the hightest scores to tmp
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]
        pos = i+1
        # NMS iteration, note that box_len changes if detection boxes fall below threshold
        while pos < box_len:
            x1, y1, x2, y2 = dets[pos, 0], dets[pos, 1], dets[pos, 2], dets[pos, 3]
            areas = (x2 -x1 + 1)*(y2 -y1 +1)
            iw = (min(tmpx2, x2) - max(tmpx1, x1) + 1)
            ih = (min(tmpy2, y2) - max(tmpy1, y1) + 1)
            if iw > 0 and ih > 0:
                overlaps = iw * ih
                ious = overlaps/((tmpx2-tmpx1+1)*(tmpy2-tmpy1+1)+areas-overlaps)

                if method ==1: # linear
                    if ious > Nt:
                        weight = 1-ious
                    else:
                        weight =1
                elif method ==2:
                    weight = np.exp(-(ious**2)/sigma)
                else:
                    if ious > Nt:
                        weight = 0
                    else:
                        weight =1
                # give new conf
                dets[pos,4] = weight* dets[pos, 4]

                # if box's score < thresh, swap with the last one
                if dets[pos, 4] < threshold:
                    dets[pos, 0] = dets[box_len-1, 0]
                    dets[pos, 1] = dets[box_len-1, 1]
                    dets[pos, 2] = dets[box_len-1, 2]
                    dets[pos, 3] = dets[box_len-1, 3]
                    dets[pos, 4] = dets[box_len-1, 4]

                    box_len = box_len-1
                    pos = pos-1
            pos += 1
    keep = [i for i in range(box_len)]
    return keep






# Fast-NMS    YOLACT  Fast
# https://blog.csdn.net/qq_40263477/article/details/103881569?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-5 
def fast_nms(self, boxes, masks, scores, iou_threshold: float=0.5, top_k: int=200, second_treshold:bool=False):
    '''
    boxes: torch.size([num_dets, 4])
    masks: torch.size([num_dets, 32])
    scores: torch.size([num_classes, num_dets])
    '''

    # step 1: sort by score get top_k
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]
    num_classes, num_dets = idx.size()

    boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    masks = masks[idx.view(-1), :].view(num_classes, num_dets, 4)

    # step 2: caculate iou between boxes
    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)  # triu_()取上三角 tril_()取下三角 此处将矩阵的下三角和对角线元素删去
    iou_max, _ = iou.max(dim=1) # 按列取大值 torch.Size([num_classes, num_dets])

    keep = (iou_max <= iou_threshold)

    if second_treshold:
        keep *= (score > self.conf_thresh)
    
    classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)

    classes = classes[keep]
    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]
    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:cfg.max_num_detections]
    scores = scores[:cfg.max_num_detections]
    classes = classes[idx]
    boxes = boxes[idx]
    masks = masks[idx]
    return boxes, masks, classes, scores


if __name__ == '__main__':
    dets = [[0, 0, 100, 101, 0.9], [5, 6, 90, 110, 0.7], [17, 19, 80, 120, 0.8], [10, 8, 115, 105, 0.5]]
    dets = np.array(dets)
    #result = soft_nms(dets, 0.5)
    result = soft_nms(dets, threshold= 0.5)
    print(result)



