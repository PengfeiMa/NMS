### 1. sort results by conf
### 2. caculate the iou with the hight conf socre obj
### 3. del the hight iou objs
### 4. recur 1-3

## Fast R-CNN

import numpyt as np 

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

    # soft-NMS



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





