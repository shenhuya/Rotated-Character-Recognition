# 测试在GPU上每张图片推理所需要的时间以及准确率
from mmdet.apis import init_detector, inference_detector
import mmcv
import mmrotate
import os
import time
import glob
import numpy as np
import os.path as osp
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
import copy
import math
import cv2
import operator

CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )
cls_map = {c: i
                   for i, c in enumerate(CLASSES)
                   }  # in mmdet v2.0 label is 0-based

def wh_dets(dets):
    '''
    网络得到的预测结果bbox的w可能大于也可能小于h，将其统一调整为w < h
    Args:
        dets (list): 预测的检测结果
    '''
    indx = np.where(dets[:,2] > dets[:,3])
    # print('indx: ', indx)
    if len(indx[0]) == 0:
        return dets
    # 将w, h互换
    else :
        # print(dets[indx, 2])
        temp = dets[indx, 2]
        dets[indx, 2] = dets[indx, 3]
        dets[indx, 3] = temp
        for i in indx[0]:
            if dets[i][4] > 0:
                dets[i][4] = dets[i][4] - np.pi/2
            else:
                dets[i][4] = np.pi/2 + dets[i][4]
    return dets

def get_angle(dets, flag=False):
    '''根据输入的bbox和landm结果，对角度进行360°处理，得到图像旋转的角度

    Args:
        dets (List): 检测结果(bbox(x,y,w,h,theta) + landm(tl_x,tl_y,tr_x,tr_y,cs,cy) + score)
        flag (Bool): 是否根据左上右上直接计算得到的角度对bbox变化得到的角度进行修正
    '''
    # print('dets: ', dets)
    temp_angle = copy.deepcopy(dets[:, 4])
    for i in range(len(dets)):
        # 获取左上右上点的中心点
        ctltr_x = (dets[i][5] + dets[i][7])/2
        ctltr_y = (dets[i][6] + dets[i][8])/2
        if ctltr_x < dets[i][0] and ctltr_y < dets[i][1]: 
            # 范围应该0-90°，theta值应该为负，出现正值说明此时可能在90°或者360°附近
            # 如果theta为正，说明检测框稍微超过90°，或者不足0°（360°）
            if dets[i][4] > 0:
                # 如果与中心点x坐标很接近，说明此时检测框不足0°
                if abs(ctltr_x - dets[i][0]) < 5: 
                    dets[i][4] = dets[i][4] - 2*np.pi
                # 如果中心点y坐标很接近，说明此时检测框超过90°
                elif abs(ctltr_y - dets[i][1]) < 5:
                    dets[i][4] = dets[i][4] - np.pi
        elif ctltr_x < dets[i][0] and ctltr_y >= dets[i][1]: 
            # 范围应该在90°-180°之间，theta值应该为正，出现负值说明此时可能在90°或者180°附近
            if dets[i][4] < 0:
                # 如果与中心点x坐标很接近，说明此时检测框超过180°
                if abs(ctltr_x - dets[i][0]) < 5: 
                    dets[i][4] = dets[i][4] - np.pi
                # 如果中心点y坐标很接近，说明此时检测框不足90°
                elif abs(ctltr_y - dets[i][1]) < 5:
                    dets[i][4] = dets[i][4]
            else:
                dets[i][4] = dets[i][4] - np.pi 
        elif ctltr_x >= dets[i][0] and ctltr_y >= dets[i][1]:
            # 范围应该在180°-270°之间，theta值应该为负，出现正值说明此时可能不足180°或超过270°
            if dets[i][4] > 0:
                # 如果与中心点x坐标很接近，说明此时检测框不足180°
                if abs(ctltr_x - dets[i][0]) < 5: 
                    dets[i][4] = dets[i][4] - np.pi
                # 如果中心点y坐标很接近，说明此时检测框超过270°
                elif abs(ctltr_y - dets[i][1]) < 5:
                    dets[i][4] = dets[i][4] - np.pi*2
            else:
                dets[i][4] = dets[i][4] - np.pi 
        elif ctltr_x >= dets[i][0] and ctltr_y < dets[i][1]:
            # 范围应该在270°-360°之间，theta值应该为正，出现负值说明此时可能不足270°或超过360°
            if dets[i][4] < 0:
                # 如果与中心点x坐标很接近，说明此时检测框超过360°
                if abs(ctltr_x - dets[i][0]) < 5: 
                    dets[i][4] = dets[i][4]
                # 如果中心点y坐标很接近，说明此时检测框不足270°
                elif abs(ctltr_y - dets[i][1]) < 5:
                    dets[i][4] = dets[i][4] - np.pi
            else:
                dets[i][4] = dets[i][4] - np.pi*2 
        # 用左上右上计算得到的角度进行修正
    if flag:
        tl_x = dets[:, 5]
        tl_y = dets[:, 6]
        tr_x = dets[:, 7]
        tr_y = dets[:, 8]
        dx = tr_x - tl_x
        # 图像的y坐标和原点是反的
        dy = tl_y - tr_y
        for j in range(len(dx)):
            if math.atan2(dy[j], dx[j]) > 0:
                temp_angle[j] = math.atan2(dy[j], dx[j]) 
            else:
                temp_angle[j] = 2*np.pi + math.atan2(dy[j], dx[j]) 
            # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于5°
            if abs(temp_angle[j] + dets[j][4]) > 0.09:
            # # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于10°
            # if abs(temp_angle[j] + dets[j][4]) > 0.18:
            # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于20°
            # if abs(temp_angle[j] + dets[j][4]) > 0.35:
            # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于30°
            # if abs(temp_angle[j] + dets[j][4]) > 0.5:
                dets[j][4] = -temp_angle[j]
    # print('angles: ', temp_angle)
    # print('bbox_angles: ', dets[:,4])
       
    # print('rot_angles: ', dets[:,4]/np.pi*180)
    rot_angle = np.median(dets[:, 4])
    # rot_angle = np.average(dets[:, 4])
    # print('rot_angle: ', rot_angle)
    return rot_angle

def NMS(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 0] + dets[:, 2]
    y2 = dets[:, 1] + dets[:, 3]
    scores = dets[:, -1]

    tags_iou = {} #用于保留所有框与其置信度大于0.5框的对应关系
    tag_warnning = 0 # 出现重叠框的警告
    #每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #按照score置信度降序排序
    order = scores.argsort()[::-1]
    # print('areas: ', areas)
    # print('order: ', order)
    keep = [] #保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i) #保留该类剩余box中得分最高的一个
        #得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        # print('inds: ', inds)
        # 得到IoU大于阈值的bbox，
        # inds_1 = np.where(ovr > thresh)[0]
        # print('inds_1: ', inds_1)
        # 如果其置信度大于0.5，仍然保留
        # tag_iou = [] #用于保留当前框置信度大于0.5的重叠框
        # for i in inds_1:
        #     if scores[order[i]] > 0.5:
        #         keep.append(order[i+1])
        #         tag_iou.append(order[i+1])
        #         tag_warnning = 1
        # if len(tag_iou) != 0:
        #     tags_iou[order[0]] = tag_iou
        # print('order: ', order)
        # print('tags_iou: ', tags_iou)
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
        #print('order: ', order)

    #返回保留的bbox
    return sorted(keep)
    #返回保留的bbox，以及出现重叠且置信度较大的警告
    # return sorted(keep), tag_warnning, tags_iou

def rot_img_360(dets, im, angle):
    '''对图像进行0-360°旋转矫正

    Args:
        dets (_type_): _description_
        im (_type_): _description_
        angle (_type_): _description_
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    im_w, im_h = im.shape[1], im.shape[0]
    cx, cy = 0.5 * im_w, 0.5 * im_h
    # print('cx: ', cx)
    # print('cy: ', cy)
    # angle = dets[0][4]/math.pi*180 #角度制
    # angle = np.median(dets[:,4])/math.pi*180
    # rot_angle = -dets[0][4] #弧度制
    rot_angle = angle / math.pi*180 #角度制
    # print('angle: ', angle)
    # print('rot_angle: ', rot_angle)
    # 逆时针旋转angle
    M = cv2.getRotationMatrix2D((cx,cy), rot_angle , 1)
    new_w = int(im_h * np.abs(M[0, 1]) + im_w * np.abs(M[0, 0]))
    new_h = int(im_h * np.abs(M[0, 0]) + im_w * np.abs(M[0, 1]))
    # print('new_w: ', new_w)
    # print('new_h: ', new_h)
    # 调整旋转矩阵以考虑平移, 使得整个图像信息不丢失
    M[0, 2] += (new_w - im_w) / 2
    M[1, 2] += (new_h - im_h) / 2
    image_rotation = cv2.warpAffine(src=im, M=M, dsize=(new_w, new_h), borderValue=(255, 255, 255))
    ncx, ncy = 0.5 * image_rotation.shape[1], 0.5 * image_rotation.shape[0]
    # print('ncx: ', ncx)
    # print('ncy: ', ncy)
    # 对旋转后的图像的bbox的中心点也进行相应旋转
    for i in range(len(dets)):
        x = dets[i][0]
        y = dets[i][1]
        dets[i][0] = math.cos(-angle) * (x - cx) - math.sin(-angle) * (y- cy) + ncx
        dets[i][1] = math.sin(-angle) * (x - cx) + math.cos(-angle) * (y - cy) + ncy      
        cv2.circle(image_rotation, (int(dets[i][0]), int(dets[i][1])), 5, (0,0,255))
        cv2.line(image_rotation, (int(dets[i][0] - dets[i][2]/2), int(dets[i][1] - dets[i][3]/2)),
                                 (int(dets[i][0] + dets[i][2]/2), int(dets[i][1] - dets[i][3]/2)), (0,0,255))
    # print("dets-rot: ", dets)
    # cv2.imwrite('/data1/hzj/mmrotate/show-dir/recog_wrong/' + osp.basename(img).replace('.bmp','-new.bmp'),image_rotation)

    return dets

def res_out_360(dets, labels):
    '''图像通过360°矫正后的识别结果排列输出

    Args:
        dets (list): 矫正后的检测框的检测框（仅中心点进行矫正）
        labels (np.array): 检测框的标签
    '''
    # print('labels: ', labels)
    sort = []
    labels_list = []
    for i in range(10):
    # for i in range(36):
        for j in range(len(labels[i])):
            labels_list.append(labels[i][j])
    # print('labels_list: ', labels_list)
    labels_list = np.array(labels_list)
    x = dets[:,0]
    y = dets[:,1]
    # 获取x,y从小到大排序
    indx_x = x.argsort()
    indx_y = y.argsort()
    indx = 0 
    indx_hang1 = []
    indx_hang2 = []
    indx_hang3 = []
    indx = 0
    y_temp = copy.deepcopy(y)
    y_temp -= y_temp[indx_y[0]]
    indx_hang1 = np.where(y_temp < 50)[0]
    # print('indx_y: ', indx_y)
    # 删除第一行的索引
    count = 0
    indx = 0 
    while(count < len(indx_hang1)):
        if indx_y[indx] in indx_hang1:
            count += 1
            indx_y = np.delete(indx_y, indx)
        else:
            indx += 1
    # print('indx_hang1: ', indx_hang1)
    # print('indx_y: ', indx_y)
    if len(indx_y) != 0:
        y_temp = copy.deepcopy(y)
        y_temp -= y_temp[indx_y[0]]
        indx_hang2 = np.setdiff1d(np.where(y_temp < 50)[0], indx_hang1)
        # 删除第二行的索引
        count = 0
        indx = 0 
        while(count < len(indx_hang2)):
            if indx_y[indx] in indx_hang2:
                count += 1
                indx_y = np.delete(indx_y, indx)
            else:
                indx += 1
    indx_hang3 = indx_y
    hang1x_indx = x[indx_hang1].argsort()
    hang2x_indx = x[indx_hang2].argsort()
    hang3x_indx = x[indx_hang3].argsort()
    for i in hang1x_indx:
        sort.append(indx_hang1[i])
    for i in hang2x_indx:
        sort.append(indx_hang2[i])
    for i in hang3x_indx:
        sort.append(indx_hang3[i]) 
    # print('sort: ', sort)
    # print('res_lable: ', labels_list[sort])
    
    return labels_list[sort].tolist()

def load_label(ann_file, version):
    '''
    输入的参数：
    对应检测图像的标签文本
    角度类别：默认为('oc')
    '''  
    cls_map = {c: i
                   for i, c in enumerate(CLASSES)
                   } 

    gt_bboxes = []
    gt_labels = []
    gt_polygons = []

    with open(ann_file) as f:
        s = f.readlines()
        for si in s:
            bbox_info = si.split()
            poly = np.array(bbox_info[:8], dtype=np.float32)
            try:
                x, y, w, h, a = poly2obb_np(poly, version)
            except:  # noqa: E722
                continue
            cls_name = bbox_info[8]
            difficulty = int(bbox_info[9])
            label = cls_map[cls_name]
            if difficulty > 100:
                pass
            else:
                gt_bboxes.append([x, y, w, h, a])
                gt_labels.append(label)
                gt_polygons.append(poly)

    return gt_labels

config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip/best_acc.pth'

dir_path = '/data1/hzj/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270-new/test/images'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0', cfg_options={'model.test_cfg.score_thr':0.5})

# save_path = 'C:/Users/10150/Desktop/windows_mmrot'

# if not os.path.exists(save_path):
#     os.mkdir(save_path)
  
# 测试单张图片  
# result = inference_detector(model, dir_path)
# dets = np.vstack(result)
# print('result: ', dets)

# 批量测试图片
img_names = glob.glob(dir_path+'/*bmp')
# img_names = glob.glob(dir_path+'/*jpg')
num_images = len(img_names)
acc_count = 0
begin = time.time()
for i, img in enumerate(img_names):
    # test a single image and show the results
    # img = os.path.join(dir_path, img_name)  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    dets = np.vstack(result)
    print('img: ', img)
    
    result_temp = copy.deepcopy(result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    # print('labels: ', labels)
    dets = np.vstack(result_temp)
    # 将检测结果完全根据关键点信息得到的，通过关键点信息得到检测框
    # 将dets都调整为 w < h
    dets = wh_dets(dets)
    # 获取360°范围的旋转角度
    # rot_angle = get_angle(dets)
    rot_angle = get_angle(dets, True)
    # print('dets: ', dets)
    # 将图片，检测结果进行旋转
    im = cv2.imread(img)
    dets = rot_img_360(dets, im, rot_angle)
    #对图像中可能存在的多个类别检测框检测出同一目标进行NMS后处理，只保留confidence最高的那个类别
    nms_dets = NMS(dets, thresh=0.1)
    # print('nms_dets: ', nms_dets)
    dets = dets[nms_dets]

    indx = 0
    all_del = 0 # 记录

    for i in range(10):
        tag = 0
        if len(result[i]) == 0:
            continue
        else:
            for j in range(len(result[i])):
                if indx not in nms_dets:
                    result[i] = np.delete(result[i], j-tag, 0)
                    labels[i] = np.delete(labels[i], j-tag, 0)
                    tag = tag + 1
                indx += 1
    # print('img: ', img)
    # print('dets: ', dets)
    # print('result: ', result)
    # print('labels: ', labels)
    # print(save_path)
    # model.show_result(img, result, out_file=os.path.join(save_path, os.path.basename(img)))
    recog_result = res_out_360(dets, labels)
    ann_file = img.replace('bmp', 'txt').replace('images', 'annfiles')
    gt_labels = load_label(ann_file, 'oc')
    if operator.eq(gt_labels, recog_result):
        acc_count = acc_count + 1
    
    print(recog_result)
    end = time.time()
    
    # or save the visualization results to image files
    # model.show_result(img, result, out_file=os.path.join(save_path, os.path.basename(img)), bbox_color=(0, 255, 255))
end = time.time()
print("all_time: " + str(end - begin))
print("average time: " + str((end - begin)/(num_images-1)*1000) + ' ms')
print(acc_count/(num_images) * 100, '%')