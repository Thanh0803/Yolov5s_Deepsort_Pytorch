import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import cv2
import math
import collections



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def calc(ImageDim, StadionDim, Point):
    return (Point[0] * StadionDim[0]/ImageDim[0], Point[1] * StadionDim[1]/ImageDim[1])

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes_tracking(img, bbox, identities, confidence, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        if  i < len(confidence):
            con_num = int(confidence[i]*100)
            color = compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            label_conf = '{}{:d}%'.format("",con_num)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), (0,255,0), -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            cv2.putText(img, label_conf, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img
    
def draw_boxes_detect_thief(img, bbox, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        label = '{}{:d}'.format("", 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 3)
        cv2.putText(img, "SOS", (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 255], 2)
    return img

def draw_boxes_detect(img, bbox, identities, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
       
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), (255,0,0), -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
# mark = 1
#                                 if (count > 30):
#                                     draw_boxes_detect_thief(frame,list(dict_tracking_coor[i]))
def draw_boxes_detect_suspect(img, bbox, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        label = '{}{:d}'.format("", 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 3)

    return img


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort_human = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    deepsort_vehicle = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    check_ID_longduong = []
    check_ID_viahe = []
    bbox_xyxy_human =[]
    bbox_xyxy_vehicle = []
    xy_center_vehicle = []
    dict_tracking_mid_human = {}
    dict_tracking_mid_human = collections.defaultdict(list)
    dict_tracking_mid_vehicle = {}
    dict_tracking_mid_vehicle = collections.defaultdict(list)
    mark=0
    mark_2 =0 
    mark_1=0
    count=0
    count_2 = 0
    element_spec = 0

    p1=np.array([300,0])
    p2=np.array([888,720])

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        
        if (mark ==1 or mark ==2) :
            count +=1
        elif mark == 0:
            count =0
        if (mark_1 == 3):
            count_2 +=1
        elif mark_1 == 0:
            count_2 = 0
        

        cv2.line(im0s,(300,0),(888,720),(255,0,0),1)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if (det is not None and len(det)) :
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                xywh_bboxs_human = []
                confs_human = []
                xywh_bboxs_vehicle = []
                confs_vehicle = []
                xywh_bboxs_vehicle = []

                
                for *xyxy, conf, cls in det:
                    
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    p5 = np.array([int(x_c),int(y_c+bbox_h/2)])
                    distance = (np.cross(p2-p1, p1-p5))/norm(p2-p1)
                    if  cls.item() == 0.0:

                        xywh_obj_human = [x_c, y_c, bbox_w, bbox_h]
                        im0 = cv2.circle(im0,(int(x_c), int(y_c + bbox_h/2)), 5, (0,255,0), thickness = -1)
                        xywh_bboxs_human.append(xywh_obj_human)
                        confs_human.append(conf.item())

                    if cls.item() == 3.0 and distance < 0:

                        xywh_obj_vehicle = [x_c, y_c, bbox_w, bbox_h]
                        # im0 = cv2.circle(im0,(int(x_c), int(y_c + bbox_h/2)), 5, (255,0,0), thickness = -1)
                        xywh_bboxs_vehicle.append(xywh_obj_vehicle)
                        confs_vehicle.append([conf.item()])
                        if (int(x_c), int(y_c + bbox_h/2)) not in xy_center_vehicle:
                            xy_center_vehicle.append([int(x_c), int(y_c+bbox_h/2),bbox_h])

                if len(xywh_bboxs_vehicle) == 0:
                    deepsort_vehicle.increment_ages()
                else:
                    xywhs_vehicle = torch.Tensor(xywh_bboxs_vehicle)
                    confss_vehicle = torch.Tensor(confs_vehicle)
                    outputs_vehicle = deepsort_vehicle.update(xywhs_vehicle, confss_vehicle, im0)

                    if len(outputs_vehicle) > 0:
                        bbox_xyxy_vehicle = outputs_vehicle[:, :4]
                        identities_vehicle = outputs_vehicle[:,-1]
                        draw_boxes_detect(im0, bbox_xyxy_vehicle,identities_vehicle)
                        for v,(iden_vehicle,bbox_vehicle) in enumerate(zip(identities_vehicle, bbox_xyxy_vehicle)) :
                            p6=np.array([int((bbox_vehicle[0]+bbox_vehicle[2])/2), int(bbox_vehicle[3]), int(bbox_vehicle[3]-bbox_vehicle[1])])
                            dict_tracking_mid_vehicle[iden_vehicle].append(p6)
                if len(xywh_bboxs_human) == 0:
                    deepsort_human.increment_ages()
                
                else:
                    dict_tracking_coor = {}
                    dict_tracking_coor = collections.defaultdict(list)
                    xywhs_human = torch.Tensor(xywh_bboxs_human)
                    confss_human = torch.Tensor(confs_human)
                    outputs_human = deepsort_human.update(xywhs_human, confss_human, im0)

                    if len(outputs_human) > 0:
                        bbox_xyxy_human = outputs_human[:, :4]
                        identities_human = outputs_human[:, -1]
                        draw_boxes_tracking(im0, bbox_xyxy_human, identities_human,confs_human)
                        for j,(iden_human,bbox_human) in enumerate(zip(identities_human, bbox_xyxy_human)) :
                            p3=np.array([int((bbox_human[0]+bbox_human[2])/2), int(bbox_human[3])])
                            p4=np.array([int((bbox_human[0]+bbox_human[2])/2), int((bbox_human[1]+bbox_human[3])/2)])
                            dict_tracking_mid_human[iden_human].append(p3)
]                            if (iden_human != element_spec):
                                dict_tracking_coor[iden_human].append(bbox_human)  
                            d = (np.cross(p2-p1, p1-p3))/norm(p2-p1)
                            if d>0 and (iden_human not in check_ID_longduong):
                                check_ID_longduong.append(iden_human)
                            if d<0 and (iden_human not in check_ID_longduong) and (iden_human not in check_ID_viahe):
                                check_ID_viahe.append(iden_human)
                            if d<0 and (iden_human in check_ID_longduong):
                                mark_1 = 3
                                element = iden_human                                

                                check_ID_viahe.append(element)   
                                
                                for key in dict_tracking_mid_vehicle:
                                    for val,(mid_human,mid_vehic) in enumerate(zip(dict_tracking_mid_human[element],dict_tracking_mid_vehicle[key])):
                                        dis_hum_vehic = math.sqrt((mid_vehic[0] - mid_human[0])**2 + (mid_vehic[1] - mid_human[1])**2)
                                        if ((mid_human[1] <= mid_vehic[1]) and (dis_hum_vehic <= (mid_vehic[2])*0.75)) or ((mid_human[1] > mid_vehic[1]) and (dis_hum_vehic <=  (mid_vehic[2]/2))):
                                            key_spec = key
                                            for mid_vehic_spec in dict_tracking_mid_vehicle[key_spec]:
                                                dis_hum_vehic_spec = math.sqrt((mid_vehic_spec[0] - mid_human[0])**2 + (mid_vehic_spec[1] - mid_human[1])**2)
                                                if (mid_human[1] <= mid_vehic[1]) and (dis_hum_vehic_spec <=  (mid_vehic[2])*0.75):
                                                    mark = 1
                                                    if (count > 30) and (mark_2 !=3) :
                                                        draw_boxes_detect_thief(im0,list(dict_tracking_coor[element]))
                                                elif (mid_human[1] > mid_vehic[1]) and (dis_hum_vehic_spec <=  (mid_vehic[2])*0.7):
                                                    mark = 2
                                                    if (count > 30) and (mark_2 !=3):
                                                        draw_boxes_detect_thief(im0,list(dict_tracking_coor[element]))
                                                else:
                                                    mark = 0 
                                                    mark_1 = 0

                                if (count == count_2) and (count !=0):
                                    mark_2 = 3
                                    element_spec = element
                                else:
                                    draw_boxes_detect_suspect(im0,list(dict_tracking_coor[element]))
                            else:
                                mark_1 = 0
                            
                                              
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
            

    if save_txt or save_vid:
        # print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    # print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
