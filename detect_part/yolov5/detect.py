# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from unittest import skip

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box ,My0_save_one_box
# from utils.plots import Annotator, colors, save_one_box , crop_plot
from utils.torch_utils import select_device, smart_inference_mode
# #000 #00E #copy----------------------------------
All_LC=[]
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# from os.path import splitext
# from keras.models import model_from_json

# from a0_MyFuction import grayscale ,noise_removal ,thin_font ,detect_lp ,UPSx8 ,My_preprocess_image ,My_get_plate ,Data

# All_Char=['0','1','2','3','4','5','6','7','8','9','ก','ข','ฃ','ค','ฅ','ฆ','ง','จ'
#           ,'ฉ','ช','ซ','ฌ','ญ','ฎ','ฏ','ฐ','ฑ','ฒ','ณ','ด','ต','ถ','ท','ธ','น'
#           ,'บ','ป','ผ','ฝ','พ','ฟ','ภ','ม','ย','ร','ล','ว','ศ','ษ','ส','ห','ฬ'
#           ,'อ','ฮ']
# res_To_BlackEnd=[]


# def load_model(path):
#     try:
#         path = splitext(path)[0]
#         with open('%s.json' % path, 'r') as json_file:
#             model_json = json_file.read()
#         model = model_from_json(model_json, custom_objects={})
#         model.load_weights('%s.h5' % path)
#         print("Loading model successfully...")
#         return model
#     except Exception as e:
#         print(e)

# # wpod_net_path = "./TestAdjust-LC-Main/wpod-net.json"
# wpod_net = load_model("./yolov5/wpod-net.json")
#------------------------------------

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0 (webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        #000 skip frame 0 รับค่าตัวแปล
        skipframe=0,  #000
        #000 legion detect 0 รับค่าตัวแปล
        RG=[0,0],
        #000
        kkkkk=0
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # crop_img = opt.crop
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #000 #xxx Main edit

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    #000 #00E #copy------------------------------------
    MyEdit=True
    # MyEdit=False
    
    if MyEdit:
        #LC
        print("Load-model m.LC")
        weights_LC = ['./yolov5/0_Model/license.pt']
        model_LC = DetectMultiBackend(weights_LC, device=device, dnn=dnn, data=data, fp16=half)
        stride_LC, names_LC, pt_LC = model_LC.stride, model_LC.names, model_LC.pt
        imgsz_LC = check_img_size(imgsz, s=stride_LC)  # check image size
        
        #PS
        print("Load-model m.PS")            # class 0 yolov5
        weights_PS = ['./yolov5/0_Model/yolov5s.pt']
        model_PS = DetectMultiBackend(weights_PS, device=device, dnn=dnn, data=data, fp16=half)
        stride_PS, names_PS, pt_PS = model_PS.stride, model_PS.names, model_PS.pt
        imgsz_PS = check_img_size(imgsz, s=stride_PS)  # check image size
        
        #HM
        print("Load-model m.HM")            # class 0 yolov5
        weights_HM = ['./yolov5/0_Model/helmet_last.pt']
        model_HM = DetectMultiBackend(weights_HM, device=device, dnn=dnn, data=data, fp16=half)
        stride_HM, names_HM, pt_HM = model_HM.stride, model_HM.names, model_HM.pt
        imgsz_HM = check_img_size(imgsz, s=stride_HM)  # check image size
    # # ------------------------------------

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    #000 #00E #copy------------------------------------
    if MyEdit:
        #LC
        print("WarmUp m.LC")
        model_LC.warmup(imgsz=(1 if pt_LC or model_LC.triton else bs, 3, *imgsz_LC))  # warmup
        dt_LC = (Profile(), Profile(), Profile())
        
        #PS
        print("WarmUp m.PS")
        model_PS.warmup(imgsz=(1 if pt_PS or model_PS.triton else bs, 3, *imgsz_PS))  # warmup
        dt_PS = (Profile(), Profile(), Profile())
        
        #HM
        print("WarmUp m.HM")
        model_HM.warmup(imgsz=(1 if pt_HM or model_HM.triton else bs, 3, *imgsz_HM))  # warmup
        dt_HM = (Profile(), Profile(), Profile())
    # # ------------------------------------
    
    #000 skip frame 1 (set up)
    SkipFrame=True # เอาไว้หยุดด้วยมือ
    SF=0
    # print("\n Zzzzzzzasd ")
    # print(len(dataset))
    # print(weights)
    # ขั้นข้างล่างน่าจะเป็นขั้น predictions ของทุก frame
    for path, im, im0s, vid_cap, s in dataset:
        
        
        with dt[0]:
            im_MoTo = torch.from_numpy(im).to(model.device)
            im_MoTo = im_MoTo.half() if model.fp16 else im_MoTo.float()  # uint8 to fp16/32
            im_MoTo /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im_MoTo.shape) == 3:
                im_MoTo = im_MoTo[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im_MoTo, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        #000    อันบนไม่เกียว
        # print(pred)
        
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            #000    det = pred
            # print("\n xxx")
            # print(det)
            
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                #000
                # 
                # print("\nzzzzzzzzz") ทำทุกอัน

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            #00E #copy
            # save_path = str(save_dir / ("Moto_" + p.name))  # im.jpg
            #000
            # print("LC_" + p.name)
            
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im_MoTo.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            #000 skip frame 2 ใช้ skip frame
            if SF and SkipFrame:
                # print("\nZzzzzzzasd")
                # print(SF)
                s += "skip frame #000 "
                SF-=1
                continue
            # print(s)  ล่างสุดของ for loop เพิ่ม ไม่ดีเทคกะเวลา Ex.image 14/14 D:\0_project\0_Git-Project\Demo\Demo_Pic2\frame12042.jpg: 384x640 1 Motorcycle, 2 Motorcycle+Drivers,
            # print(len(det))   จำนวนทุกคราสที่ค้นพบ
            # print(det)
            
            #000 ---- เจอก็ทำด้านล่าง
            
            
            
            # น่าจะถ้าเจอก็ทำด้านล่าง
            if len(det):
                #000 skip frame 3 ใช้เพิ่มจำนวนที่จะ skip
                SF+=skipframe
                # print("\n Zzzzzzzasd ")
                # print(SF)
                #000 #00E #copy------------------------------------
                if MyEdit:
                    #LC-xxxxxxxxxxxxxxxxxxxxxxxx
                    print("dt_LC m.LC")
                    with dt_LC[0]:
                        im_LC = torch.from_numpy(im).to(model_LC.device)
                        im_LC = im_LC.half() if model_LC.fp16 else im_LC.float()  # uint8 to fp16/32
                        im_LC /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im_LC.shape) == 3:
                            im_LC = im_LC[None]  # expand for batch dim

                    # Inference
                    with dt_LC[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred_LC = model_LC(im_LC, augment=augment, visualize=visualize)

                    # NMS
                    with dt_LC[2]:
                        pred_LC = non_max_suppression(pred_LC, conf_thres, iou_thres, 0, agnostic_nms, max_det=max_det)
            
                    # print(pred_LC)
            
                    #PS-xxxxxxxxxxxxxxxxxxxxxxxx
                    print("dt_PS m.PS")
                    with dt_PS[0]:
                        im_PS = torch.from_numpy(im).to(model_PS.device)
                        im_PS = im_PS.half() if model_PS.fp16 else im_PS.float()  # uint8 to fp16/32
                        im_PS /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im_PS.shape) == 3:
                            im_PS = im_PS[None]  # expand for batch dim

                    # Inference
                    with dt_PS[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred_PS = model_PS(im_PS, augment=augment, visualize=visualize)

                    # NMS
                    with dt_PS[2]:  ###
                        pred_PS = non_max_suppression(pred_PS, conf_thres, iou_thres, 0, agnostic_nms, max_det=max_det)
                    
                    # print(pred_PS)
                    # print(len(pred_PS[0]))
                    # if (len(pred_PS[0]))>2:
                    #     print("false - PS")
                    # else:
                    #     print("true - PS")
                    
                    #HM-xxxxxxxxxxxxxxxxxxxxxxxx
                    print("dt_HM m.HM")
                    with dt_HM[0]:
                        im_HM = torch.from_numpy(im).to(model_HM.device)
                        im_HM = im_HM.half() if model_HM.fp16 else im_HM.float()  # uint8 to fp16/32
                        im_HM /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im_HM.shape) == 3:
                            im_HM = im_HM[None]  # expand for batch dim

                    # Inference
                    with dt_HM[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred_HM = model_HM(im_HM, augment=augment, visualize=visualize)

                    # NMS
                    with dt_HM[2]:  ###
                        pred_HM = non_max_suppression(pred_HM, conf_thres, iou_thres, 0, agnostic_nms, max_det=max_det)
                    
                    
                    # print(pred_HM)
                    # print("FOR_test")
                    # print(pred_PS[0])
                    pred_PS[0][:, :4] = scale_coords(im_PS.shape[2:], pred_PS[0][:, :4], im0.shape).round()
                    pred_HM[0][:, :4] = scale_coords(im_HM.shape[2:], pred_HM[0][:, :4], im0.shape).round()
                    pred_LC[0][:, :4] = scale_coords(im_LC.shape[2:], pred_LC[0][:, :4], im0.shape).round()
                    # print(pred_PS[0])
                    # print(pred_PS[0][0])    # tensor([173.00000, 410.00000, 278.00000, 620.00000,   0.77452,   0.00000])
                    # print(pred_PS[0][0][0]) # tensor(173.)
                    # print(len(pred_PS[0]))    # 3
                    
                    # *psps, conf, cls = pred_PS[0][0]
                    # print(psps)           moto base [tensor(1113.), tensor(484.), tensor(1334.), tensor(730.)]
                    
                #-------------------------------------------------------
                
                #000
                # print("BF-cal")
                # print(det)
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im_MoTo.shape[2:], det[:, :4], im0.shape).round()
                
                #000
                # print("AF-cal")
                # print(det)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #000 อัน for loop ด้านบน แต่ใส่ตรงนี้เพราะจะได้ดูง่ายๆ
                # print("\n Zzzzzzzasd")
                # print(c)  #รหัส class
                # print(n) # ต่อคราสใน loop Ex. 1 Motorcycle, 2 Motorcycle+Drivers ได้ tensor(1) tensor(2) ตามลำดับลูป
                # print(s) # Texe ที่ขาดเวลาประมวณผล Ex.image 14/14 D:\0_project\0_Git-Project\Demo\Demo_Pic2\frame12042.jpg: 384x640 1 Motorcycle, 2 Motorcycle+Drivers,
                # print(names[int(c)])
                
                #000
                # kkkkk=0
                # print("\n Zzzzzzzasd")
                # print(xyxy[0])    Error
                
                #000 
                # print("BFxyxy-cal")
                # print(det)
                
                for *xyxy, conf, cls in reversed(det):
                    # Sub_LC=SubLC()
                    # #000
                    print("\nxyxy")
                    print(xyxy)
                    # 000 legion detect 1
                    if (RG[0]>xyxy[0] or RG[1]>xyxy[1]) and (RG[0]!=0 or RG[1]!=0): # >800 ติดรถขวา
                        # print("yes")
                        s += "Dont in Legion #000 "
                        continue
                    #000 #00E #copy-----------------------------
                    if MyEdit:
                        print("xyxy - myEdit")
                        res_PS, res_HM, res_LC=[],[],[]
                        # print(pred_PS[0][0])    # tensor([173.00000, 410.00000, 278.00000, 620.00000,   0.77452,   0.00000])
                        # print(pred_PS[0][0][0]) # tensor(173.)
                        # print(len(pred_PS[0]))    # 3
                        # moto base [tensor(1113.), tensor(484.), tensor(1334.), tensor(730.)]
                        # Add_more_range = 50
                        for i in range(len(pred_PS[0])):
                            if(xyxy[0]-50<=pred_PS[0][i][2]<=xyxy[2]+50 and xyxy[1]-50<pred_PS[0][i][3]<=xyxy[3]+50):
                                res_PS.append(pred_PS[0][i])
                        # print(res_PS)

                        # ติดเรื่องเช็คหัวกะคน
                        for i in range(len(pred_HM[0])):
                            if(xyxy[0]<=pred_HM[0][i][2]<=xyxy[2] and xyxy[1]-250<pred_HM[0][i][3]<=xyxy[3]):
                                res_HM.append(pred_HM[0][i])
                        # print(res_HM)

                        
                        cause_My = []
                        if (len(res_PS)>2):
                            cause_My.append("ซ้อนเกิน")
                        if (len(res_PS)>len(res_HM)):
                            cause_My.append("ไม่สวมหมวก")
                        
                        #000 #00E
                        print(cause_My)
                        #000 #เช็คว่ามีผิดกฎไหม
                        if (len(cause_My)!=0):
                            # *******ใส่rang คำตอบ
                            
                            #เช็คว่ามีทะเบียนที่อยู่ในบริเวณใกล้เคียงไหม
                            for i in range(len(pred_LC[0])):
                                if(xyxy[0]-50<=pred_LC[0][i][0]<=xyxy[2]+50 and xyxy[1]<pred_LC[0][i][1]<=xyxy[3]):
                                    res_LC.append(pred_LC[0][i])
                            # print(res_LC)
                            # หาภาพ LC
                            if (len(res_LC)!=0):
                            # print(res_LC[0])    # tensor([684.00000, 701.00000, 774.00000, 797.00000,   0.91613,   0.00000])
                            # print(res_LC[0][:4])    # tensor([684., 701., 774., 797.])
                                # print("TEST_EDITxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                                # print(save_dir)
                                kkkkk+=1
                                im_LC=My0_save_one_box(res_LC[0][:4], imc, file=save_dir / 'crops' / "MyTEST" / f'{p.stem}.png', BGR=True,save=False)
                                
                                # im_LC=UPSx8(im_LC)
                                
                                part_LC="./yolov5/FOR_NEW_TEST/LC_"+str(kkkkk)+".png"
                                cv2.imwrite(part_LC, im_LC)
                                # Sub_LC.licensepic_path=part_LC
                                All_LC.append(part_LC)
                                
                                # im_LC=UPSx8(im_LC)
                                
                                # vehicle, LpImg,cor = My_get_plate(im_LC,wpod_net)
                                
                            #non OCR-------------------------------
                            
                            
                #-------------------------------------------------------
                    # print(conf)
                    # print(cls)
                    # -----res
                    # tensor([[370.92435, 173.20700, 444.63321, 255.16812,   0.79673,   3.00000]])
                    
                    # [tensor(1113.), tensor(484.), tensor(1334.), tensor(730.)]
                    # tensor(0.79673)
                    # tensor(3.)
                    
                    #000
                    # print(torch.tensor(xyxy))
                    # print("test")
                    # xyxy[1]=250
                    # print(xyxy[3])
                    
                    # print("KUYYYYYYY")
                    # print(RG)
                    
                    
                    # print(torch.tensor(xyxy))
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        #000 E
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.png', BGR=True)
                        
                        #000
                        # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}+{str(kkkkk)}.jpg', BGR=True)
                        # kkkkk+=1
            
            #000
            # print(f"\n{str(kkkkk)}\n")
            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            #000
            # print(save_path)
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
            
            
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        
    #000 #00E #copy OCR--------------------------------------------------------------
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(len(All_LC))
    
    # for LC_Part in All_LC:
    #     print("In loop "+str(LC_Part))
    #     im_LC = cv2.imread(LC_Part)
    #     vehicle, LpImg,cor = My_get_plate(im_LC,wpod_net)
    fileTxT_LC = open("./a0_Num_LC.txt", "a")
    fileTxT_LC.write("9")
    fileTxT_LC.close()
    
    fileTxT_LC = open("./a0_Num_LC.txt", "r")
    print(fileTxT_LC.read())
    #000 #00E #ลบ
    # open("./a0_Num_LC.txt", 'w').close()
    # fileTxT_LC = open("./a0_Num_LC.txt", "r")
    # print(fileTxT_LC.read())
    
    print("---END---")


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')      ##00 #01
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')   #00
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')   #00
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    #000 skip frame 4 เซ็ตคำสั่ง
    parser.add_argument('--skipframe', type=int, default=0, help='skipframe (int)')
    #000 legion detect 2 เซ็ตคำสั่ง
    parser.add_argument('--RG', type=int,nargs='+' , default=[0,0], help='legion detect (int x) (int y)')
    # parser.add_argument('--crop', action='store_true',help='crop the border box')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
