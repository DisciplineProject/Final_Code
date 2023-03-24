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
from typing import List
from pydantic import BaseModel

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
from utils.plots import Annotator, colors, save_one_box ,My0_save_one_box ,My1Moto_save_one_box
# from utils.plots import Annotator, colors, save_one_box , crop_plot
from utils.torch_utils import select_device, smart_inference_mode
# #000 #00E #copy----------------------------------
# All_LC=[]   #ส่ง่ part LC
# All_LC=0
# res_DATA=[]

class Data(BaseModel):
    filename: str           #---
    framename: str          #---
    license_num: str        #OCR - ไม่พบทะเบียน
    over_person: bool       #---
    not_wear_helmet: bool   #---
    timestamp: str          #---
    motorcyclepic_path: str     #OCR
    # motorcyclepic_name: str   
    licensepic1_path: str       #OCR
    # licensepic1_name: str
    licensepic2_path: str       #OCR
    # licensepic2_name: str
    licensebw1_path: str       #OCR
    # licensebw1_name: str
    licensebw2_path: str       #OCR
    # licensebw2_name: str
    edit_status: bool
    class Config:
        orm_mode = True
#------------------------------------


# @smart_inference_mode()
# def run(
# def xxx():
#     return True
def yolov5(
        weights="./yolov5/0_Model/yolov5s.pt",  # model path or triton URL
        source="./0_Data/Demo_Pic2",  # file/dir/URL/glob/screen/0 (webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=3,  # filter by class: --class 0, or --class 0 2 3
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
        num_frame_skip=0,  #000
        #000 legion detect 0 รับค่าตัวแปล
        RG=True,
        #000
        kkkkk=0,
        filename="Auto_Put_FILE_NAME.mp4",
        fpsss=30
):
    #000
    All_LC=0    # เก็บจำนวนรูปภาพ
    res_DATA=[]    # เก็บข้อมูล class Data แบบ list
    gg=0
    # counter_MoTo=0
    # Stable_Det=0
    my_TEST=True
    
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
    
    #000
    # print("Find ลำดับ")
    # print(imgsz)    #s  #imgsz [640, 640] #
    
    #000 #00E #copy------------------------------------
    MyEdit=True
    # MyEdit=False
    # USE
    if MyEdit:
        #LC
        # print("Load-model m.LC")
        weights_LC = ['./detect_part/yolov5/0_Model/license.pt']
        model_LC = DetectMultiBackend(weights_LC, device=device, dnn=dnn, data=data, fp16=half)
        stride_LC, names_LC, pt_LC = model_LC.stride, model_LC.names, model_LC.pt
        imgsz_LC = check_img_size(imgsz, s=stride_LC)  # check image size
        
        #PS
        # print("Load-model m.PS")            # class 0 yolov5
        weights_PS = ['./detect_part/yolov5/0_Model/yolov5s.pt']
        model_PS = DetectMultiBackend(weights_PS, device=device, dnn=dnn, data=data, fp16=half)
        stride_PS, names_PS, pt_PS = model_PS.stride, model_PS.names, model_PS.pt
        imgsz_PS = check_img_size(imgsz, s=stride_PS)  # check image size
        
        #HM
        # print("Load-model m.HM")            # class 0 yolov5
        weights_HM = ['./detect_part/yolov5/0_Model/helmet_last.pt']
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
        # print("WarmUp m.LC")
        model_LC.warmup(imgsz=(1 if pt_LC or model_LC.triton else bs, 3, *imgsz_LC))  # warmup
        dt_LC = (Profile(), Profile(), Profile())
        
        #PS
        # print("WarmUp m.PS")
        model_PS.warmup(imgsz=(1 if pt_PS or model_PS.triton else bs, 3, *imgsz_PS))  # warmup
        dt_PS = (Profile(), Profile(), Profile())
        
        #HM
        # print("WarmUp m.HM")
        model_HM.warmup(imgsz=(1 if pt_HM or model_HM.triton else bs, 3, *imgsz_HM))  # warmup
        dt_HM = (Profile(), Profile(), Profile())
    # # ------------------------------------
    # print("\n Zzzzzzzasd ")
    # print(len(dataset))
    # print(weights)
    # ขั้นข้างล่างน่าจะเป็นขั้น predictions ของทุก frame
    #000
    # print("Find ลำดับ xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # print(dataset.frame)    #s  #imgsz [640, 640] #
    # fpsss=dataset.cap.get(cv2.CAP_PROP_FPS)
    # print(fpsss)    # fps frame rate
    # print(bs)   #all frame
    # print(dataset.vid_stride)
    # print(dataset.frames)   # video ถึง print ได้
    #000 #set Data
    
    SkipFrame_bool=True # เปิด
    # SkipFrame_bool=False
    SF=0
    num_frame_skip=0
    # print(dataset.mode)
    # if dataset.mode=="video":
    #     # print("sdfsdfsdf")
    #     # print("FGFGFGFGFGs")
    #     fpsss=dataset.cap.get(cv2.CAP_PROP_FPS)
    #     num_frame_skip=fpsss*2
        # print(fpsss)
        
    
    for path, im, im0s, vid_cap, s in dataset:
        #000
        if dataset.mode=="video":
            fpsss=dataset.cap.get(cv2.CAP_PROP_FPS)
            num_frame_skip=fpsss*2
        # print("Find ลำดับ 2")
        # # print(dataset.count)  #cur frame
        # print(dataset.mode)       #000 video or image 
            #000 fpsss
        # if dataset.count != 1:
        # if (dataset.frame==6000 and my_TEST):
        #     break
        
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

            #000
            # print("Find ลำดับ")
            # print(s)
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
            # print("--------SF")
            # if(SF<0):
            # print("WTF")
            # print(SF)
            # print(num_frame_skip)
            if ((SF>0) and SkipFrame_bool):
                s += "skip frame #000 "
                SF-=1
                continue
            # print(s)  ล่างสุดของ for loop เพิ่ม ไม่ดีเทคกะเวลา Ex.image 14/14 D:\0_project\0_Git-Project\Demo\Demo_Pic2\frame12042.jpg: 384x640 1 Motorcycle, 2 Motorcycle+Drivers,
            # print(len(det))   จำนวนทุกคราสที่ค้นพบ
            # print(det)
            
            #000 ---- เจอก็ทำด้านล่าง
            
            
            
            # น่าจะถ้าเจอก็ทำด้านล่าง
            if len(det):
                #000
                    
                #000 skip frame 3 ใช้เพิ่มจำนวนที่จะ skip
                #000 #00E #copy------------------------------------
                if MyEdit:
                    #LC-xxxxxxxxxxxxxxxxxxxxxxxx
                    # print("dt_LC m.LC")
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
                        pred_LC = non_max_suppression(pred_LC, 0.6, iou_thres, 0, agnostic_nms, max_det=max_det)
            
                    # print(pred_LC)
            
                    #PS-xxxxxxxxxxxxxxxxxxxxxxxx
                    # print("dt_PS m.PS")
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
                        pred_PS = non_max_suppression(pred_PS, 0.34, iou_thres, 0, agnostic_nms, max_det=max_det)
                    
                    # print(pred_PS)
                    # print(len(pred_PS[0]))
                    # if (len(pred_PS[0]))>2:
                    #     print("false - PS")
                    # else:
                    #     print("true - PS")
                    
                    #HM-xxxxxxxxxxxxxxxxxxxxxxxx
                    # print("dt_HM m.HM")
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
                        pred_HM = non_max_suppression(pred_HM, 0.55, iou_thres, 0, agnostic_nms, max_det=max_det)
                    
                    
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
                    #000
                    # print("\nxyxy")
                    print(xyxy)     # [tensor(1382.), tensor(466.), tensor(1903.), tensor(896.)]
                    
                    # print(im0.shape)    (1080, 1920, 3)
                    # print(im0[1])   # ค่าไรไม่รู้
                    SIZE_OF_IMG_XYXY=im0.shape  # (1080, 1920, 3) (y x class)
                    # 1628
                    # print(SIZE_OF_IMG_XYXY)
                    # print(SIZE_OF_IMG_XYXY[0]//4)
                    #C frame6359    xyxy = [tensor(1341.), tensor(484.), tensor(1721.), tensor(829.)]
                    #               SIZE_OF_IMG_XYXY[1]//4 = 480 ****
                    #               (SIZE_OF_IMG_XYXY[1]//4)<xyxy[1]
                    # print(x[1])
                    # print(im.shape) # (3, 384, 640)
                    # print(gn)         # tensor([1920, 1080, 1920, 1080])
                    # 000 legion detect 1
                    
                    if (xyxy[1]<(SIZE_OF_IMG_XYXY[0]//4)) and RG: # y1>120 ติดรถขวา
                        # print("yes")
                        s += "Dont in Legion Y #000 "
                        continue
                    # print(xyxy)
                    # print(SIZE_OF_IMG_XYXY[1]-300)
                    #000 ***or  (xyxy[0]<300)
                    ArarX=500
                    if (xyxy[2]>(SIZE_OF_IMG_XYXY[1]-ArarX)) and RG: # 
                        # print("yes")
                        s += "Dont in Legion X1 #000 "
                        continue
                    if (xyxy[0]<ArarX) and RG: # 
                        # print("yes")
                        s += "Dont in Legion X2 #000 "
                        continue
                    # print("WTFFFFFFFFFF")
                    #000 #00E #copy-----------------------------
                    if MyEdit:
                        # print(len(pred_LC[0]))
                        if len(pred_LC[0])!=0 and SF<=0:
                            SF+=num_frame_skip
                        # print("xyxy - myEdit")
                        res_PS, res_HM, res_LC=[],[],[]
                        # SF+=num_frame_skip
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
                        
                        #000 SET DATA
                        if dataset.mode=="video":
                            CUR_F=dataset.frame
                            tmp_SEC=int(dataset.frame//fpsss)
                        else:
                            CUR_F=1
                            tmp_SEC=0
                        Sub_DATA=Data(filename=filename,
                                      framename="frame"+str(CUR_F) + ".png",
                                      license_num="ไม่พบป้ายทะเบียน",
                                      over_person=False,
                                      not_wear_helmet=False,
                                      timestamp = "0",
                                      motorcyclepic_path = ".",
                                      licensepic1_path="ไม่พบป้ายทะเบียน",
                                      licensepic2_path="ไม่พบป้ายทะเบียน",
                                      licensebw1_path="ไม่พบป้ายทะเบียน",
                                      licensebw2_path="ไม่พบป้ายทะเบียน",
                                      edit_status=False)
                        # Sub_DATA.Data()
                        # Sub_DATA.filename=filename
                        # Sub_DATA.framename="frame"+str(dataset.count) + ".png"
                        # Sub_DATA.license_num="ไม่พบป้ายทะเบียน"
                        # Sub_DATA.over_person=False
                        # Sub_DATA.not_wear_helmet=False
                        # Sub_DATA.licensepic1_path="ไม่พบป้ายทะเบียน"
                        # Sub_DATA.licensepic2_path="ไม่พบป้ายทะเบียน"
                        # Sub_DATA.licensebw1_path="ไม่พบป้ายทะเบียน"
                        # Sub_DATA.licensebw2_path="ไม่พบป้ายทะเบียน"

                        # tmp_SEC = dataset.count//fpsss
                        # print("----------------dataset.count")
                        # print(dataset.count)
                        if tmp_SEC>=3600:
                            Sub_DATA.timestamp=str(tmp_SEC//3600)+":"+str((tmp_SEC//60) - (60*(tmp_SEC//3600)))+":"+str(tmp_SEC%60)
                        elif tmp_SEC>=60:
                            second = tmp_SEC%60
                            if second < 10:
                                Sub_DATA.timestamp=str(tmp_SEC//60)+":0"+str(second)
                            else:
                                Sub_DATA.timestamp=str(tmp_SEC//60)+":"+str(second)
                        else:
                            second = tmp_SEC%60
                            if second < 10:
                                Sub_DATA.timestamp="0:0" + str(second)
                            else:
                                Sub_DATA.timestamp="0:" + str(second)
                                
                        cause_My = []
                        # print(len(res_PS))
                        if (len(res_PS)>2):
                            cause_My.append("ซ้อนเกิน")
                            Sub_DATA.over_person=True
                        if (len(res_PS)>len(res_HM)):
                            cause_My.append("ไม่สวมหมวก")
                            Sub_DATA.not_wear_helmet=True
                                
                        # cause_My = []
                        # if (len(res_PS)>2):
                        #     cause_My.append("ซ้อนเกิน")
                        #     Sub_DATA.over_person=True
                        # if (len(res_PS)>len(res_HM)):
                        #     cause_My.append("ไม่สวมหมวก")
                        #     Sub_DATA.not_wear_helmet=True
                        
                        # im_MoTo_2=My1Moto_save_one_box(xyxy, imc, file=save_dir / 'crops' / "MyTEST2" / f'{p.stem}.png', BGR=True,save=False)
                        # part_Moto="./static/pic/Moto_FORTEST_" + filename +str(kkkkk)+".png"
                        # cv2.imwrite(part_Moto, im_MoTo_2)
                        #000 #00E
                        # print(cause_My)
                        #000 #เช็คว่ามีผิดกฎไหม
                        if (len(cause_My)!=0):
                            # *******ใส่rang คำตอบ
                            
                            #เช็คว่ามีทะเบียนที่อยู่ในบริเวณใกล้เคียงไหม
                            for i in range(len(pred_LC[0])):
                                if(xyxy[0]-50<=pred_LC[0][i][0]<=xyxy[2]+50 and xyxy[1]<pred_LC[0][i][1]<=xyxy[3]):
                                    res_LC.append(pred_LC[0][i])
                            # print("xyxyxyxyxyxyxyxyxyxyxyxyxy")
                            # print(res_LC)
                            # หาภาพ LC
                            # if SF<=0:
                            #     SF+=num_frame_skip
                                    # counter_MoTo=0
                            if (len(res_LC)!=0):
                                # print(res_LC[0])    # tensor([684.00000, 701.00000, 774.00000, 797.00000,   0.91613,   0.00000])
                                # print(res_LC[0][:4])    # tensor([684., 701., 774., 797.])
                                    # print("TEST_EDITxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                                    # print(save_dir)
                                    
                                    #000 ตัดภาพ
                                for *lcxy,_,_ in res_LC:
                                    # print(xyzwww)
                                        # im_LC=My0_save_one_box(res_LC[0][:4], imc, file=save_dir / 'crops' / "MyTEST" / f'{p.stem}.png', BGR=True,save=False)
                                    im_LC=My0_save_one_box(lcxy, imc, file=save_dir / 'crops' / "MyTEST" / f'{p.stem}.png', BGR=True,save=False)
                                im_MoTo_2=My1Moto_save_one_box(xyxy, imc, file=save_dir / 'crops' / "MyTEST2" / f'{p.stem}.png', BGR=True,save=False)
                                # im_LC=UPSx8(im_LC)
                                        
                                # Save ภาพ
                                kkkkk+=1
                                part_LC="./static/pic/LC_"+filename+str(kkkkk)+".png"
                                # part_LC="./static/pic/LC_"+str(kkkkk)+".jpg"
                                cv2.imwrite(part_LC, im_LC)
                                Sub_DATA.licensepic1_path=part_LC
                                
                                part_Moto="./static/pic/" + filename + "Moto_"+str(kkkkk)+".png"
                                cv2.imwrite(part_Moto, im_MoTo_2)
                                # เอาข้อมูลลง Data
                                Sub_DATA.motorcyclepic_path=part_Moto
                                    # Sub_LC.licensepic_path=part_LC
                                    # All_LC.append(part_LC)
                                    # All_LC+=1
                                    
                                    # im_LC=UPSx8(im_LC)
                                    # res_DATA.append(Sub_DATA)
                                All_LC+=1
                                res_DATA.append(Sub_DATA)
                                    # counter_MoTo=0
                                    # vehicle, LpImg,cor = My_get_plate(im_LC,wpod_net)
                            else:
                                gg+=1
                                #non OCR-------------------------------
                            
                            
                #-------------------------------------------------------
            
            
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        
    #000 #00E #copy OCR--------------------------------------------------------------
    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(All_LC)
    # print(gg)
    print("---END---")
    return res_DATA ,All_LC
    # return res_DATA ,0