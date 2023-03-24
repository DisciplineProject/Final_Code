import torch
import os
import cv2
import json
import time
# from a0_MyFuction import UPSx8,grayscale

from roboflow import Roboflow
rf = Roboflow(api_key="s01WNoZ7wk7AhtO70ayC")
project = rf.workspace().project("lru-license-plate")
model = project.version(1).model

#--------------------------------------------------
# project = rf.workspace().project("test-thailand-vehicle-plate")
# model = project.version(2).model

# infer on a local image
# print(model.predict("your_image.jpg", confidence=40, overlap=30).json())
#--------------------------------------------------

All_Char=['ก','ข','ฃ','ค','ฅ','ฆ','ง','จ'
          ,'ฉ','ช','ซ','ฌ','ญ','ฎ','ฏ','ฐ','ฑ','ฒ','ณ','ด','ต','ถ','ท','ธ','น'
          ,'บ','ป','ผ','ฝ','พ','ฟ','ภ','ม','ย','ร','ล','ว','ศ','ษ','ส','ห','ฬ'
          ,'อ','ฮ']

Map_dic={
    "A1" :"ก", 
    "A10" :"ช", 
    "A11" :"ซ", 
    "A12" :"ฌ", 
    "A13" :"ญ", 
    "A14" :"ฎ", 
    "A15" :"ฎ", 
    "A16" :"ฐ", 
    "A18" :"ฒ", 
    "A19" :"ณ", 
    "A2" :"ข", 
    "A20" :"ด", 
    "A21" :"ต", 
    "A22" :"ถ", 
    "A23" :"ท", 
    "A24" :"ธ", 
    "A25" :"น", 
    "A26" :"บ", 
    "A28" :"ผ", 
    "A3" :"ข", 
    "A30" :"พ", 
    "A32" :"ภ", 
    "A33" :"ม", 
    "A34" :"ย", 
    "A35" :"ร", 
    "A36" :"ล", 
    "A37" :"ว", 
    "A38" :"ศ", 
    "A39" :"ษ", 
    "A4" :"ค", 
    "A40" :"ส", 
    "A41" :"ห", 
    "A42" :"ฬ", 
    "A43" :"อ", 
    "A6" :"ฆ", 
    "A7" :"ง", 
    "A8" :"จ", 
    "A9" :"ฉ", 
    "ANC" :"", 
    "BK." :"", 
    "BKK" :"", 
    "BRR" :"", 
    "CHAN" :"", 
    "CHM" :"", 
    "CHON" :"", 
    "CHP" :"", 
    "CHR" :"", 
    "CYP" :"", 
    "KAL" :"", 
    "KHON" :"", 
    "KOR" :"", 
    "KPP" :"", 
    "LAMP" :"", 
    "LB" :"", 
    "LP" :"", 
    "MHK" :"", 
    "MHS" :"", 
    "N0" :"0", 
    "N1" :"1", 
    "N13" :"1", 
    "N2" :"2", 
    "N23" :"2", 
    "N3" :"3", 
    "N4" :"4", 
    "N5" :"5", 
    "N6" :"6", 
    "N7" :"7", 
    "N8" :"8", 
    "N9" :"9", 
    "NPT" :"", 
    "NSA" :"", 
    "PY" :"", 
    "RE" :"", 
    "SRTN" :"", 
    "SSK" :"", 
    "SUPB" :"", 
    "SUR" :"", 
    "TAK" :"", 
    "UBON" :"", 
    "UDON" :"", 
    "UTT" :"", 
    "YST" :"", 
    "a40" :"ส", 
    "a9." :"ฉ", 
    "na" : "3"
}

class sub_OCR():
    len_LC: int
    license_num: str        #OCR - ไม่พบทะเบียน
    licensepic1_path: str       #OCR



def read_char (filename="AutoPutName",len_LC=177,TestOCR=False):
    All_LC=[]
    for i in range(len_LC):
        print(i+1)
        #00E
        LC_part="./static/pic/LC_"+filename+str(i+1)+".png"
        # LC_part="./static/pic/LC_"+str(i+1)+".jpg"
        if TestOCR:
            LC_part="./detect_part/0_Data/LicensePlate/Test_LC_1Min"+str(i+1)+".png"
        CUR_LC=sub_OCR()
        CUR_LC.len_LC=i
        CUR_LC.license_num="ไม่พบตัวอักษร"
        CUR_LC.licensepic1_path="ตรวจไม่พบ"
        #000
        # part_x8="./static/pic/X8.png"
        # img_x8=cv2.imread(LC_part)
        # img_x8=UPSx8(img_x8)
        # part_x8="./static/pic/X8.png"
        # cv2.imwrite(part_x8, img_x8)
        # res = model.predict(part_x8, confidence=80, overlap=30).json()["predictions"] #get prediction
        
        res = model.predict(LC_part, confidence=30, overlap=30).json()["predictions"] #get prediction
        if len(res)==0:
            All_LC.append(CUR_LC)
            continue
        # print(res)
        res.sort(key=lambda x: x["y"])
        # print(res)
        Sord_Y=[]
        x_old,y_old=0,0
        # counter_sord=1
        for _ in range(len(res)):
            y_cur=res[_]["y"]
            if ((y_cur - y_old )>= -0.4 and (y_cur - y_old) <= 0.4 ):
            #     if(res[_-1]["confidence"]<res[_]["confidence"]):
            #         Sord_Y[len(Sord_Y)-1]=res[_]
                y_old=y_cur
                continue
            else:
                # counter_sord+=1
                Sord_Y.append(res[_])
            y_old=y_cur
        # print("xxx")
        # print(Sord_Y)
        LC_PLAT=""
        SUB_DIC={
            "x":int(),
            "y":int(),
            "char":str(),
            "confidence":float()
        }
        SUB_LIT=[]
        # chack_dupicate=0
        x_old,y_old=0,0
        #dd
        for _ in range(len(Sord_Y)):
            # z=Map_dic[res[_]["class"]]
            x_cur,y_cur=Sord_Y[_]["x"],Sord_Y[_]["y"]
            # print([Sord_Y[_]["x"],Sord_Y[_]["y"],Sord_Y[_]["confidence"],Map_dic[Sord_Y[_]["class"]]])
                
            if ((x_cur - x_old )>= -0.4 and (x_cur - x_old) <= 0.4 ) or ((y_cur - y_old) >= -0.4 and (y_cur - y_old) <= 0.4):
            # if ((y_cur - y_old) >= -0.4 and (y_cur - y_old) <= 0.4):
            
                # if(SUB_LIT[len(SUB_LIT)-1]["confidence"]<Sord_Y[_]["confidence"]):
                #     SUB_DIC={"x":x_cur,"y":y_cur,"char":Map_dic[Sord_Y[_]["class"]],"confidence":Sord_Y[_]["confidence"]}
                #     SUB_LIT[len(SUB_LIT)-1]=SUB_DIC
                x_old,y_old=x_cur,y_cur
                continue
            else:
                # print("Not Depupicate")
                SUB_DIC={"x":x_cur,"y":y_cur,"char":Map_dic[Sord_Y[_]["class"]],"confidence":Sord_Y[_]["confidence"]}
                # SUB_DIC["x"]=x_cur
                # SUB_DIC["y"]=y_cur
                # SUB_DIC["char"]=Map_dic[res[_]["class"]]
                SUB_LIT.append(SUB_DIC)
                # chack_dupicate.append(SUB_DIC)
                
            x_old,y_old=x_cur,y_cur
        #
        # print("SUB_LIT")
        # print(SUB_LIT)
        line1=[]
        line2=[]
        chang_line=False
        x_old,y_old=SUB_LIT[0]["x"],SUB_LIT[0]["y"]
        for _ in range(len(SUB_LIT)):
            x_cur,y_cur=SUB_LIT[_]["x"],SUB_LIT[_]["y"]
            if ((y_old-y_cur) < -10) or chang_line:  
                chang_line=True
                line2.append(SUB_LIT[_])
            else:
                line1.append(SUB_LIT[_])
            x_old,y_old=x_cur,y_cur
        # print("Line")
        # print(line1)
        # print(line2)
        line1.sort(key=lambda x: x["x"])
        line2.sort(key=lambda x: x["x"])
        
        line1_2=[]
        line2_2=[]
        if len(line1)!=0:
            x_old=0
        for _ in range(len(line1)):
            x_cur=line1[_]["x"]
            if ((x_cur - x_old )>= -0.5 and (x_cur - x_old) <= 0.5 ):
                # if(line1[_-1]["confidence"]<line1[_]["confidence"]):
                #     line1_2.append(line1[_])
                x_old=x_cur
                continue
            else:
                line1_2.append(line1[_])
            x_old=x_cur
            
        if len(line2)!=0:
            x_old=0
            # print("xxx")
        for _ in range(len(line2)):
            # print(line2[_]["x"])
            x_cur=line2[_]["x"]
            if ((x_cur - x_old )>= -0.5 and (x_cur - x_old) <= 0.5 ):
                # if(line2[_-1]["confidence"]<line2[_]["confidence"]):
                #     line2_2.append(line2[_])
                x_old=x_cur
                continue
            else:
                line2_2.append(line2[_])
            x_old=x_cur
            
        RES_LC=""
        for _ in range(len(line1_2)):
            # if (_==0):
            #     RES_LC+=line1_2[_]["char"]
            # elif (_>0) and (line1_2[_]["char"] in All_Char):
            #     RES_LC+=line1_2[_]["char"]
            RES_LC+=line1_2[_]["char"]
        
        RES_LC+=" "
        for _ in range(len(line2_2)):
            RES_LC+=line2_2[_]["char"]
        
        if RES_LC == " " or RES_LC == "":
            RES_LC="ไม่พบตัวอักษร"
        CUR_LC.license_num=RES_LC
        All_LC.append(CUR_LC)
    
    # print(All_LC[0].license_num) ก5ธ1814
    # print(All_LC[1].license_num) 91
    if TestOCR:
            for _ in range(len(All_LC)):
                # print(All_LC[_].len_LC)
                # print(All_LC[_].len_LC+1)
                # print(All_LC[_].license_num)
                print([All_LC[_].len_LC+1+couter_pic,All_LC[_].license_num])
            
            return None
    else:        
        return All_LC
    
# read_char()
# V2
# //////////////////////////////////////////////////////
# AutoPut
# frame160.png
# ไม่พบตัวอักษร
# False
# True
# 0:03
# ./static/pic/AutoPutMoto_1.png
# ตรวจไม่พบ
# ไม่พบป้ายทะเบียน
# ไม่พบป้ายทะเบียน
# ไม่พบป้ายทะเบียน
# //////////////////////////////////////////////////////
# AutoPut
# frame686.png
# 1968
# False
# True
# 0:13
# ./static/pic/AutoPutMoto_2.png
# ตรวจไม่พบ
# ไม่พบป้ายทะเบียน
# ไม่พบป้ายทะเบียน
# ไม่พบป้ายทะเบียน
# //////////////////////////////////////////////////////
# AutoPut
# frame4929.png
# ไม่พบตัวอักษร
# False
# True
# 1:38
# ./static/pic/AutoPutMoto_3.png
# ตรวจไม่พบ
# ไม่พบป้ายทะเบียน
# ไม่พบป้ายทะเบียน
# ไม่พบป้ายทะเบียน