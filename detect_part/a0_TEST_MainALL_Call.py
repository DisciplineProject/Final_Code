from pydantic import BaseModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from .yolov5.detect_NextGen import yolov5, Data
# from .TestAdjust_LC_Main.a1_TEST_NextGen import runOCR
from .New_Ocr.New_OCR import read_char

# class Data(BaseModel):
#     filename: str           #---
#     framename: str          #---
#     license_num: str        #OCR        -Auto "ไม่พบทะเบียน"
#     over_person: bool       #---        -Auto False
#     not_wear_helmet: bool   #---        -Auto False
#     timestamp: str          #---
#     motorcyclepic_path: str     #---  
#     licensepic1_path: str       #OCR    -Auto "ไม่พบทะเบียน"
#     licensepic2_path: str       #OCR    -Auto "ไม่พบทะเบียน"
#     licensebw1_path: str       #OCR     -Auto "ไม่พบทะเบียน"
#     licensebw2_path: str       #OCR     -Auto "ไม่พบทะเบียน"
#     class Config:
#         orm_mode = True

class sub_OCR():
    len_LC: int
    license_num: str        #OCR -      Auto ""
    licensepic1_path: str       #OCR    Auto "ตรวจไม่พบ"
    licensepic2_path: str       #OCR    Auto "ตรวจไม่พบ"
    licensebw1_path: str       #OCR     Auto "ตรวจไม่พบ"
    licensebw2_path: str       #OCR     Auto "ตรวจไม่พบ"

# ***ไม่พบป้ายทะเบียน ขี้นเมื่อขั้น Part2 หาทะเบียนไม่เจอ
# ***ส่วน ตรวจไม่พบ ขึ้นตอน Part2 พบทะเบียนแต่ หาอักษรไม่เจอ

# ###------- CONTROL ROOM
# ##- STAGE OF PROGEAM
# play_yolo=True
# # play_yolo=False
# play_ocr=True
# # play_ocr=False
# ##- STAGE OF PROGEAM FOR TEST
# # TEST=True
# TEST=False
# # TEST_YOLO=True
# TEST_YOLO=False
# # TEST_OCR=True
# TEST_OCR=False
# TEST_ALL=True
# TEST_ALL=False

###-------Main Code
# ตัวอย่างการใช้งาน
# MainAll("./0_Data/TEST_VIDEO/0_Short.mp4","Auto_Put_NAME.mp4")

def MainAll(sourdvideo,filename="Auto_Put_NAME.mp4",TEST_ALL=False):
    # TEST_ALL=False
    TEST_OCR=False
    # os.system("python ./yolov5/detect.py --weights ./yolov5/0_Model/yolov5s.pt --source ./0_Data/Demo_Pic2 --classes 3")
    Main_DATA,num_LC = yolov5(source=sourdvideo, filename=filename)
    # os.system("python ./TestAdjust_LC_Main/a1_TEST_NextGen.py")
    # res_LC=runOCR(num_LC,filename)
    res_LC=read_char(filename=filename,len_LC=num_LC)
    # if TEST_OCR:
    #     print("----------xxxxxxxxxxxxxx--------------")
    #     print(len(res_LC))
    #     for cur_LC in range(len(res_LC)):
    #         print("------------------------------------------")
    #         print(res_LC[cur_LC].len_LC)
    #         print(res_LC[cur_LC].license_num)
    #         print(res_LC[cur_LC].licensepic1_path)
    #         # print(res_LC[cur_LC].licensepic2_path)
    #         # print(res_LC[cur_LC].licensebw1_path)
    #         # print(res_LC[cur_LC].licensebw2_path)
    # เอาข้อมูลมารวมกัน
    for i in range(len(res_LC)):
        # print(f"i = {i}")
        cur_LC=res_LC[i].len_LC
        Main_DATA[cur_LC].license_num = res_LC[i].license_num
        # Main_DATA[cur_LC].licensepic1_path = res_LC[i].licensepic1_path
        # Main_DATA[cur_LC].licensepic2_path = res_LC[i].licensepic2_path
        # Main_DATA[cur_LC].licensebw1_path = res_LC[i].licensebw1_path
        # Main_DATA[cur_LC].licensebw2_path = res_LC[i].licensebw2_path
    if TEST_ALL:
        for cur_MoTo in range(len(Main_DATA)):
            print("//////////////////////////////////////////////////////")
            print(Main_DATA[cur_MoTo].filename)
            print(Main_DATA[cur_MoTo].framename)
            print(Main_DATA[cur_MoTo].license_num)
            print(Main_DATA[cur_MoTo].over_person)
            print(Main_DATA[cur_MoTo].not_wear_helmet)
            print(Main_DATA[cur_MoTo].timestamp)
            print(Main_DATA[cur_MoTo].motorcyclepic_path)
            print(Main_DATA[cur_MoTo].licensepic1_path)
            # print(Main_DATA[cur_MoTo].licensepic2_path)
            # print(Main_DATA[cur_MoTo].licensebw1_path)
            # print(Main_DATA[cur_MoTo].licensebw2_path)
    return Main_DATA

# MainAll("./0_Data/TEST_Short_MoTo")

#เพื่อจะเอาไปใช้ TEST
# TEST_GET_NAME="./0_Data/TEST_VIDEO/0_Short.mp4"
# print(TEST_GET_NAME.split())
# sourdvideo="./0_Data/TEST_ALL_MAX"
# sourdvideo="./0_Data/motocycle11"
# sourdvideo="./0_Data/TEST_VIDEO/0_Short.mp4"
# sourdvideo="./0_Data/TEST_Short_MoTo"
    # if TEST and TEST_YOLO:
    #     for cur_MoTo in range(len(Main_DATA)):
    #         print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    #         print(Main_DATA[cur_MoTo].filename)
    #         print(Main_DATA[cur_MoTo].framename)
    #         # ไม่มีปัญหา
    #         # print(Main_DATA[cur_MoTo].license_num)
    #         print(Main_DATA[cur_MoTo].over_person)
    #         print(Main_DATA[cur_MoTo].not_wear_helmet)
    #         print(Main_DATA[cur_MoTo].timestamp)
    #         print(Main_DATA[cur_MoTo].motorcyclepic_path)
    #         # ไม่มีปัญหา
    #         # print(Main_DATA[cur_MoTo].licensebw1_path)
    #         # print(Main_DATA[cur_MoTo].licensebw2_path)
    #         # print(Main_DATA[cur_MoTo].licensepic1_path)
    #         # print(Main_DATA[cur_MoTo].licensepic2_path)

    # if TEST and TEST_OCR:
    #     for cur_LC in range(len(res_LC)):
    #         print("------------------------------------------")
    #         print(res_LC[cur_LC].len_LC)
    #         print(res_LC[cur_LC].license_num)
    #         print(res_LC[cur_LC].licensepic1_path)
    #         print(res_LC[cur_LC].licensepic2_path)
    #         print(res_LC[cur_LC].licensebw1_path)
    #         print(res_LC[cur_LC].licensebw2_path)

# if TEST_ALL:
#     for cur_MoTo in range(len(Main_DATA)):
#         print("//////////////////////////////////////////////////////")
#         print(Main_DATA[cur_MoTo].filename)
#         print(Main_DATA[cur_MoTo].framename)
#         print(Main_DATA[cur_MoTo].license_num)
#         print(Main_DATA[cur_MoTo].over_person)
#         print(Main_DATA[cur_MoTo].not_wear_helmet)
#         print(Main_DATA[cur_MoTo].timestamp)
#         print(Main_DATA[cur_MoTo].motorcyclepic_path)
#         print(Main_DATA[cur_MoTo].licensepic1_path)
#         print(Main_DATA[cur_MoTo].licensepic2_path)
#         print(Main_DATA[cur_MoTo].licensebw1_path)
#         print(Main_DATA[cur_MoTo].licensebw2_path)

