from fastapi import FastAPI, UploadFile, Request, Form, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
# from fastapi_pagination import Page, add_pagination, paginate
from typing import Union,List,Optional
from pydantic import BaseModel
import pymongo
from pymongo import MongoClient
import json
import os

from detect_part.yolov5.detect_NextGen import Data
from detect_part.a0_TEST_MainALL_Call import MainAll


from pdf.pdf import add_data_image, img2pdf

cluster = MongoClient("")
db = cluster["File"]
collection = db["Motorcyclist"]

app = FastAPI(debug=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# class Data(BaseModel):
#     filename: str
#     framename: str
#     license_num: str
#     over_person: bool
#     not_wear_helmet: bool
#     timestamp: str
#     motorcyclepic_path: str
#     licensepic1_path: str
#     licensepic2_path: str
#     licensebw1_path: str
#     licensebw2_path: str
#     class Config:
#         orm_mode = True

#Get path
@app.get("/", response_model=List[Data], name="home")
def read_root(request: Request, skip: int = 0, limit: int = 10):
    cursor = collection.find().skip(skip).limit(limit)
    alldata = collection.find()
    list_data = []
    list_alldata = []
    # print(alldata)
    for i in alldata:
        list_alldata.append(Data(**i))
    for i in cursor:
        list_data.append(Data(**i))
    total_data = len(list_alldata)
    len_data = len(list_data)
    if total_data % 10 == 0:
        total_page = total_data // 10
    else:
        total_page = (total_data // 10) + 1
    # print(total_data , total_page)

    current_page = int(skip / limit) + 1
    return templates.TemplateResponse("index.html", {"request": request, "list_data": list_data, "skip" : skip , "limit" : limit, "total_data": total_data, "total_page": total_page, "current_page": current_page, "len_data": len_data})

# @app.get("/getdata", response_model=List[Data])
# async def alldata():
#     alldata = collection.find()
#     list_data = []
#     for i in alldata:
#         list_data.append(Data(**i))
#     return list_data

@app.get("/getdata/filename/{filename}")
async def getbydatafile(filename : str):
    data = collection.find({"filename": filename})
    list_data = []
    for i in data:
        list_data.append(Data(**i))
    return list_data

@app.get("/getdata/framename/{framename}")
async def getframedata(request: Request, framename : str):
    data = collection.find({"framename": framename})
    for i in data:
        filename = Data(**i).filename
        license_num = Data(**i).license_num
        over_person = Data(**i).over_person
        not_wear_helmet = Data(**i).not_wear_helmet
        timestamp = Data(**i).timestamp
        motorcyclepic_path = Data(**i).motorcyclepic_path
        licensepic1_path = Data(**i).licensepic1_path
        licensepic2_path = Data(**i).licensepic2_path
        licensebw1_path = Data(**i).licensebw1_path
        licensebw2_path = Data(**i).licensebw2_path
        edit_status = Data(**i).edit_status
        # print(licensepic1_path)
    return templates.TemplateResponse("show.html", {"request": request, "framename": framename, "filename": filename, "license_num": license_num,"over_person": over_person,"not_wear_helmet": not_wear_helmet, "timestamp": timestamp, "motorcyclepic_path": motorcyclepic_path, "licensepic1_path": licensepic1_path, "licensepic2_path": licensepic2_path, "licensebw1_path": licensebw1_path, "licensebw2_path": licensebw2_path, "edit_status": edit_status})
    # return templates.TemplateResponse("show.html", {"request": request, "framename": framename, "filename": filename, "license_num": license_num,"over_person": over_person,"not_wear_helmet": not_wear_helmet, "timestamp": timestamp, "motorcyclepic_path": motorcyclepic_path, "licensepic1_path": licensepic1_path})

@app.get("/edit/{framename}")
async def getdata(request: Request, framename : str):
    data = collection.find({"framename": framename})
    for i in data:
        k = Data(**i)
    return templates.TemplateResponse("edit.html", {"request": request, "data": k})

@app.get("/download/{framename}")
async def download(framename: str):
    data = collection.find({"framename": framename})
    for i in data:
        license_num = Data(**i).license_num
        # print(f"{license_num}")
        over_person =  Data(**i).over_person
        # print(f"{over_person}")
        not_wear_helmet = Data(**i).not_wear_helmet
        # print(f"{not_wear_helmet}")
    add_data_image(license_num, over_person, not_wear_helmet)
    # print("complete_adddataimage")
    file_path = img2pdf()
    # print("complete_img2pdf")
    return  FileResponse(path=file_path, filename="TrafficTicket.pdf") 

#Post path
@app.post("/upload", response_class=RedirectResponse)
async def upload(file : UploadFile):
    # print(f"filename = {file.filename}.")
    if file.filename == "":
        return RedirectResponse(url=app.url_path_for("home"), status_code=status.HTTP_303_SEE_OTHER)
    path = "source/" + file.filename  
    print(f"source/{path}")
    await file.seek(0)
    with open("./static/" + path, "wb+") as file_object:
        file_object.write(file.file.read())
    
    # mainfunc here !!!!
    final_path = "./static/" + path
    # print(f"filename = {file.filename}")
    data = MainAll(final_path, file.filename)
    # print(f"Final DATA = {data1}")
#     #test data
# async def upload():
    # data = []
    # data1 = Data(filename = "SAdad",
    #     framename = "frame20",
    #     license_num = "กเ655",
    #     over_person = True,
    #     not_wear_helmet = False,
    #     timestamp = "6:34",
    #     motorcyclepic_path = "sadfaf/fdsf",
    #     licensepic1_path = "/dsf/",
    #     licensepic2_path = "/dfs",
    #     licensebw1_path = "dfskj/",
    #     licensebw2_path = "467/dfs/")
#     data2 = Data(filename = "SAdad",
#         framename = "frame21",
#         license_num = "กเ655",
#         over_person = False,
#         not_wear_helmet = False,
#         timestamp = "6:34",
#         motorcyclepic_path = "sadfaf/fdsf",
#         licensepic1_path = "/dsf/",
#         licensepic2_path = "/dfs",
#         licensebw1_path = "dfskj/",
#         licensebw2_path = "467/dfs/")
    # data.append(data1)
#     data.append(data2)
    
    for i in range(len(data)):
        # if data[i].over_person == True or data[i].not_wear_helmet == True:  
            data[i].motorcyclepic_path = data[i].motorcyclepic_path.lstrip(".")
            data[i].licensepic1_path = data[i].licensepic1_path.lstrip(".")
            data[i].licensepic2_path = data[i].licensepic2_path.lstrip(".")
            data[i].licensebw1_path = data[i].licensebw1_path.lstrip(".")
            data[i].licensebw2_path = data[i].licensebw2_path.lstrip(".")
            data[i].framename = data[i].filename + "_" + data[i].framename
            # print(type(data))
            # print(type(data[i]))
            # print(dict(data[i]))
            _id = collection.insert_one(dict(data[i]))
            # print(data[i].motorcyclepic_path)
            # print(data[i].licensepic1_path)
            # print(data[i].licensepic2_path)
            # print(data[i].licensebw1_path)
            # print(data[i].licensebw2_path)
        # _id = collection.insert_one(dict(data[i])) 
    delete_path = "./static/source/" + file.filename
    os.remove(delete_path)
    # print("Finish post data")
    # return {"Finish post data"}
    return RedirectResponse(url=app.url_path_for("home"), status_code=status.HTTP_303_SEE_OTHER)

@app.post("/edit/{framename}", response_description="Edit Data", response_class=RedirectResponse)
async def edit(request: Request, framename : str, license_num: str = Form(), over_person: str = Form(), not_wear_helmet: str = Form()):
    # print(f"{over_person}/{not_wear_helmet}")
    if over_person == "true":
        over_person = True
    else:
        over_person = False
    if not_wear_helmet == "true":
        not_wear_helmet = True
    else:
        not_wear_helmet = False
    # print(f"bool = {over_person}/{not_wear_helmet}")
    myquery = {"framename": framename}
    newvalue = {"$set": {"license_num": license_num, "over_person" : over_person, "not_wear_helmet": not_wear_helmet, "edit_status": True}}
    
    _id = collection.update_one(myquery, newvalue)

    return RedirectResponse(url=app.url_path_for("home"), status_code=status.HTTP_303_SEE_OTHER)