from PIL import Image, ImageDraw, ImageFont

def add_data_image(license_num: str, over_person: bool, not_wear_helmet: bool):
    img = Image.open("./static/pdf_source/TrafficTicket.png")
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("./static/pdf_source/TH Krub.ttf", 50)
    d.chord(xy=[(700,1180), (900,1261)], start=0, end=70, fill=(255,0,0), width=1)
    d.text(xy=(1250,1635), text=license_num, font=fnt, fill=(255,0,0))
    d.text(xy=(1250,1690), text=license_num, font=fnt, fill=(255,0,0))
    # Condition
    if over_person and not_wear_helmet:
        d.text(xy=(740,2450), text="นั่งซ้อนเกิน 2 คน และไม่สวมใส่หมวกนิรภัย", font=fnt, fill=(255,0,0))
    elif not_wear_helmet:
        d.text(xy=(740,2450), text="ไม่สวมใส่หมวกนิรภัย", font=fnt, fill=(255,0,0))
    elif over_person:
        d.text(xy=(740,2450), text="นั่งซ้อนเกิน 2 คน", font=fnt, fill=(255,0,0))
    else:
        d.text(xy=(740,2450), text="-", font=fnt, fill=(255,0,0))
    img.save("./static/pdf_source/PDFaddtext.png")
    # print("adddataimage")
    return 

def img2pdf():
    image_1 = Image.open("./static/pdf_source/PDFaddtext.png")
    im_1 = image_1.convert('RGB')
    im_1.save("./static/create_pdf/TrafficTicket.pdf")
    # print("img2pdf")
    return  "./static/create_pdf/TrafficTicket.pdf"
    
# add_data_image("8กฮ1912", False, True)
# img2pdf()