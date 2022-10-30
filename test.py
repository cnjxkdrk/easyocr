import easyocr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

# img maximum : 178956970 pixels

reader = easyocr.Reader(['ko','en'], gpu=False) # this needs to run only once to load the model into memory
font = ImageFont.truetype("fonts/malgun.ttf", size=25)

def run(img):   
    # result = reader.readtext(img)
    # matrix = get_matrix(result)
    # print(matrix)

    # img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = reader.readtext(img)
    matrix = get_matrix(result)
    print(matrix)

    img = Image.fromarray(img)
    drawRect(img, result)
    

def get_matrix(result):
    ret = np.array(result, dtype='object')
    return ret[:, 1:]


def drawRect(img, result):
    draw = ImageDraw.Draw(img)
    for i in result :
        x = i[0][0][0]
        y = i[0][0][1]
        w = i[0][1][0] - i[0][0][0]
        h = i[0][2][1] - i[0][1][1]

        draw.rectangle(((x, y), (x+w, y+h)), outline='blue', width=2)
        draw.text((x , y-h/2), str(i[1]), font=font, fill='blue')

    plt.imshow(img)
    plt.show()


def main():
    # img_path = './image/crop3.PNG'
    # img_path = './image/search.shopping.naver.com_search_all_query=B3ECB898%95&cat_id=&frm=NVSHATC.png'
    # img_path = './image/HighResSrc.png'
    # img_path = './image/2K.png'
    # img_path = './image/1080p.png'
    img_path = './image/2K_GOM.png'

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    CROP_HEIGHT = 1000
    pages = height//CROP_HEIGHT + 1
    for idx in range(pages):
        # final page
        if idx==pages-1:
            crop_img = img[idx*CROP_HEIGHT:]
        else:
            crop_img = img[idx*CROP_HEIGHT:(idx+1)*CROP_HEIGHT]
        
        print(f'== Page{idx+1} ' + '='*100)
        run(crop_img)
        # plt.close()    

    # run(img_path)


if __name__ == '__main__':
    main()

