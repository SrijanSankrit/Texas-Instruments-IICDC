import cv2
from math import sin, cos, radians

camera =  cv2.VideoCapture(0)
face = cv2.CascadeClassifier("Haar_Cascades/haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier('Haar_Cascades/cascade_10_8000.xml')

settings = {
    'scaleFactor': 1.3,
    'minNeighbors': 3,
    'minSize': (50, 50),
    # 'flags': cv2.CV_HAAR_FIND_BIGGEST_OBJECT|cv2.CV_HAAR_DO_ROUGH_SEARCH
}

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.8)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

def rotate_image_back(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.25)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1]*0.4
    y = pos[1] - img.shape[0]*0.4
    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + img.shape[1]*0.4
    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + img.shape[0]*0.4
    return int(newx), int(newy), pos[2], pos[3]

xbeg=0.1
xend=0.9
ybeg=0.25
yend=0.55

count_frames=0
count_eyes=0

initial_count=0

def func(img):

    a=1.2
    b=15

    count_eyes=0

    for angle in [0, -25, 25]:
        rimg = rotate_image(img, angle)
        detected = face.detectMultiScale(rimg, **settings)
        for x, y, w, h in detected[-1:]:

            cv2.rectangle(rimg, (x, y), (x+w, y+h), (255,0,0), 2)

            if angle==0:
                a=1.2
                b=15

            if angle!=0:
                a=1.2
                b=7


            roi_eyel = rimg[int(y+ybeg*h):int(y+yend*h), int(x+xbeg*w):int(x+w/2)]
            roi_eyer = rimg[int(y+ybeg*h):int(y+yend*h), int(x+w/2):int(x+xend*w)]
            eyel = eye_cascade.detectMultiScale(roi_eyel,a,b)
            eyer = eye_cascade.detectMultiScale(roi_eyer,a,b)

            for (ex,ey,ew,eh) in eyel:
                cv2.rectangle(roi_eyel, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)
                count_eyes +=1
            for (ex,ey,ew,eh) in eyer:
                cv2.rectangle(roi_eyer, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)
                count_eyes+=1

        rimg = rotate_image_back(rimg, -1*angle)
        #cv2.imshow('eyes', rimg)

        if len(detected):
            break
    return count_eyes

flag=1
scale = 50

while True:
    if flag==1:
        print('Initializing...')

        initial_count=0
        for i in range(0,80):
            ret, img = camera.read()
            width = int(img.shape[1]*scale/100)
            height = int(img.shape[0]*scale/100)
            img = cv2.resize(img, (width,height))
            initial_count+=func(img)
            print(i)
            k = cv2.waitKey(5) & 0xFF

        initial_count=int(initial_count/4)
        flag=0
        continue

    ret, img = camera.read()
    width = int(img.shape[1]*scale/100)
    height = int(img.shape[0]*scale/100)
    img = cv2.resize(img, (width,height))

    if count_frames==0:
        count_eyes=0

    if count_frames==20:
        print(count_eyes-initial_count+5)
        count_eyes=0
        count_frames=0

    count_frames +=1

    count_eyes+=func(img)


    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

    if k==ord('i'):
        flag=1


cv2.destroyAllWindows()
