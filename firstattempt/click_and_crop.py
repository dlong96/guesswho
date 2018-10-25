
import argparse
from paths import list_images
import cv2


ref=[]
cropping=False

def click_and_crop(event,x,y,flags,paramter):
    global ref,cropping

    if event ==cv2.EVENT_LBUTTONDOWN:
        ref=[(x,y)]
        cropping=True

    elif event==cv2.EVENT_LBUTTONUP:
        ref.append((x,y))
        cropping=False

    cv2.rectangle(image,ref[0],ref[1],(0,255,0),2)
    cv2.imshow("image%d"%i,image)


ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset', required=True, help="path to orginal")
args=vars(ap.parse_args())


image_paths=list(list_images(args["dataset"]))
for i,image in enumerate(image_paths):
    image =cv2.imread(image)
    image = cv2.resize(image,(800,600),interpolation=cv2.INTER_AREA)
    copy=image.copy()
    i=i+290
    cv2.namedWindow("image%d"%i)
    #cv2.resizeWindow("image", 1980, 1060)
    cv2.setMouseCallback("image%d"%i,click_and_crop)
    #cv.SetMouseCallback(windowName, onMouse)

    while True:
        cv2.imshow("image%d"%i,image)
        key = cv2.waitKey(1) & 0xFF
    # waitKey(0) will display the window infinitely until any keypress

        if key == ord("r"):#reset region
            image= copy.copy()

        elif key == ord("s"): #skip blury image
            break

        elif key == ord("c"):
            break

    if len(ref) == 2:
        region=copy[ref[0][1]:ref[1][1],ref[0][0]:ref[1][0]]
        cv2.imshow("cropped",region)
        cv2.imwrite(r'C:\Users\Desktop\crop_image\crop_image\%d.jpg'%i,region)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
