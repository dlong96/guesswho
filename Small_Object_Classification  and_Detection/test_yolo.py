import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os
from generate_xml import write_xml

#path='../'
out_path='out'
if not os.path.exists(out_path):os.mkdir(out_path)
savedir='predict_annotation'
if not os.path.exists(savedir):os.mkdir(savedir)





options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 1125,
    'threshold': 0.1,
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]


capture = cv2.VideoCapture('IMG_1185.MOV')
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

count=0
while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        tl_list=[]
        br_list=[]
        objects=[]
        idx_list=[]
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            tl_list.append(tl)
            br = (result['bottomright']['x'], result['bottomright']['y'])
            br_list.append(br)
            label = result['label']
            objects.append(label)
            confidence = result['confidence']
            idx_list.append(confidence)
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))


        #each frame write_xml once
        if results:
        #select all object in a frame which have confidence >0.2 by index
            count +=1
            target = np.where(np.array(idx_list)>0.1)
            #save the images
            tl_list=np.array(tl_list)[target]
            br_list=np.array(br_list)[target]
            objects=np.array(objects)[target]
            image_name=out_path+"/%05d.jpg"%(count)
            print(image_name)
            cv2.imwrite(image_name,frame)
            write_xml(out_path,image_name,objects,tl_list,br_list,savedir)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
