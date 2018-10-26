import os
import shutil
label_dir=r"C:\Users\DE108470\Desktop\video\BBOX"
image_dir=r"C:\Users\DE108470\Desktop\video\frames"
save_path=r"C:\Users\DE108470\Desktop\video\clean"
for label in os.scandir(label_dir):
    label=label.name.split(".")[0]
    for image in os.scandir(image_dir):
        img=image.name.split(".")[0]
        if img==label:
            shutil.move(image.path, save_path)
