import os
def list_images(basePath):
  return list_files(basePath, validex=(".jpg","jpeg","png"))

def list_files(basePath, validex=(".jpg","jpeg","png")):
  for (rootdir,dirname,filename) in os.walk(basePath):
      for file in filename:
          ext=file[file.rfind("."):].lower()
          if ext.endswith(validex):
              imagepath=os.path.join(rootdir,file)
              yield imagepath
