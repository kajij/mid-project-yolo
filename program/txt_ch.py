import os, re, shutil
from xml.etree import ElementTree
folder = 'C:/ml/val_data'
txt = []
txt1 = []
for file in os.listdir(folder):
    x = file.split(".")
    a = x[-1]
    if a =="jpg":
        file_path = folder +'/'+ file
        txt.append(file_path)
txt='\n'.join(txt)
open('C:/ml/val_data.txt', 'w').write(txt)


