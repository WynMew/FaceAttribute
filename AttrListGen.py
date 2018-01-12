import numpy as np
import os
import linecache as lc

filelist ='/home/SSD1/celebA/detlist'
dicclebA = '/home/SSD1/celebA/list_attr_celeba.txt'

arr = []

with open(dicclebA) as lmfile:
    lineNum=sum(1 for _ in lmfile)

it=iter(range(1, lineNum))
for m in it:
    line = lc.getline(dicclebA, m)
    Name = line.rstrip('\n')
    file = Name.split(" ")
    ImgName = '/home/SSD1/celebA/det/' + file[0]
    attr = Name[11:]
    arr.append(ImgName)
    arr.append(attr)


with open(filelist) as lmfile:
    lineNum=sum(1 for _ in lmfile)

it=iter(range(1, lineNum))
for m in it:
    line = lc.getline(filelist, m)
    line = line.rstrip('\n')
    file = line.split('/')
    ImgName = '/home/SSD1/celebA/det/' + file[2]
    try:
        idx = arr.index(ImgName)
        attr = arr[idx+1]
        newline = ImgName + " " + attr
        with open('detcelebAG',"a+") as outfile:
            outfile.write(newline + "\n")
    except:
        print(ImgName)