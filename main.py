import cv2
import time
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


thres = 0.5 # Threshold to detect object
nms_threshold = 0.1  # Non-maximum Suppression

video_path = "C:/Users/Dell/Desktop/Minor/video5.mp4"

cap = cv2.VideoCapture(video_path)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)


if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the duration of the video in seconds
duration = total_frames / fps

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320) 
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



t_end = time.time() + duration

init_time = time.time() 


computers = []
keyboards = []
mouses = []

c=0
curtime=time.time()
while time.time() < t_end:
    
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    # print(classIds)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    # indices = list(indices)
    # print(indices)


    if(int(curtime+1)==int(time.time())):
        
        curtime = time.time()

        count_computer = 0
        count_keyboard = 0
        count_mouse = 0
        
        for i in indices:
            # print(classIds[i])
            if classIds[i] in [72, 73]:
                count_computer += 1
            if classIds[i] == 76:
                count_keyboard += 1
            if classIds[i] == 74:
                count_mouse += 1

        print("---------------------------------------")
        print(f"No of Computers : {count_computer}")
        print(f"No of keyboard : {count_keyboard}")
        print(f"No of Mouse : {count_mouse}")
        print("---------------------------------------")
        print()

        if count_computer == 0:
            print("---------------------------------------")
            print("WARNING !!! NO COMPUTER DETECTED !!!")
            print("---------------------------------------")

        if count_keyboard == 0:
            print("---------------------------------------")
            print("WARNING !!! NO KEYBOARD DETECTED !!!")
            print("---------------------------------------")

        if count_mouse == 0:
            print("---------------------------------------")
            print("WARNING !!! NO MOUSE DETECTED !!!")
            print("---------------------------------------")

        computers.append(count_computer)
        keyboards.append(count_keyboard)
        mouses.append(count_mouse)

    for i in indices:
            # i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        # if len(classIds) != 0:
        #     for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        #         cv2.rectangle(img,box,color=(0,255,0),thickness=2)
        #         cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
        #         cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        #         cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
        #         cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('Output',img)
    cv2.waitKey(1)



# print(computers)
# print(keyboards)
# print(mouses)

print()
print()

table_data = [(i + 1, computers[i], keyboards[i], mouses[i]) for i in range(min(len(computers), len(keyboards), len(mouses)))]

headers = ['Second', 'No of Computer', 'No of Keyboard', 'No of Mouse']

table = tabulate(table_data, headers=headers, tablefmt='grid')

print(table)

print()
print()

not_computers = []
not_keyboards = []
not_mouses = []

for index,i in enumerate(computers):
    if i == 0 :
        not_computers.append(index+1)

for index,i in enumerate(keyboards):
    if i == 0 :
        not_keyboards.append(index+1)

for index,i in enumerate(mouses):
    if i == 0 :
        not_mouses.append(index+1)



if len(not_computers) == 0 :
    print("Computer is detected all the time !!!")
else :
    print(f"Computer is not detected in {not_computers} seconds")

if len(not_keyboards) == 0 :
    print("Keyboard is detected all the time !!!")
else :
    print(f"Keybord is not detected in {not_keyboards} seconds")

if len(not_mouses) == 0 :
    print("Mouse is detected all the time !!!")
else :
    print(f"Mouse is not detected in {not_mouses} seconds")


# Plot Graph 
n = len(computers)
seconds = [i + 1 for i in range(n)]

plt.figure(figsize=(8, 6))
plt.plot(seconds, computers, marker='o', color='b', label='Computers Detected')
plt.xlabel('Seconds')
plt.ylabel('Number of Computers Detected')
plt.title('Number of Computers Detected over Time')
plt.xticks(seconds)  
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(seconds, keyboards, marker='o', color='b', label='Keyboards Detected')
plt.xlabel('Seconds')
plt.ylabel('Number of Keyboards Detected')
plt.title('Number of Keyboards Detected over Time')
plt.xticks(seconds)  
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(seconds, mouses, marker='o', color='b', label='Mouses Detected')
plt.xlabel('Seconds')
plt.ylabel('Number of Mouses Detected')
plt.title('Number of Mouses Detected over Time')
plt.xticks(seconds)  
plt.grid(True)
plt.legend()
plt.show()

print()
print()
