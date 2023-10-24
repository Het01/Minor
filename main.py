import cv2
import time
import numpy as np
thres = 0.5 # Threshold to detect object
nms_threshold = 0.1  # Non-maximum Suppression

cap = cv2.VideoCapture("C:/Users/Dell/Desktop/Minor/video2.mp4")
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

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

duration = 10

t_end = time.time() + duration
person_time = []
bottle_time = []
init_time = time.time() 

ct_person_maxi =  -1
person_ct = []

c=0
curtime=time.time()
while time.time() < t_end:

    if(int(curtime+1)==int(time.time())):
        curtime = time.time()


        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        # print(classIds)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))
        #print(type(confs[0]))
        #print(confs)
        
        set_id = 0 
        ct = 0 
        for i in classIds :
            if i == 73 or i==72:
                set_id = 1 
                ct+=1
    
        person_ct.append(ct)
        ct_person_maxi = max(ct_person_maxi,ct) 
        
        if(set_id == 1) :
        # print("Person Detected")
            curr = time.time()
            person_time.append(int(curr-init_time)) 
        # print(curr)
    
    # set_id2= 0 
    # for i in classIds :                
    #     if i ==  44:
    #         set_id2 = 1 
    #         break
        
    # if(set_id2 == 1) :
    #     # print("Bottle Detected")
    #     curr = time.time()
    #     bottle_time.append(int(curr-init_time)) 
    #     # print(curr)
        
        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
        # indices = list(indices)
        # print(indices[1])

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


# print()
# print("Person count : " , person_ct)
# print()
# print("Maximum Person Detect : ",ct_person_maxi)
# print()

# print()
# print("|  Second  |  Number of Persons  |")
# print("|----------|---------------------|")

# j=1
# for i in person_ct:
#     print(f"|  "+j+"   |    "+str(i)+"     |")
#     j+=1

print()
print()
column_width = 10

# Print the header
print(f"{'Second':<{column_width}}{'Number of Laptops':<{column_width}}")

# Iterate over the list and print in tabular form
for index, value in enumerate(person_ct):
    print(f"{index+1:<{column_width}}{value:<{column_width}}")

print()  
# print("Person dectect in time : " , set(person_time)) 
size = len(set(person_time))
arr = list(set(person_time))
# print(arr)

# print()
# print()
# print(set(bottle_time));
# size2 = len(set(bottle_time))
# arr2 = list(set(bottle_time))

# if(arr[0]!=0):
#     print("Person is not detected in between : 0 - ", arr[0])

# for i in range(0,size-1) :
#     if(arr[i+1] - arr[i] > 1) :
#         print("Person Not Detected in between : ",arr[i] , "-" , arr[i+1])

# if(arr[size-1]!=duration-1):
#     print("Person is not detected in between : ", arr[size-1] , "-", int(duration))


# if(size2 > 0 & arr2[0]!=0):
#     print("Bottle is not detected in between : 0 - ", arr2[0])
        
# for i in range(0,size2-1) :
#     if(arr2[i+1] - arr2[i] > 1) :
#         print("Bottle Not Detected in between : ",arr2[i] , "-" , arr2[i+1])
        
# if(arr2[size2-1]!=duration-1):
#     print("Bottle is not detected in between : ", arr2[size2-1] , "-", int(duration))