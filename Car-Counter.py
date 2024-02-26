from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*  
import numpy as np
#bunu emalweyden aldık.sort.py
 
cap = cv2.VideoCapture("videos\cars.mp4")  # For Video
model = YOLO("yolov8m.pt")
 
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
 
mask = cv2.imread("Project-1\mask.png") #bunu kendimiz oluşturup buraya vericez.
mask=cv2.resize(mask,(1280,720))
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)#burdaki max tespit edilen şeyden sonra kaç frame sonra diğer şeyi tespit etsin demek.
 #threshold bounding box ne kadar iyi çerçevelenmiş onun için var. emin değilim bak buna 
limits = [400, 297, 673, 297] #oluşturduğumuz çizgi 
totalCount = [] #arabaların id'sini tutucak
while True:
    success, img = cap.read()
  #  img=cv2.resize(img,(950,480))
    imgRegion = cv2.bitwise_and(img,mask) #iki görüntüyü birleştirir.
 
    imgGraphics = cv2.imread("Project-1\graphics.png", cv2.IMREAD_UNCHANGED) #o yukardaki arabalı sayım yazısı içi resim koyduk.
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0)) #bu koyduğumuz resim ile gerçek görüntüyü birleştirdik.
    results = model(imgRegion, stream=True) #modeli mask ile birleştirdiğimiz video ile kurduk. 
    #stream=True kullanılarak her bir video karesi üzerinde modelin çıktısının sürekli olarak işlenmesi ve akışın anlık olarak elde edilmesi sağlanabilir.
 
    detections = np.empty((0, 5)) #burası fonksıyon gereği 5 adet veri alıyor ve içi boş olarak geliyor. definitondan bakabilirsin 
    #x1,y1,x2,y2,score 
 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
 
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
 
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])  #buradaki şartlarımız gerçekleştikten sonra bu detections 
                #arrayına bu değerlerimizi eklicez.
                detections = np.vstack((detections, currentArray)) #np arrayda biz append yapmıyoruz stack ile ekleme yapıyoz. bu şekilde yaptık yani 
 
    resultsTracker = tracker.update(detections) #sort.pydan aldığımız kodları bu şekilde çalıştırıcaz. 
    #bunun için detections diye bir nesne oluşturduk.
 
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5) #burada kırmızı çizgi çektik.
    for result in resultsTracker: #burası id bulma kısmı 
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), #arabaların id'sini göstericek.
                           scale=2, thickness=3, offset=10)
 
        cx, cy = x1 + w // 2, y1 + h // 2 #burada arabaların orta notasını bulmak için işlem yaptık.
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED) #bu orta noktayı arabalrın üstüne koyduk.
 
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15: #orta noktalar o koyduğumuz kırmızı çizgiden 
            #geçince totalcount dizisinin içine yerleşecek. ve bu şekilde sayım yapılmış olacak. 
            if totalCount.count(id) == 0: #daha once hiç totalcount içine girmediyse o id, o zaman totalcount içine al dedik.
                totalCount.append(id) #totalcount içine ekledik
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)#üstlerinden geçince kırmızı çizgiyi yeşil yaptık.
 
    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8) #burada dizinin boyutunu yazdırarak sayım işlemi yapmış olduk.
 
    cv2.imshow("Image", img)
    key=cv2.waitKey(1) 
    if key == ord(' '):
        break
    
    
     
    
     
