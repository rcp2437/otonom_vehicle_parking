#!/usr/bin/env python3

import cv2
import rospy
import json
from std_msgs.msg import String  # JSON formatı için String mesaj tipi
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO('/home/recep/catkin_ws/src/azed_with_park/script/simulasyon_bitirme/train45/weights/best.pt')

def main():
    # ROS düğümünü başlat
    rospy.init_node('yolo_detection', anonymous=True)

    # Yayıncı oluştur (JSON formatında mesaj yayınlayacak)
    publisher = rospy.Publisher('/tespit', String, queue_size=10)

    # CvBridge nesnesi oluştur
    bridge = CvBridge()
     
    classes = ['ileri_veya_sol', 'sag_mecburi', 'durak']



    def image_callback(msg):
        try:
            # ZED2'den gelen görüntüyü OpenCV formatına çevir
            fraM = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.resize(fraM, (720, 480))

        except Exception as e:
            rospy.logerr(f"cv_bridge hatası: {e}")
            return

        # YOLO modelini çalıştır
        results = model.predict(frame, conf=0.5)

        detected_objects = []  # Tespit edilen nesneler listesi
        durak_sayisi = 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())  # Sınıf indeksini al
                class_name = classes[class_id]  # Sınıf ismini belirle

                # Nesnenin koordinatlarını al (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Köşe koordinatları

                # Nesnenin orta noktasını hesapla
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                detected_objects.append({
                    "class": class_name,
                    "x": x_center,
                    "y": y_center
                })

                # Eğer nesne "durak" ise, sayacı artır
                if class_name == "durak":
                    durak_sayisi += 1

                # Nesnenin etrafına bir dikdörtgen çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil renk, kalınlık 2

        # JSON formatında mesaj hazırla
        json_msg = json.dumps({
            "durak_sayisi": durak_sayisi,
            "tespitler": detected_objects
        })

        rospy.loginfo(f"JSON Mesaj: {json_msg}")

        # JSON mesajını yayınla
        publisher.publish(json_msg)

        # Görüntüyü ekranda göster (isteğe bağlı)
        cv2.imshow("Detected Image", frame)
        cv2.waitKey(1)  # Bu, pencereyi sürekli güncellemek için gereklidir

    # `/zed2/left_raw/image_raw_color` konusundan görüntü al
    rospy.Subscriber("/zed2/left_raw/image_raw_color", Image, image_callback, queue_size=1)

    # ROS döngüsünü başlat
    rospy.spin()

    # Pencereyi kapat
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

