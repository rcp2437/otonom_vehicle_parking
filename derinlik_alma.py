#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from std_msgs.msg import String, Float32
import json
from cv_bridge import CvBridge, CvBridgeError

# CvBridge nesnesi
cv_bridge = CvBridge()
depth_image = None

# Publisher tanımla
nesne_uzaklik_pub = None

def depth_callback(msg):
    """ ZED2'den gelen derinlik görüntüsünü OpenCV formatına çevirir """
    global depth_image
    try:
        depth_imag = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_image = cv2.resize(depth_imag, (720, 480))  
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Hatası: {e}")

def detected_object_callback(msg):
    """ Nesne tespit verisini alır ve uzaklık değerini hesaplar """
    global depth_image, nesne_uzaklik_pub

    if depth_image is None:
        rospy.logwarn("Derinlik görüntüsü henüz alınmadı!")
        return

    detection_data = json.loads(msg.data)

     

    # İlk tespit edilen nesnenin merkez koordinatlarını al
    x_orta = int(detection_data["tespitler"][0]["x"])
    y_orta = int(detection_data["tespitler"][0]["y"])

    # Koordinatları kullanarak derinlik verisini al
    try:
        nesne_uzaklik = depth_image[y_orta, x_orta]
        rospy.loginfo(f"Nesne: {detection_data['tespitler'][0]['class']}, Uzaklık: {nesne_uzaklik} metre")

        # Uzaklık değerini yayınla
        nesne_uzaklik_pub.publish(Float32(nesne_uzaklik))

    except IndexError:
        rospy.logerr("Geçersiz koordinatlar! Derinlik görüntüsü sınırlarını aşıyor.")

def main():
    """ ROS düğümünü başlatır ve abone olur """
    global nesne_uzaklik_pub

    rospy.init_node('uzaklik', anonymous=True)

    # Publisher başlat
    nesne_uzaklik_pub = rospy.Publisher('/nesne_uzaklik', Float32, queue_size=10)

    # Abonelikler
    rospy.Subscriber('/zed2/depth/depth_registered', Image,depth_callback)
    rospy.Subscriber('/tespit', String, detected_object_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass





