#!/usr/bin/env python3
 
import json
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import numpy as np
import random
import string
from std_msgs.msg import Int32 ,Float32,Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import math
nekadar_duz_DGS=0
sag_sol_KONTROL=33
green_pixels=0
kutu_ici_yesil_adet = 0
ori=0
tespit=5  # 
save_dir = "/home/recep/catkin_ws/src/image_processing/script/record"
processed_dir = "/home/recep/catkin_ws/src/image_processing/script/islenen_fotolar"
bridge = CvBridge()
last_saved_time = 0

first_message_received = False  # İlk mesaj geldiğinde True olacak
ilk_yaw = 0.0  # İlk yaw açısını saklamak için değişken


# Rastgele dosya adı oluşturma fonksiyonu
def generate_random_filename():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + ".jpg"

# Görüntü callback fonksiyonu
def image_callback(msg):
    global last_saved_time,green_pixels,sag_sol_KONTROL
    current_time = time.time()
    
    if current_time - last_saved_time >= 1.0:
        try:
            # ROS görüntüsünü OpenCV formatına çevir
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

            # Rastgele dosya adlarını oluştur
            raw_file_name = os.path.join(save_dir, generate_random_filename())
            processed_file_name = os.path.join(processed_dir, generate_random_filename())

            # Orijinal görüntüyü kaydet
            cv2.imwrite(raw_file_name, cv_image)

            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([100, 150, 0])
            upper_blue = np.array([140, 255, 255])

            blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

            # Mavi pikselleri yeşile çevir
            cv_image[blue_mask > 0] = [0, 255, 0]  # BGR formatında saf yeşil (0, 255, 0)

            # İşlenmiş görüntüdeki yeşil pikselleri say
            green_pixels = np.sum((cv_image == [0, 255, 0]).all(axis=2))

            pubb = rospy.Publisher('yesil_adet_topic', Int32, queue_size=10)   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            pubb.publish(green_pixels)
            #rospy.loginfo(f"Alınan mesaj(YESİL ADET): {green_pixels}")
 
            # Sağ-sol ayrımı için görüntünün genişliğini al
            height, width, _ = cv_image.shape
            mid_x = width // 2  # Orta çizgi

            # Sağ ve sol bölgelerde yeşil pikselleri say
            left_region = cv_image[:, :mid_x]  # Sol yarı
            right_region = cv_image[:, mid_x:]  # Sağ yarı

            left_green_pixels = np.sum((left_region == [0, 255, 0]).all(axis=2))
            right_green_pixels = np.sum((right_region == [0, 255, 0]).all(axis=2))

            # Yeşil piksellerin konumunu belirle
            if right_green_pixels > 0:
                sag_sol_KONTROL = 1  # Sağda var
            elif left_green_pixels > 0 :
                sag_sol_KONTROL = -1  # Solda var

            PUB_SAGMI_SOLMU = rospy.Publisher('sag_mı_sol_mu', Int32, queue_size=10)   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            PUB_SAGMI_SOLMU.publish(sag_sol_KONTROL)

            cv2.imwrite(processed_file_name, cv_image)
            last_saved_time = current_time
        except Exception as e:
            rospy.logerr(f"Hata: {e}")
 
def sagmı_solmu_kontrol_callback(msg):
    global sag_sol_KONTROL
    sag_sol_KONTROL=int(msg.data)
                    
                
                    
def tespit_callback(msg):
    global veriler
    veriler=json.loads(msg.data)
    
   

def callback(msg):
    global kutu_ici_yesil_adet 
    kutu_ici_yesil_adet=msg.data
    #rospy.loginfo(f"Alınan mesaj(KUTU İÇİ): {msg.data}")
    
def quaternion_to_yaw(z, w):
    """IMU'dan gelen z ve w değerlerini açısal (yaw) dereceye çevirir."""
    yaw = math.atan2(2.0 * (w * z), 1.0 - 2.0 * (z * z))
    return math.degrees(yaw)  # Radyanı dereceye çevir




def imu_callback(msg):
    global first_message_received, ilk_yaw ,ori,delta_yaw
    ori=msg		
 
    
    anlık_z = msg.orientation.z
    anlık_w = msg.orientation.w

    # Quaternion -> Euler Yaw açısını hesapla
    anlık_yaw = quaternion_to_yaw(anlık_z, anlık_w)

    if not first_message_received:  # İlk mesajı yazdır
        ilk_yaw = anlık_yaw  # İlk yaw açısını sakla
        rospy.loginfo(f"IMU İlk Oryantasyon Açısı: {ilk_yaw:.2f}°")
        first_message_received = True  # Bir kez yazdırdıktan sonra dur

    # Açı değişimini hesapla
    delta_yaw = anlık_yaw - ilk_yaw
    #rospy.loginfo(f"IMU Anlık Oryantasyon Açısı: {anlık_yaw:.2f}° | Açı Değişimi: {delta_yaw:.2f}°")


 
def nesne_uzaklık_callback(msg):
    global nesne_uzaklık 
    nesne_uzaklık=msg.data
   # print(nesne_uzaklık)
 
 
    
 
def ne_kadar_duz_callback(msg):
    global nekadar_duz, first_received
    if not first_received:  # Eğer daha önce veri alınmadıysa
        nekadar_duz = msg.data
        first_received = True  # Artık veri alındığını işaretle


first_received = False
def ne_kadar_duz_degisen_callback(msg):
    global nekadar_duz_DGS
    nekadar_duz_DGS = msg.data




def move_robot():
    global kutu_ici_yesil_adet, green_pixels, tespit, sag_sol_KONTROL, veriler, nesne_uzaklık, delta_yaw, nekadar_duz, nekadar_duz_DGS
    nesne_uzaklık = 0
    exit_loop = False
    cmd_vel_pub = rospy.Publisher('/ackermann_steering_controller/cmd_vel', Twist, queue_size=10)
    
    if veriler["durak_sayisi"] == 1: 
        tolerans = 0.07
        ilk_ori = 0
        ilk_ori2 = 0
        smn = 0
        smn2 = 0        
        cntrl = 0
        gir = 0
        rospy.sleep(1)
 
        
        rate = rospy.Rate(120)  # 150 Hz döngü hızı burada tanımlanmalı
        
        while not rospy.is_shutdown():
            if nekadar_duz_DGS > 1.70:  # Düz gider
                if math.isnan(nekadar_duz_DGS):  # Eğer NaN ise çık
                    print("NaN değeri algılandı, if bloğundan çıkılıyor.")
                    break

                move_cmd = Twist()
                move_cmd.linear.x = 1.0
                move_cmd.angular.z = 0.0
                print("!'!'!'!'!'!'!'!'!'")
                cmd_vel_pub.publish(move_cmd)

            elif   gir == 0:
                
                while not exit_loop: 
                       # SAĞ YAPAR
                    move_cmd = Twist()
                    move_cmd.linear.x = 0.50
                    move_cmd.angular.z = -0.48 * sag_sol_KONTROL
                    cmd_vel_pub.publish(move_cmd)
 
                    gir = 1

                    print(nekadar_duz_DGS)
                     
                    if nesne_uzaklık - nekadar_duz_DGS<0.1:
                        while True:
                            if 0.90 < nesne_uzaklık :
                                print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")  # BURADA ZEDLE DAHA DİNAMİK YAPACAM
                                print(nesne_uzaklık)
                                move_cmd = Twist()
                                move_cmd.linear.x = 0.4  # Düz giderek tabelaya yaklaşma
                                move_cmd.angular.z = 0.0
                                cmd_vel_pub.publish(move_cmd)
                                print("GİT")
                            else:
                                print("hoppppppppppppppppppppppp")                            
                                move_cmd = Twist()
                                move_cmd.linear.x = 0.0  # Tabelaya yaklaşınca durma
                                move_cmd.angular.z = 0.0
                                cmd_vel_pub.publish(move_cmd)
                                rospy.sleep(2)
                                exit_loop = True
                                ilk_ori = ori.orientation.w
                                ilk_ori2 = ilk_ori * (-1)
                                cntrl = 1
                                break
                                
                    if exit_loop:
                        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        break

            else:
                move_cmd = Twist()
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                move_cmd.linear.x = 0.4  # sol yapmaya başlama
                move_cmd.angular.z = 0.5 * sag_sol_KONTROL
                print(delta_yaw)

                if (-10 < delta_yaw < 10) and smn == 0:  # İlk hareket noktasına göre dinamik yönlendirme
                    print("DEVAMMMMMMMMMMMMMMMMMMMMM")									     
                    move_cmd = Twist()
                    move_cmd.linear.x = 0.0  # Yolcu alma							
                    move_cmd.angular.z = 0.0  # Durma
                    cmd_vel_pub.publish(move_cmd)
                    rospy.sleep(3)  # 3 saniye bekler ve sonra devam eder
                    smn = 1                    
                    
                if sag_sol_KONTROL == 1:
                    if delta_yaw > 40:  # İstikamet 1'in en son durdurulması
                        print("SAGLI GİRİŞ")
                        move_cmd = Twist()
                        move_cmd.linear.x = 0.0   
                        move_cmd.angular.z = 0.0
                        cmd_vel_pub.publish(move_cmd)
                        rospy.sleep(3)
                        print("????????")
                        
                      #////////////////////////________________&&&&&&&&&&&&&&&&&&&  
        
                cmd_vel_pub.publish(move_cmd)
            
            rate.sleep()  # Döngüyü sabit hızda çalıştır
        
    else:
        while True:
            move_cmd = Twist()
            move_cmd.linear.x = 0.8
            move_cmd.angular.z = 0.0
            cmd_vel_pub.publish(move_cmd)
            rospy.sleep(0.1)  # Sabit hızda gitmesini sağlamak için küçük bir bekleme süresi


def main():
    rospy.init_node('arac_hareket_node', anonymous=True)
 

    #rospy.Subscriber('/camera1/image_raw', Image, image_callback)
    rospy.Subscriber('/zed2/left_raw/image_raw_color', Image, image_callback)
    #"dcjkalkşfjLŞJNFŞOJSFBURALA
    
    
    rospy.Subscriber('/nesne_uzaklik', Float32, nesne_uzaklık_callback)   # TESPİTİ YOLO DA PUBLİSH EDİYORUM.                                                            # tespit json dönüyor
    
    rospy.Subscriber('/kutu_ici_adet', Int32,callback)
    rospy.Subscriber('/imu', Imu, imu_callback)
    rospy.Subscriber('/tespit', String, tespit_callback)   # TESPİTİ YOLO DA PUBLİSH EDİYORUM.                                                            # tespit json dönüyor
    rospy.Subscriber('/sag_mı_sol_mu', Int32, sagmı_solmu_kontrol_callback)   # TESPİTİ YOLO DA PUBLİSH EDİYORUM.    
    rospy.Subscriber('/ne_kadar_duz', Float64, ne_kadar_duz_callback)
    rospy.Subscriber('/ne_kadar_duz_degisen', Float64, ne_kadar_duz_degisen_callback)
    rospy.sleep(0.5)
    move_robot()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Düğüm durduruldu.")

