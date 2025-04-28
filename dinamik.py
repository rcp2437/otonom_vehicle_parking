#!/usr/bin/env python3
import rospy
import math
import json
from std_msgs.msg import Float64, Float32, String
nesne_distance = None  # Global değişken olarak başlatma

# /goal konusuna yayın yapacak publisher
sonuc_publisher = rospy.Publisher('/goal', Float64, queue_size=10)

# /sonuc_c_topic konusuna yayın yapacak yeni publisher
#sonuc_c_publisher = rospy.Publisher('/ne_kadar_duz', Float64, queue_size=10)
sonuc_c_publisher2 = rospy.Publisher('/ne_kadar_duz_degisen', Float64, queue_size=10)
# Global variable for the distance ratio
distance_ratio = None

def mesafe_piksellerle(P1, P2):
    # Pikseller arasındaki mesafeyi hesapla
    return math.sqrt((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2)

def pikselden_metresel_mesafeye(dik_mesafe_pikseller, piksel_basi_metre):
    # Pikselleri metreye çevir
    return dik_mesafe_pikseller * piksel_basi_metre

def ucgen_acilari_ve_kenarlari(A, B, C):
    # Noktalardan kenar uzunluklarını hesapla (piksel birimlerinde)
    def mesafe(P1, P2):
        return math.sqrt((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2)

    a = mesafe(B, C)
    b = mesafe(A, C)
    c = mesafe(A, B)

    # Kosinüs Teoremi ile açıları hesapla
    def aci_hesapla(a, b, c):
        return math.degrees(math.acos((b**2 + c**2 - a**2) / (2 * b * c)))

    A_acisi = aci_hesapla(a, b, c)
    B_acisi = aci_hesapla(b, a, c)
    C_acisi = aci_hesapla(c, a, b)

    return A_acisi, B_acisi, C_acisi, a, b, c

def cos_Aacisi_ile_kenar_b(A_acisi, kenar_b_metresel):
    
    return math.cos(math.radians(A_acisi)) * kenar_b_metresel

def calculate_distance_ratio(nesne_distance, kenar_b_metresel):
    global distance_ratio

    # Eğer yeni bir nesne mesafesi gelmezse, en son hesaplanan değeri döndür
    if nesne_distance is None or kenar_b_metresel == 0:
        rospy.logwarn("Yeni nesne mesafesi alınamadı! Son hesaplanan oran kullanılacak.")
        return distance_ratio if distance_ratio is not None else 1.0  # Eğer önceki değer yoksa, 1.0 kullan
    
    # Yeni değer geldiyse, oranı güncelle
    distance_ratio = nesne_distance / kenar_b_metresel
   # rospy.loginfo("Distance ratio updated________---: {:.2f}".format(distance_ratio))
    
    return distance_ratio


def tespit_callback(msg):
    global distance_ratio
#    rospy.loginfo("Alınan /tespit mesajı: " + msg.data)
    
    try:
        data = json.loads(msg.data)
    except Exception as e:
        rospy.logerr("JSON parse hatası: " + str(e))
        return
    
    if "tespitler" in data and len(data["tespitler"]) > 0:
        x = data["tespitler"][0]["x"]
        y = data["tespitler"][0]["y"]
        print(x)
        print(y)
         # Gelen x, y değerlerini C noktasına ata
        
        # B'nin y koordinatını C'den gelen y ile değiştir
    else:
        rospy.logwarn("Tespit verisi eksik, işlem yapılmadı.")
        return

    x_x=abs(360)
    y_y=abs(480)
    A = (x_x, y_y)  # Kamera noktası
    B = (x_x, y)
    C = (x, y)
  
    piksel_basi_metre = 0.1
 

    
    A_acisi, B_acisi, C_acisi, kenar_a, kenar_b, kenar_c = ucgen_acilari_ve_kenarlari(A, B, C)
   # rospy.loginfo("A açısı: {:.2f}°, B açısı: {:.2f}°, C açısı: {:.2f}°".format(A_acisi, B_acisi, C_acisi))
    pub = rospy.Publisher('/a_acisi', Float32, queue_size=10)
    pub.publish(A_acisi)

    kenar_a_metresel = pikselden_metresel_mesafeye(kenar_a, piksel_basi_metre)
    kenar_b_metresel = pikselden_metresel_mesafeye(kenar_b, piksel_basi_metre)
    kenar_c_metresel = pikselden_metresel_mesafeye(kenar_c, piksel_basi_metre)
    #rospy.loginfo("Kenar uzunlukları: A-B: {:.2f} m, A-C: {:.2f} m, B-C: {:.2f} m(YALAN)".format(kenar_c_metresel, kenar_b_metresel, kenar_a_metresel))
  
  
    
 
    distance_ratio=calculate_distance_ratio(nesne_distance, kenar_b_metresel)
    
 
    sonuc_c = distance_ratio * kenar_c_metresel

    sonuc_c_publisher2.publish(sonuc_c)
    rospy.loginfo("->>>>>sonuc_c yayınlandı: {:.2f}".format(sonuc_c))
    

def nesne_uzaklik_callback(msg):
    global nesne_distance
    #rospy.loginfo("Alınan /nesne_uzaklik mesajı: {:.2f}".format(msg.data))
    nesne_distance = msg.data

if __name__ == '__main__':
    # ROS node'u başlat
    rospy.init_node('goal_calculator', anonymous=True)
    rospy.sleep(0.5)
    # /tespit topic'ine abone ol
    rospy.Subscriber('/tespit', String, tespit_callback)
    # /nesne_uzaklik topic'ine abone ol (yayın yapmadan sadece abone oluyoruz)
    rospy.Subscriber('/nesne_uzaklik', Float32, nesne_uzaklik_callback)

    # Programı sürekli çalıştırmak için
    rospy.spin()

