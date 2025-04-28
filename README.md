# otonom_vehicle_parking
Bu projede,GAZEBO ortamında otonom araçların, nesne algılama teknolojisi kullanarak park alanlarına girmelerini sağlayan dinamik bir algoritma geliştirilmiştir. Algoritmanın amacı, araçların bulduğu park alanlarına, park büyüklüğü ve araç boyutları gibi parametrelerin değişkenlik göstermesi durumunda bile sorunsuz bir şekilde park etmelerini sağlamaktır. Bu, otonom araçların park yapma sürecini daha esnek ve verimli hale getirmeyi hedefler.

Algoritma, park alanının boyutunu ve aracın boyutunu dikkate alarak, aracın park alanına girip giremeyeceğini dinamik bir şekilde hesaplar. Bu hesaplamalar, park yerinin büyüklüğünde yaşanan değişiklikleri de göz önünde bulundurur. Yani, park alanının boyutundaki küçük değişiklikler, aracın park yapabilme yeteneğini etkilemeden işleyişin devam etmesini sağlar.

Algoritmanın temel matematiksel yapısı, cosinus teoremi üzerine kuruludur. Bu teori, araç ve park alanı arasındaki açıların hesaplanmasında kullanılarak, aracın park alanına girişine uygun bir yol haritası çıkarılmasına olanak tanır. Ayrıca, matematiksel fonksiyonlardan faydalanarak algoritmanın dinamik bir yapıya kavuşturulması sağlanmıştır. Böylece algoritma, çevredeki değişken koşullara anında adapte olabilen ve sürekli olarak iyileştirilen bir yapıya sahiptir.

KODUN ÇALIŞTIRILMASI:
ilk olarak yolo_topic.py  dosyasını çalıştırınız . (ortamdaki tabelaları tespit edecek olan kod)
dah sonrasında derinlik 

