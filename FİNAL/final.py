import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu
import pandas as pd
from matplotlib.widgets import Slider, Button


class ImageProcessingFinalProject:
    def __init__(self):
        self.fig = None
        self.current_image = None
        self.original_image = None

    # ==================== GÖREV 1: S-CURVE KONTRAST GÜÇLENDİRME ====================
    def sigmoid_contrast(self, img, alpha=1, beta=0):
        """Standart sigmoid fonksiyonu ile kontrast artırma"""
        img_normalized = img / 255.0
        sigmoid = 1 / (1 + np.exp(-alpha * (img_normalized - beta)))
        return (sigmoid * 255).astype(np.uint8)

    def shifted_sigmoid(self, img, alpha=1, beta=0.5):
        """Yatay kaydırılmış sigmoid fonksiyonu"""
        return self.sigmoid_contrast(img, alpha, beta)

    def tilted_sigmoid(self, img, alpha=0.5, beta=0):
        """Eğimli sigmoid fonksiyonu"""
        img_normalized = img / 255.0
        sigmoid = 1 / (1 + np.exp(-alpha * (img_normalized - beta)))
        tilted = sigmoid * img_normalized  # Eğim efekti için çarpım
        return (tilted * 255).astype(np.uint8)

    def custom_curve(self, img, alpha=1, beta=0, gamma=1):
        """Özel fonksiyon: Sigmoid ve gama düzeltme kombinasyonu"""
        img_normalized = img / 255.0
        sigmoid = 1 / (1 + np.exp(-alpha * (img_normalized - beta)))
        custom = np.power(sigmoid, gamma)
        return (custom * 255).astype(np.uint8)

    def show_s_curve_results(self, image_path):
        """Görev 1: S-Curve kontrast güçlendirme sonuçlarını göster"""
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError("Görüntü yüklenemedi. Lütfen dosya yolunu kontrol edin.")

        # Kontrast artırma işlemleri
        sigmoid_img = self.sigmoid_contrast(self.original_image)
        shifted_img = self.shifted_sigmoid(self.original_image)
        tilted_img = self.tilted_sigmoid(self.original_image)
        custom_img = self.custom_curve(self.original_image)

        # Sonuçları görselleştirme
        plt.figure(figsize=(15, 10))
        plt.suptitle("Görev 1: S-Curve ile Kontrast Güçlendirme", fontsize=16)

        plt.subplot(2, 3, 1), plt.imshow(self.original_image, cmap='gray'), plt.title('Orijinal Görüntü')
        plt.subplot(2, 3, 2), plt.imshow(sigmoid_img, cmap='gray'), plt.title('Standart Sigmoid')
        plt.subplot(2, 3, 3), plt.imshow(shifted_img, cmap='gray'), plt.title('Yatay Kaydırılmış')
        plt.subplot(2, 3, 4), plt.imshow(tilted_img, cmap='gray'), plt.title('Eğimli Sigmoid')
        plt.subplot(2, 3, 5), plt.imshow(custom_img, cmap='gray'), plt.title('Özel Fonksiyon')

        # S-Curve grafiklerini çiz
        x = np.linspace(0, 1, 256)
        plt.subplot(2, 3, 6)
        plt.plot(x, 1 / (1 + np.exp(-1 * (x - 0))), label='Standart Sigmoid')
        plt.plot(x, 1 / (1 + np.exp(-1 * (x - 0.5))), label='Yatay Kaydırılmış')
        plt.plot(x, (1 / (1 + np.exp(-0.5 * (x - 0)))) * x, label='Eğimli Sigmoid')
        plt.plot(x, np.power(1 / (1 + np.exp(-1 * (x - 0))), 0.5), label='Özel Fonksiyon')
        plt.title('S-Curve Fonksiyonları'), plt.legend()

        plt.tight_layout()
        plt.show()

    # ==================== GÖREV 2: HOUGH DÖNÜŞÜMÜ ====================
    def detect_lane_lines(self, image_path):
        """Görev 2a: Yol çizgilerini tespit et"""
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Görüntü yüklenemedi.")

        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gürültüyü azalt
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Kenar tespiti
        edges = cv2.Canny(blur, 50, 150)

        # ROI (Region of Interest) maskesi oluştur
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (width // 2, height // 2),
            (width, height),
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough çizgi tespiti
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=20)

        # Çizgileri çiz
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Sonucu birleştir
        result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

        # Görselleştirme
        plt.figure(figsize=(15, 5))
        plt.suptitle("Görev 2a: Hough Dönüşümü ile Yol Çizgileri Tespiti", fontsize=16)
        plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Orijinal')
        plt.subplot(1, 3, 2), plt.imshow(edges, cmap='gray'), plt.title('Kenarlar')
        plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Tespit Edilen Çizgiler')
        plt.show()

    def detect_eyes(self, image_path):
        """Görev 2b: Gözleri tespit et"""
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Görüntü yüklenemedi.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Yüz ve göz sınıflandırıcıları
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Yüzleri tespit et
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Her yüz için gözleri tespit et
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

            # Gözleri tespit et
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Hough dairesi için göz bölgesini işle
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]

                # Hough dairesi tespiti
                circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1, 20,
                                           param1=50, param2=30, minRadius=0, maxRadius=0)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        # Daireyi çiz
                        cv2.circle(roi_color, (ex + i[0], ey + i[1]), i[2], (0, 255, 0), 2)
                        # Daire merkezini çiz
                        cv2.circle(roi_color, (ex + i[0], ey + i[1]), 2, (0, 0, 255), 3)

        # Sonucu göster
        plt.figure(figsize=(10, 5))
        plt.suptitle("Görev 2b: Hough Dönüşümü ile Göz Tespiti", fontsize=16)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Göz Tespiti (Hough Daireleri)')
        plt.axis('off')
        plt.show()

    # ==================== GÖREV 3: DEBLURRING ALGORİTMASI ====================
    def motion_deblur(self, image_path, kernel_size=15, angle=0):
        """Görev 3: Motion blur giderme"""
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Görüntü yüklenemedi.")

        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Wiener filtresi için kernel oluştur
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Kernel'i belirtilen açıya döndür
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

        # Wiener filtresini uygula
        deblurred = cv2.filter2D(gray, -1, kernel)
        deblurred = np.uint8(cv2.normalize(deblurred, None, 0, 255, cv2.NORM_MINMAX))

        # Sonuçları göster
        plt.figure(figsize=(15, 5))
        plt.suptitle("Görev 3: Motion Deblurring Algoritması", fontsize=16)
        plt.subplot(1, 3, 1), plt.imshow(gray, cmap='gray'), plt.title('Bulanık Görüntü')
        plt.subplot(1, 3, 2), plt.imshow(kernel, cmap='gray'), plt.title('Motion Kernel')
        plt.subplot(1, 3, 3), plt.imshow(deblurred, cmap='gray'), plt.title('Deblurred Görüntü')
        plt.show()

    # ==================== GÖREV 4: NESNE SAYMA VE ÖZELLİK ÇIKARMA ====================
    def analyze_green_areas(self, image_path):
        """Görev 4: Yeşil bölgeleri analiz et ve Excel'e kaydet"""
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Görüntü yüklenemedi.")

        # HSV renk uzayına çevir
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Yeşil renk aralığı (koyu yeşil)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        # Maske oluştur
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morfolojik işlemler
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Bağlantılı bileşen analizi
        labeled = label(mask)
        regions = regionprops(labeled, intensity_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # Özellikleri topla
        data = []
        for i, region in enumerate(regions):
            # Küçük bölgeleri filtrele
            if region.area < 100:
                continue

            # Özellikleri hesapla
            center = region.centroid[::-1]  # (x,y) formatına çevir
            length = region.major_axis_length
            width = region.minor_axis_length
            diagonal = np.sqrt(length ** 2 + width ** 2)

            # Enerji ve entropi
            hist = np.histogram(region.intensity_image[region.image], bins=256, range=(0, 256))[0]
            hist = hist / hist.sum()  # Normalize
            energy = np.sum(hist ** 2)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))  # log(0) hatasını önle

            # Ortalama ve medyan
            mean_val = np.mean(region.intensity_image[region.image])
            median_val = np.median(region.intensity_image[region.image])

            data.append([
                i + 1,
                f"{center[0]:.1f},{center[1]:.1f}",
                f"{length:.1f} px",
                f"{width:.1f} px",
                f"{diagonal:.1f} px",
                f"{energy:.3f}",
                f"{entropy:.2f}",
                f"{mean_val:.1f}",
                f"{median_val:.1f}"
            ])

        # DataFrame oluştur
        df = pd.DataFrame(data, columns=[
            'No', 'Center', 'Length', 'Width', 'Diagonal',
            'Energy', 'Entropy', 'Mean', 'Median'
        ])

        # Sonuçları göster
        plt.figure(figsize=(15, 5))
        plt.suptitle("Görev 4: Yeşil Bölge Analizi", fontsize=16)
        plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Orijinal Görüntü')
        plt.subplot(1, 2, 2), plt.imshow(mask, cmap='gray'), plt.title('Koyu Yeşil Bölgeler')
        plt.show()

        # Excel'e kaydet
        df.to_excel('green_areas_analysis.xlsx', index=False)
        print("\nAnaliz Sonuçları:")
        print(df)
        return df

    # ==================== MENÜ SİSTEMİ ====================
    def run_menu(self):
        """Ana menüyü çalıştır"""
        print("\n" + "=" * 50)
        print("Dijital Görüntü İşleme Final Ödevi")
        print("=" * 50)
        print("1. S-Curve ile Kontrast Güçlendirme")
        print("2. Hough Dönüşümü ile Çizgi Tespiti (Yol Çizgileri)")
        print("3. Hough Dönüşümü ile Göz Tespiti")
        print("4. Motion Deblurring Algoritması")
        print("5. Yeşil Bölge Analizi ve Özellik Çıkarma")
        print("6. Tüm Görevleri Otomatik Çalıştır (Örnek Görüntülerle)")
        print("7. Çıkış")

        while True:
            choice = input("\nSeçiminizi yapın (1-7): ")

            if choice == '1':
                image_path = input("Görüntü yolunu girin (veya örnek için 'sample1.jpg' yazın): ")
                if image_path == 'sample1.jpg':
                    # Örnek bir görüntü oluştur
                    self.create_sample_image()
                    image_path = 'sample1.jpg'
                self.show_s_curve_results(image_path)

            elif choice == '2':
                image_path = input("Yol görüntüsü yolunu girin (veya örnek için 'road.jpg' yazın): ")
                if image_path == 'output_image.jpg':
                    self.create_sample_road_image()
                    image_path = 'output_image.jpg'
                self.detect_lane_lines(image_path)

            elif choice == '3':
                image_path = input("Yüz görüntüsü yolunu girin (veya örnek için 'face.jpg' yazın): ")
                if image_path == 'face.jpg':
                    self.create_sample_face_image()
                    image_path = 'face.jpg'
                self.detect_eyes(image_path)

            elif choice == '4':
                image_path = input("Bulanık görüntü yolunu girin (veya örnek için 'blurred.jpg' yazın): ")
                if image_path == 'blurred.jpg':
                    self.create_sample_blurred_image()
                    image_path = 'blurred.jpg'
                kernel_size = int(input("Kernel boyutu (varsayılan 15): ") or "15")
                angle = int(input("Açı (varsayılan 0): ") or "0")
                self.motion_deblur(image_path, kernel_size, angle)

            elif choice == '5':
                image_path = input("Tarla görüntüsü yolunu girin (veya örnek için 'field.jpg' yazın): ")
                if image_path == 'field.jpg':
                    self.create_sample_field_image()
                    image_path = 'field.jpg'
                self.analyze_green_areas(image_path)

            elif choice == '6':
                print("Tüm görevler örnek görüntülerle çalıştırılıyor...")
                try:
                    # Örnek görüntüleri oluştur
                    self.create_sample_image()
                    self.create_sample_road_image()
                    self.create_sample_face_image()
                    self.create_sample_blurred_image()
                    self.create_sample_field_image()

                    # Tüm görevleri çalıştır
                    self.show_s_curve_results('sample1.jpg')
                    self.detect_lane_lines('output_image')
                    self.detect_eyes('face.jpg')
                    self.motion_deblur('blurred.jpg', 30, 45)
                    self.analyze_green_areas('field.jpg')
                    print("\nTüm görevler başarıyla tamamlandı!")
                except Exception as e:
                    print(f"Hata oluştu: {str(e)}")

            elif choice == '7':
                print("Programdan çıkılıyor...")
                break

            else:
                print("Geçersiz seçim! Lütfen 1-7 arasında bir sayı girin.")

    # ==================== ÖRNEK GÖRÜNTÜ OLUŞTURMA ====================
    def create_sample_image(self):
        """Görev 1 için örnek görüntü oluştur"""
        # Düşük kontrastlı bir görüntü oluştur
        img = np.zeros((256, 256), dtype=np.uint8)
        for i in range(256):
            img[:, i] = i

        # Gürültü ekle
        noise = np.random.normal(0, 15, (256, 256)).astype(np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite('sample1.jpg', img)
        print("Örnek görüntü 'sample1.jpg' olarak kaydedildi.")

    def create_sample_road_image(self):
        """Görev 2a için örnek yol görüntüsü oluştur"""
        # Beyaz arka plan
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Yol çizgileri çiz
        cv2.line(img, (100, 400), (300, 200), (0, 0, 0), 10)
        cv2.line(img, (300, 200), (500, 400), (0, 0, 0), 10)

        # Gürültü ekle
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite('output_image', img)
        print("Örnek yol görüntüsü 'road.jpg' olarak kaydedildi.")

    def create_sample_face_image(self):
        """Görev 2b için örnek yüz görüntüsü oluştur"""
        # Beyaz arka plan
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255

        # Yüz çiz (oval)
        cv2.ellipse(img, (150, 150), (100, 150), 0, 0, 360, (200, 200, 200), -1)

        # Gözler çiz
        cv2.circle(img, (100, 120), 20, (255, 255, 255), -1)
        cv2.circle(img, (100, 120), 10, (0, 0, 0), -1)
        cv2.circle(img, (200, 120), 20, (255, 255, 255), -1)
        cv2.circle(img, (200, 120), 10, (0, 0, 0), -1)

        # Ağız çiz
        cv2.ellipse(img, (150, 200), (50, 30), 0, 0, 180, (0, 0, 0), 2)

        cv2.imwrite('face.jpg', img)
        print("Örnek yüz görüntüsü 'face.jpg' olarak kaydedildi.")

    def create_sample_blurred_image(self):
        """Görev 3 için örnek bulanık görüntü oluştur"""
        # Orijinal görüntüyü oluştur (dikey çizgiler)
        img = np.zeros((256, 256), dtype=np.uint8)
        for i in range(0, 256, 10):
            img[:, i:i + 5] = 255

        # Motion blur uygula
        kernel_size = 30
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        blurred = cv2.filter2D(img, -1, kernel)

        # Gürültü ekle
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        blurred = cv2.add(blurred, noise)
        cv2.imwrite('blurred.jpg', blurred)
        print("Örnek bulanık görüntü 'blurred.jpg' olarak kaydedildi.")

    def create_sample_field_image(self):
        """Görev 4 için örnek tarla görüntüsü oluştur"""
        # Yeşil arka plan
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img[:, :, 1] = 100  # Yeşil kanal

        # Rastgele yeşil bölgeler ekle
        for _ in range(10):
            center = (np.random.randint(50, 550), np.random.randint(50, 350))
            axes = (np.random.randint(20, 50), np.random.randint(10, 30))
            angle = np.random.randint(0, 180)
            color = (0, np.random.randint(150, 255), 0)
            cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)

        # Gürültü ekle
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite('field.jpg', img)
        print("Örnek tarla görüntüsü 'field.jpg' olarak kaydedildi.")


# Programı başlat
if __name__ == "__main__":
    project = ImageProcessingFinalProject()
    project.run_menu()