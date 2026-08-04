VoltSight

Akıllı elektrikli araç şarj istasyonu yer seçimi ve kentsel altyapı planlaması için açıklanabilir yapay zekâ projesi.

VoltSight; yeni elektrikli araç şarj istasyonları için uygun konumları belirlemeyi amaçlayan, uçtan uca bir yapay zekâ ve mekânsal veri bilimi projesidir.

Proje; açık coğrafi verileri, mekânsal özellik mühendisliğini, makine öğrenmesini ve açıklanabilir yapay zekâ yöntemlerini bir araya getirerek kentsel alanları değerlendirir ve şarj istasyonu uygunluk skorları üretmeyi hedefler.

Proje durumu: Çalışma alanı, analiz gridleri, yol, otopark ve mevcut şarj istasyonu özellikleri hazırlanmıştır. Proje; ek mekânsal veri kaynaklarının toplanması, özellik mühendisliğinin genişletilmesi ve makine öğrenmesi veri kümesinin oluşturulması aşamasında devam etmektedir.

Çalışma Alanı

VoltSight'ın ilk sürümü Çankaya, Ankara, Türkiye bölgesine odaklanmaktadır.

Çankaya ilçe sınırı, sabit 250 × 250 metre boyutundaki analiz hücrelerine ayrılmıştır. Her grid hücresi; ulaşım, altyapı, arazi kullanımı ve kentsel hareketlilik özellikleri kullanılarak ayrı ayrı değerlendirilecektir.

Mevcut İlerleme

Şu ana kadar tamamlanan çalışmalar:

Proje deposu ve klasör yapısı oluşturuldu.

Python sanal ortamı yapılandırıldı.

Veri bilimi ve coğrafi veri işleme bağımlılıkları kuruldu.

Çankaya idari sınırı OpenStreetMap üzerinden elde edildi.

İdari sınır, metre tabanlı izdüşümlü bir koordinat sistemine dönüştürüldü.

250 × 250 metre boyutunda mekânsal analiz gridi üretildi.

GeoPackage ve GeoJSON grid çıktıları oluşturuldu.

Grid önizleme görseli ve teknik özet raporu üretildi.

Geometri, şema, koordinat sistemi ve grid bütünlüğü için Pytest testleri yazıldı.

Yol geometrileri, yol uzunlukları, yoğunluk hesapları ve ana yola uzaklık değerleri üretildi.

Yol özellikleri ve makine öğrenmesi çıktıları için otomatik bütünlük testleri eklendi.

Çankaya ve çevresindeki OpenStreetMap otopark verileri toplandı.

Grid bazında otopark erişilebilirliği, alanı, kapasitesi ve yakınlık özellikleri üretildi.

Otopark geometrileri, yarıçap sayımları, mesafe hesapları ve alan oranları test edildi.

OpenStreetMap üzerindeki mevcut elektrikli araç şarj istasyonları toplandı ve temizlendi.

Grid bazında şarj istasyonu yakınlığı, yoğunluğu, kapasitesi ve bağlantı türü özellikleri üretildi.

İstasyon geometrileri, grid eşleştirmeleri, yarıçap sayımları ve en yakın mesafe hesapları test edildi.

Proje test paketinde 144 başarılı test seviyesine ulaşıldı.

Çalışma Gridi Önizlemesi



Yukarıdaki grid hücreleri VoltSight'ın temel analiz birimlerini oluşturmaktadır. Makine öğrenmesi özellikleri ve uygunluk tahminleri her hücre için ayrı ayrı hesaplanacaktır.

Proje Hedefleri

Açık ve yeniden kullanılabilir coğrafi veri kaynaklarını toplamak

Tekrar çalıştırılabilir bir mekânsal veri işleme hattı geliştirmek

Sabit boyutlu kentsel analiz gridleri üretmek

Ulaşım, altyapı ve kentsel aktivite özellikleri çıkarmak

Makine öğrenmesi modellerini eğitmek ve karşılaştırmak

Şarj istasyonu yer uygunluğu skorları üretmek

Tahminleri SHAP ile açıklamak

Tahmin belirsizliğini değerlendirmek

Model sonuçlarını FastAPI üzerinden sunmak

Sonuçları React ve OpenLayers tabanlı bir web uygulamasında görselleştirmek

Projeyi Docker ile paketlemek

Makine öğrenmesi deneylerini MLflow ile takip etmek

Planlanan Mekânsal Özellikler

Her grid hücresinde aşağıdaki özelliklerin bulunması hedeflenmektedir:

En yakın ana yola uzaklık

Toplam yol uzunluğu ve yol yoğunluğu

Yakındaki otopark sayısı

En yakın otoparka uzaklık

Yakındaki ilgi noktası sayısı

Ticari aktivite yoğunluğu

Konut aktivitesi yoğunluğu

Mevcut şarj istasyonlarına uzaklık

Belirli yarıçaplardaki mevcut şarj istasyonu sayısı

Elektrik altyapısına uzaklık

Nüfus ve kentsel yoğunluk göstergeleri

Hastane, üniversite ve alışveriş merkezlerine yakınlık

Akaryakıt istasyonlarına yakınlık

Toplu taşıma erişilebilirliği

Arazi kullanım özellikleri

Yöntem

VoltSight için planlanan genel iş akışı:

Açık Coğrafi Veriler
        |
        v
Veri Temizleme ve Doğrulama
        |
        v
250 x 250 Metre Analiz Gridi
        |
        v
Mekânsal Özellik Mühendisliği
        |
        v
Keşifsel Veri Analizi
        |
        v
Temel Makine Öğrenmesi Modelleri
        |
        v
Gelişmiş Uygunluk Modeli
        |
        v
SHAP Açıklamaları ve Belirsizlik Analizi
        |
        v
FastAPI Backend
        |
        v
React ve OpenLayers Web Uygulaması

Çalışma Gridi Üretimi

Çalışma gridi veri hattı aşağıdaki işlemleri gerçekleştirir:

OpenStreetMap üzerinden Çankaya idari sınırını sorgular.

Dönen geometrinin poligon yapısında olduğunu doğrular.

Orijinal sınırı EPSG:4326 koordinat sisteminde saklar.

Bölge için uygun yerel UTM koordinat sistemini belirler.

İdari sınırı metre tabanlı koordinat sistemine dönüştürür.

Sabit 250 × 250 metre kare hücreler oluşturur.

Merkez noktası Çankaya sınırı içinde kalan hücreleri seçer.

Her grid hücresine benzersiz bir kimlik atar.

Grid merkez koordinatlarını ve hücre alanlarını hesaplar.

GeoPackage, GeoJSON, önizleme görseli ve özet raporu üretir.

Yol Özellik Mühendisliği

OpenStreetMap üzerindeki sürüşe uygun yol ağı, Çankaya ilçesi ve ilçe çevresindeki ek bir kilometrelik tampon alan için indirilmiştir.

Yol geometrileri, 250 × 250 metre boyutundaki analiz hücreleriyle kesiştirilmiştir. Her grid hücresi için aşağıdaki özellikler üretilmiştir:

road_length_m

road_segment_count

main_road_length_m

main_road_segment_count

road_density_km_per_km2

distance_to_main_road_m

nearest_main_road_type

Üretilen veri kümesi 7.227 grid kaydı içermektedir.

İlk Yol İstatistikleri

Yol verisi içeren grid hücresi: 3.346

Yol verisi içermeyen grid hücresi: 3.881

Ortalama yol yoğunluğu: 5,48 km/km²

Ana yola medyan uzaklık: 375,49 metre

Ana yola maksimum uzaklık: 4.037,23 metre

Yol özellikleri veri hattı ve üretilen çıktılar otomatik Pytest kontrolleriyle doğrulanmaktadır.

Otopark Özellik Mühendisliği

OpenStreetMap üzerindeki amenity=parking etiketi kullanılarak Çankaya ve ilçe çevresindeki bir kilometrelik tampon alanda bulunan otopark verileri toplanmıştır.

Her otopark kaydı temizlenmiş, analiz koordinat sistemine dönüştürülmüş ve benzersiz bir kimlikle eşleştirilmiştir.

Grid bazında aşağıdaki özellikler üretilmiştir:

parking_count

parking_area_m2

parking_area_ratio

distance_to_nearest_parking_m

parking_count_within_500m

parking_count_within_1000m

known_parking_capacity

parking_capacity_record_count

Poligon biçimindeki otopark alanları 250 × 250 metre analiz gridiyle kesiştirilmiştir. Yerel grid atamaları ve yarıçap tabanlı erişilebilirlik hesapları için temsili noktalar kullanılmıştır.

Veri sınırlaması: OpenStreetMap üzerindeki otopark kapsamı ve kapasite bilgileri eksik olabilir. Bu özellikler, resmi ve eksiksiz bir otopark envanterini değil, haritalanmış otopark erişilebilirliğini temsil eder.



Şarj İstasyonu Özellik Mühendisliği

Mevcut elektrikli araç şarj istasyonları, OpenStreetMap üzerindeki amenity=charging_station etiketi kullanılarak Çankaya ve çevresindeki 2,5 kilometrelik tampon alan için toplanmıştır.

Her şarj istasyonu kaydı temizlenmiş, analiz koordinat sistemine dönüştürülmüş ve benzersiz bir kimlikle eşleştirilmiştir.

Grid bazında aşağıdaki sütunlar üretilmiştir:

charging_station_count

has_existing_charging_station

distance_to_nearest_charging_station_m

charging_station_count_within_1000m

charging_station_count_within_2000m

known_charging_capacity

charging_capacity_record_count

ac_station_count_within_1000m

dc_station_count_within_1000m

Grid eşleştirmeleri ve yarıçap tabanlı erişilebilirlik hesapları için temsili istasyon noktaları kullanılmıştır. Mesafeler, izdüşümlü ve metre tabanlı koordinat sisteminde her grid hücresinin merkezinden hesaplanmıştır.

İlk Şarj İstasyonu İstatistikleri

Haritalanmış benzersiz şarj istasyonu: 18

En az bir şarj istasyonu içeren grid hücresi: 9

1.000 metre içinde istasyon bulunan grid hücresi: 322

2.000 metre içinde istasyon bulunan grid hücresi: 1.156

En yakın şarj istasyonuna medyan uzaklık: 5.417,67 metre

En yakın şarj istasyonuna maksimum uzaklık: 26.524,34 metre

AC bağlantısı haritalanmış istasyon: 6

DC bağlantısı haritalanmış istasyon: 3



Bilimsel Sınırlama

OpenStreetMap, mevcut çalışma alanı ve çevresindeki tampon bölgede yalnızca 18 haritalanmış şarj istasyonu kaydı içermektedir. Bu nedenle söz konusu kaynak, eksiksiz ve resmi bir şarj istasyonu envanteri olarak değerlendirilmemelidir.

Nihai makine öğrenmesi hedefi oluşturulmadan önce şarj istasyonu veri kümesinin, açık lisanslı ek veri kaynaklarıyla zenginleştirilmesi planlanmaktadır.

charging_station_count ve has_existing_charging_station sütunları hedef veya betimleyici değişkenlerdir. Mevcut istasyon dağılımını yeniden üretmeyi amaçlayan bir modelde doğrudan tahmin girdisi olarak kullanılmamalıdır.

Mesafe ve çevredeki istasyon sayısı sütunları da model geliştirme sırasında veri sızıntısını önleyecek biçimde değerlendirilmelidir.

Üretilen Grid Çıktıları

Mevcut veri hattı aşağıdaki temel dosyaları üretmektedir:

data/raw/cankaya_boundary_osm.geojson
data/processed/cankaya_grid_250m.gpkg
data/processed/cankaya_grid_250m.geojson
docs/cankaya_grid_preview.png
docs/cankaya_grid_summary.md

Yol, otopark ve şarj istasyonu veri hatları da ilgili ara veri dosyalarını, işlenmiş veri kümelerini, görselleri ve teknik özetleri üretmektedir.

Büyük ham, ara ve işlenmiş veri dosyaları .gitignore kullanılarak Git deposunun dışında tutulur. Bu dosyalar veri hatları yeniden çalıştırılarak üretilebilir.

Kullanılan ve Planlanan Teknolojiler

Veri Bilimi ve Makine Öğrenmesi

Python

NumPy

Pandas

GeoPandas

Scikit-learn

XGBoost

LightGBM

SHAP

Matplotlib

JupyterLab

Coğrafi Veri İşleme

OpenStreetMap

OSMnx

Shapely

PyProj

Pyogrio

GeoJSON

GeoPackage

Backend ve Veri Saklama

FastAPI

PostgreSQL

PostGIS

Pydantic

Frontend ve Görselleştirme

React

OpenLayers

JavaScript

HTML

CSS

Yazılım Mühendisliği ve MLOps

Git

GitHub

Docker

MLflow

Pytest

Proje Yapısı

voltsight-ai/
|-- backend/
|   `-- app/
|       |-- core/
|       |-- routers/
|       |-- schemas/
|       |-- services/
|       |-- __init__.py
|       `-- main.py
|
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|
|-- docs/
|   |-- cankaya_grid_preview.png
|   |-- cankaya_grid_summary.md
|   |-- cankaya_parking_features_preview.png
|   `-- cankaya_charging_features_preview.png
|
|-- frontend/
|
|-- notebooks/
|
|-- src/
|   `-- voltsight/
|       |-- data/
|       |   |-- __init__.py
|       |   `-- create_study_grid.py
|       |
|       |-- evaluation/
|       |-- features/
|       |-- models/
|       `-- __init__.py
|
|-- tests/
|-- .gitignore
|-- README.md
`-- requirements.txt

Yerel Kurulum

Depoyu klonlayın:

git clone https://github.com/ilginbor/voltsight-ai.git
cd voltsight-ai

Python sanal ortamını oluşturun:

python -m venv .venv

Windows PowerShell üzerinde sanal ortamı etkinleştirin:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

Paket yöneticisini güncelleyin ve proje bağımlılıklarını kurun:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Çalışma Gridi Veri Hattını Çalıştırma

Çankaya idari sınırı ve analiz gridi üretim hattını proje ana dizininden çalıştırın:

python src/voltsight/data/create_study_grid.py

Veri hattı başarıyla tamamlandığında aşağıdaki çıktıları üretir:

Çankaya idari sınırı

250 × 250 metre analiz gridi

GeoPackage çıktısı

GeoJSON çıktısı

Grid önizleme görseli

Markdown özet raporu

Testleri Çalıştırma

Proje ana dizininde aşağıdaki komutu çalıştırın:

python -m pytest

Bu komut; grid, yol, otopark ve şarj istasyonu veri hatları için tanımlanan otomatik testleri çalıştırır.

Veri Politikası

VoltSight, kamuya açık ve açık lisanslı veri kümelerini kullanacak şekilde tasarlanmıştır.

Büyük ham, ara ve işlenmiş veri dosyaları doğrudan Git deposunda saklanmaz. Bunun yerine gerekli veri kümelerini elde eden ve yeniden üreten Python veri hatları depoda tutulur.

Özel, kısıtlı veya kamuya açık olmayan veri kümeleri izin alınmadan projeye eklenmemelidir.

Yol Haritası

Aşama 1 — Proje Başlatma

GitHub deposunu oluştur

Python sanal ortamını yapılandır

Tam kapsamlı proje klasör yapısını oluştur

Proje dokümantasyonunu ekle

Git ignore kurallarını yapılandır

Aşama 2 — Çalışma Alanını Hazırlama

Çankaya idari sınırını elde et

İdari sınırı izdüşümlü koordinat sistemine dönüştür

250 × 250 metre analiz gridlerini oluştur

Grid önizlemesini ve teknik özeti üret

Grid geometrilerini doğrula

Grid veri hattı için otomatik testler ekle

Aşama 3 — Mekânsal Veri Toplama

Yol ağını topla

Grid bazında yol özelliklerini hesapla

Yol özelliklerini otomatik testlerle doğrula

Otopark alanlarını topla

Grid bazında otopark özelliklerini hesapla

Otopark özelliklerini otomatik testlerle doğrula

OpenStreetMap şarj istasyonlarını topla

Grid bazında şarj istasyonu özelliklerini hesapla

Şarj istasyonu çıktılarını otomatik testlerle doğrula

Ek bir açık şarj istasyonu veri kaynağı bul

Şarj istasyonu envanterlerini birleştir ve mükerrer kayıtları temizle

Alışveriş merkezlerini ve ticari noktaları topla

Hastane ve üniversiteleri topla

Akaryakıt istasyonlarını topla

Toplu taşıma özelliklerini topla

Konut ve ticari arazi kullanım verilerini topla

Aşama 4 — Özellik Mühendisliği

Ana yola uzaklığı hesapla

Yol yoğunluğunu hesapla

Otopark erişilebilirliğini hesapla

Mevcut şarj istasyonu yoğunluğunu hesapla

Şarj istasyonlarına uzaklık özelliklerini hesapla

Yakındaki ilgi noktası sayılarını hesapla

Ek mesafe tabanlı kentsel özellikleri hesapla

Nihai makine öğrenmesi veri kümesini oluştur

Eksik değerleri ve mekânsal tutarlılığı doğrula

Aşama 5 — Makine Öğrenmesi

Keşifsel veri analizi gerçekleştir

Hedef etiketleri ve kontrol örneklerini tanımla

Lojistik Regresyon temel modelini eğit

Rastgele Orman temel modelini eğit

XGBoost modelini eğit

Modellerin performansını karşılaştır

Mekânsal çapraz doğrulama uygula

Model hiperparametrelerini optimize et

Aşama 6 — Açıklanabilir Yapay Zekâ

SHAP genel açıklamalarını ekle

Grid bazında yerel açıklamalar üret

Tahmin güven değerlerini hesapla

Belirsizlik göstergeleri ekle

Model sınırlamalarını ve olası önyargıları analiz et

Aşama 7 — Backend

FastAPI uygulamasını başlat

Sağlık kontrolü endpointlerini ekle

Grid ve uygunluk endpointlerini ekle

Eğitilmiş modeli entegre et

PostgreSQL ve PostGIS bağlantısını kur

İstek doğrulama mekanizmasını ekle

Backend testlerini yaz

Aşama 8 — Frontend

React uygulamasını başlat

OpenLayers entegrasyonunu gerçekleştir

Çankaya gridini haritada göster

Grid hücrelerini uygunluk skoruna göre renklendir

Tahmin açıklamalarını göster

Filtre ve katman kontrollerini ekle

Aday konum ayrıntılarını göster

Duyarlı arayüz tasarımı ekle

Aşama 9 — Dağıtım ve Dokümantasyon

Docker yapılandırmasını ekle

MLflow deney takibini ekle

Otomatik test iş akışını ekle

Mimari diyagramları hazırla

Nihai teknik raporu hazırla

Demo videosu kaydet

Uygulamayı yayımla

Önemli Bilimsel Sınırlama

VoltSight'ın ilk sürümü, mekânsal örüntülere ve mevcut kentsel özelliklere dayalı konum uygunluğu tahmin edecektir.

Gerçek istasyon kullanım oranı, enerji tüketimi, doluluk veya gelir verileri elde edilmediği sürece model sonuçları; garantili talep, kârlılık veya ticari başarı olarak yorumlanmamalıdır.

VoltSight aşağıdaki kavramları açık biçimde birbirinden ayıracaktır:

Mekânsal uygunluk

Tahmin edilen talep

Finansal uygulanabilirlik

Model güveni

Bu ayrım, bilimsel açıdan sorumlu ve şeffaf bir makine öğrenmesi yaklaşımı için gereklidir.

Lisans

Projede kullanılan tüm dış veri kümelerinin lisansları ve atıf koşulları incelendikten sonra uygun bir proje lisansı seçilecektir.

Geliştirici

Ilgın Bor

Bilgisayar Mühendisliği öğrencisiYapay zekâ, veri bilimi, siber güvenlik ve coğrafi bilgi sistemleri alanlarıyla ilgileniyorum.
