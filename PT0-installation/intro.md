# Anlamsal arama Motorlarına Giriş

Bu dosya içerisinde; Anlamsal arama motorları nelerdir, nasıl çalışırlar, hangi yöntemleri uygularlar gibi sorularını bulabilir, aynı zamanda vektör veritabanı kurulumu ve örnek kullanım kodlarına da ulaşabilirsiniz.

##  İçindekiler
1. [Vector DB nedir?](#1-vektör-veritabanı-nedir)
2. [qdrant kurulumu](#2-qdrant-kurulumu)
3. [Embedding modelleri nelerdir, nasıl kullanılır?](#3-embedding-modelleri-nelerdir-nasıl-kullanılır)
4. Python client
5. Collection nedir?
6. Örnek doc idexleme - örnek sorgu nasıl yapılır.


## 1. Vektör Veritabanı nedir?

Yazılım uygulamalarında veritabanları sıklıkla kullanılır. Geliştirilen uygulama ister bir websitesi isterse yapay zeka uygulaması olsun, gerek uygulama içerisinde kullanacağımız gerekse kullanıcıdan alacağımız her türlü bilgiyi saklamak için veritabanlarına ihtiyaç duyarız.

Vektör veritabanları adından da anlaşılabileceği üzere vektör verilerini saklamakta ve bu vektörler üzerinden optimize bir şekide arama yapmamıza olanak sağlamaktadır. Peki anlamsal arama motorlarında veya büyük dil modellerinde hangi aşamada ve amaçla bunları kullanmaktayız?

Anlamsal arama motorları metinlerin anlamlarını temsil etmek için makine öğrenmesi yöntemlerinden faydalanmaktadırlar. Bu makine öğrenmesi yöntemlerinden en yaygın olanı ve yıllariçerisinde gelişmiş olanı "Embedding" olarak adlandırdığımız metini sayılardan oluşan bir liste olarak temsil etmekten geçmektedir. İlerleyen bölümlerde bu yöntemler detaylı olarak anlatılacaktır. 

Vektör veritabanları verilen bir "embedding" ile içerisinde bulunan diğer embedding'leri karşılaştırarak aralarındaki mesafeyi ölçebilmektedirler. Bu mesafe bize iki dökümanın birbirlerine ne kadar benzer olduklarını göstermektedir. Mesafe ne kadar az ile dökümanların birbirlerine o kadar benzer olduğunu varsayarız. Aradaki mesafeyi ölçmek için birden fazla yöntem bulunmaktadır, bunların detayları PT1-intro kısmında incelenecektir.

### ***embedding indexleme ve search için bir foto***

Birçok farklı veritabanı vektör saklama ve arama özelliği sunmaktadır. Genel olarak hepsinin çalışma mantığı aynıdır fakat aralarında memory tüketimi ve hız gibi farklılıklar bulunmaktadır. Biz bu eğitim serisi boyunca *qdrant* veritabanını kullanacağız.

## 2. qdrant Kurulumu

qdrant, anlamsal arama ve vektör veritabanı çözümüdür. Metinlerin anlamlarını temsil etmek için embedding yöntemlerini kullanır ve vektörler üzerinde optimize bir şekilde arama yapmayı sağlar.

qdrant veritabanını indirmenin birden fazla yöntemi bulunmaktadır, Docker bunlar içerisinde en kolay yöntemlerden birisi olabilir. Bu seri içerisinde biz bir docker-compose dosyası yardımı ile qdrant veritabanını çalıştıracağız.

Diğer indirme seçeneklerini görmek için qdrant'in kendi dokümanlarını [buradan](https://qdrant.tech/documentation/guides/installation/) inceleyebilirsiniz.

qdrant'ı kurmak için aşağıdaki adımları izleyebilirsiniz:

1. qdrant'ı yüklemek için terminale aşağıdaki komutu girin:
    ```
    pip install qdrant-client
    ```

2. qdrant'ı başlatmak için terminale aşağıdaki komutu girin:
    > ! Bu komutu çalıştırmadan önce bilgisayarınızda Docker'ın yüklü ve çalışır durumda olduğundan emin olun.

    ``` bash
    docker-compose -f path/to/file/docker-compose.yml up
    ```

3. qdrant'ı kullanmaya başlayabilirsiniz.


## 3. Embedding modelleri nelerdir, nasıl kullanılır?

Metinleri vektörel olarak temsil etmek kullanılan embedding modelleri yıllar içinde gelişmiş ve yeni yöntemler ortaya çıkmıştır. Günümüzde bunlardan en yaygın ve sık kullanılan hali SentenceTransformer modelleridir. 
> SentenceTransformer modelleri benzer metinlerin vektörleri arasındaki mesafeyi azaltıp, benzer olmayan metinlerin vektörleri arasındaki mesafeyi arttırarak eğitilen modellerdir.\
> Bu modellerin eğitimine ve Fine-Tuning konularını ilerleyen bölümlerde detaylı bir şekilde göreceğiz.

BERT temelli bir model mimariye sahip olan SentenceTransformer'lar hakkında detaylı bilgiye [buradan](https://www.sbert.net) ulaşabilirsiniz.

HuggingFace sitesinde sunulan binlerce modele bu [linkten](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=trending) bakabilir istediğiniz modeli kendi projelerinizde kullanabilirsiniz. SentenceTransformers modellerini kullanmak için aşağıdaki adımları takip edebilirsiniz:

1. sentence-transformers kütüphanesini yüklemek için aşağıdaki komutu çalıştırın:
    ```bash
    pip install sentence-transformers
    ```
2. HuggingFace üzerinden bir model seçin ve bunu kopyalayın.
    ```Python
    MODEL_NAME_OR_PATH = "sentence-transformers/all-mpnet-base-v2"
    ```
3. Modeli yükleyin
    ```Python
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(MODEL_NAME_OR_PATH)
    ```

    **Kodu ilk defa çalıştırdığınız zaman ilk başta modeli yükleyeceği için bu biraz uzun sürebilir. Daha sonradan tekrar çalıştırdığınızda bu süre büyük oranda azalacaktır.**

