# Anlamsal Arama Motorları Temel Giriş

Bir önceki bölümde vektör veri tabanlarından, vektörlerin nasıl oluşturulduğundan ve vektör veri tabanı kullanılarak örnek bir anlamsal arama yapmıştık. Bu bölümde ise anlamsal arama motorlarının temelini oluşturan embedding modelleri ve benzerlik hesaplama yöntemlerinden bahsedeceğiz.

## İçindekiler
1. [Embedding nedir??](#1-embedding-nedir)
2. [Embedding Model Çeşitleri](#2-embedding-model-çeşitleri)
    * [SentenceTransformers](#sentenceTransformers)
    * [Cross Encoders](#cross-encoders)
3. [Anlamsal Benzerlik](#3-benzerlik-nasıl-hesaplanır)
    * [Cosine Similarity](#cosine-similarity)
    * [Euclidean Distance](#euclidean-distance)
    * [Manhattan Distance](#manhattan-distance)


## 1. Embedding nedir?
Eğer daha önceden yapay zeka veya makine öğrenmesi ile ilgilendiyseniz bu algoritmaların sayılar ile işlemler yaptığını bilirsiniz, çünkü bu algoritmalar kelimelerin ne demek olduğunu aslında bilmiyor. Bu yüzden Doğal Dil İşleme algoritmaları bir şekilde metinleri sayılar ile temsil etmek zorundadırlar. Bunlara arama motorları, ChatGPT gibi sohbet yapay zekaları veya metin içerisinden bir bilgi çıkartmak isteyen herhangi bir uygulama örnek olarak verilebilir.

Embedding dediğimiz şey temelde kelimelerin numaralardan oluşan bir listedir. Yıllar içerisinde bu numaraları oluşturmanın birçok farklı yöntemi keşfedilmiştir. Bunlardan en basiti **CountVectorizer** olarak bilinen, kelimelerin frekansı ile temsil etmektir. Bu yöntemde öncelikle bütün dokümanlarda kullanılan kelimeler bir listeye konulur, her bir doküman ise bu kelimelerden kaç tane içerdiğine göre numaralandırılarak temsil edilir.

## ***TODO:*** COUNTVECTORIZER ORNEGI BIR FOTO.


## 2. Embedding Model Çeşitleri

Tabi ki **CountVectorizer** bir metini temsil etmek için kullanılabilecek en basit yöntemdir ve tahmin edilebileceği üzere performans olarak oldukça kötü sonuçlar vermektedir. Geçmişten günümüze kadar sıklıkla kullanılan, adından sıklıkla söz ettirmiş ve birçok alanda kullanılmış bazı embedding modelleri:

* CountVectorizer
* Term Frequency Inverse Document Frequency (TFIDF)
* Word2Vec
* GloVe
* fastText
* BERT
* SentenceTransformers
* Cross Encoders


Biz bu tutorial boyunca ağırlıklı olarak SentenceTransformers ve Cross Encoders üzerinde duracağız. Çünkü bu modellerin temelinde BERT modeli yatmaktadır ve bu modellerin performansı diğer modellere göre oldukça yüksektir. Bu modeller cümlelerin anlamını temsil etmek için özel olarak eğitilmektedirler. Bu modeller eğitimleri sırasında cümlelerin benzerliklerini ve farklılıklarını öğrenirler ve bu sayede cümleleri anlamlarına göre temsil edebilirler.

### SentenceTransformers
SentenceTransformers modelleri cümleleri, kelimeleri veya dokümanları temsil etmek için kullanılabilir. Bu modeller temelde iki farklı katmandan oluşmaktadırlar:
1. BERT modeli
2. Pooling katmanı

BERT modeli transformer tabanlı bir modeldir ve cümlelerin anlamını öğrenmek için eğitilmiştir. Eğitimi genellikle MNR (Masked Language Model) yöntemi yani maskelenmiş dil modeli yöntemi ile yapılmaktadır. Bu yöntemde cümlelerin içerisinden rastgele kelimeler maskelenerek modelin bunları tahmin etmesi istenir ve model bu sayede cümlelerin anlamını öğrenir.

BERT çıktı olarak 512x768 boyutunda bir matris verir. 512 token sayısını temsil ederken, 768 ise her bir token için oluşturulan vektörlerin uzunluğunu temsil eder. Pooling yönteminde ise bu vektörlerin nasıl işleneceğine karar verilir. En yaygın pooling yöntemi **mean pooling** adı verilen yöntemdir. Bu yöntemde tüm token vektörlerinin ortalaması alınarak 1x768 boyutunda bir vektör elde edilir.
**mean pooling** dışında **mean pooling**, **max pooling**, **cls token pooling** gibi birçok farklı pooling yöntemi bulunmaktadır.

## ***TODO:*** ST FLOW ORNEGI BIR FOTO.

### Cross Encoders
Cross Encoder modelleri ise iki farklı cümlenin karşılaştırılması için kullanılır. **SentenceTransformers** modellerindeki gibi ayrı ayrı embeddingler alınarak karşılaştırma yapmak yerine, iki cümleyi aynı anda modelin içerisine vererek aradaki benzerliğin hesaplanmasını sağlar. 

Cross Encoders modelleri genellikle daha başarılı sonuçlar vermektedirler fakat dokümanların ayrı olarak temsil edilmesini sağlayamadıkları için oldukça yavaş çalışırlar. Bu sebeple genellikle az sayıda dokümanın bulunduğu durumlarda kullanılırlar, ilerleyen bölümlerde bu modellerin hangi aşamada kullanılabileceğini göreceğiz.

## ***TODO:*** CE FLOW ORNEGI BIR FOTO.

## 3. Anlamsal Benzerlik
Şimdiye kadar dokümanları nasıl temsil edeceğimizden bahsettik, şimdi ise dokümanların temsillerini kullanarak aralarındaki benzerliği nasıl hesaplayacağımızdan bahsedeceğiz. Benzerlik dediğimiz şey matematiksel olarak vektörler arasındaki mesafeyi hesaplamak demektir. Vektörler arasındaki mesafe ne kadar küçükse dokümanlar arasındaki benzerlik o kadar yüksek kabul edilir.
Benzerlik hesaplamak için kullanılan birçok yöntem bulunmaktadır, bunlardan bazıları:

* Cosine Similarity
* Euclidean Distance
* Manhattan Distance
> Burada anlatılan bütün formüllerin kodlarına [buradan](./distances.py) erişebilirsiniz.

## ***TODO:*** Vektör Uzayı foto

Bu düşüncenin altında yatan en büyük düşünce ise şudur: embedding hesaplaması sırasında kelimelerin anlamları temsil edildiği için, birbirine benzer anlamda olan metinlerin vektörleri de birbirine benzer olacaktır yani birbirine çok yakın olacaktır.


### Cosine Similarity
Cosine Similarity, iki vektör arasındaki açıyı hesaplayarak benzerlik hesaplar. Cosine Similarity değeri -1 ile 1 arasında değişir, -1 tamamen zıt iken 1 tamamen benzer anlamına gelir. Cosine Similarity hesaplamak için aşağıdaki formül kullanılır:

## TODO: COSINE SIMILARITY FORMULU
Benzerlik hesaplamaları için en sık kullanılan formul olan Kosinüs benzerliğini Python ile hesaplaması oldukça kolaydır:
```Python
import numpy as np
embed1 = np.array([1, 2, 3]) # Vektor1
embed2 = np.array([4, 5, 6]) # Vektor2
dot_product = np.dot(embed1, embed2)
norm_embed1 = np.linalg.norm(embed1)
norm_embed2 = np.linalg.norm(embed2)
cosine = dot_product / (norm_embed1 * norm_embed2)
```

Hiçbir harici kütüphane kullanmadan aşağıdaki kod ile de Kosinüs benzerliği hesabı yapabilirsiniz:
```Python
def dot_scratch(embed1, embed2):
    """dot product of two vectors is the sum of element wise multiplication of vectors.
    """
    dot = sum(i*j for i,j in zip(embed1, embed2))
    return dot

def norm_scratch(embed):
    """norm of magnitude. Square root of the sum of the square of each item.
    1. take square of each item in vector.
    2. take sum of the results..
    3. take square root of the sum."""
    return sum(i**2 for i in embed) ** 0.5

def calculate_cosine_scratch(embed1, embed2):
    """cosine similarity is calculating by:
    1. calculate dot product between embeddings
    2. calculate norm of each embed
    3. divine dot product to multiplication of norms.
    dot(embed1, embed2) / (norm(embed1) * norm(embed2))
    """
    cosine = dot_scratch(embed1, embed2) / (norm_scratch(embed1) * norm_scratch(embed2))
    return cosine
```

Temel olarak *dot_product* dediğimiz şey iki vektörün elemanlarının çarpımının toplamıdır. *norm* dediğimiz şey ise her bir elemanın karesinin toplamının kareköküdür. Bu iki değeri kullanarak Cosine Similarity hesaplanır. Temelde formül şu şekilde ifade edilebilir:

> dot(embed1, embed2) / (norm(embed1) * norm(embed2))


### Euclidean Distance
Euclidean distance aslında hepimizin liseden bildiği Pisagor teoremidir. İki vektör arasında kalan doğru parçasına verilen isimdir. Numpy vektörler üzerinde toplama/çıkartma gibi işlemleri kolaylıkla yaptığı için bu hesaplamayı numpy ile yapmak oldukça kolaydır.

```Python
distance = np.linalg.norm(embed1 - embed2)
```

Cosine kadar yaygın olmasa da sık kullanılan yöntemlerden birisidir, çoğu vektör veritabanı Euclidean Distance'ı desteklemektedir. Formül temelde vektörlerin elemanlarının farkının karelerinin toplanması ve ardından karekökünün alınması ile hesaplanmaktadır. Numpy kullanmadan şu şekilde hesaplaması yapılabilir:

```Python
distance = sum([(i-j)**2 for i,j in zip(embed1, embed2)]) ** 0.5
```


### Manhattan Distance


