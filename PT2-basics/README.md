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




