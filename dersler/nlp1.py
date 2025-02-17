#IBM TEKNOLOJİ YOUTUBE,MEDİUM,MİUL,simplilearn  vb.

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from colorama import Fore, Style

print(Fore.RED + 'Bu kırmızı bir yazıdır.' + Style.RESET_ALL)
print(Fore.GREEN + 'Bu yeşil bir yazıdır.' + Style.RESET_ALL)
print(Fore.YELLOW + 'Bu sarı bir yazıdır.' + Style.RESET_ALL)


##################################################
# !1. Text Preprocessing
##################################################
import pandas  as pd
df=pd.read_csv(r"C:\Users\murat\OneDrive\Masaüstü\NLP\dersler\datasets\amazon_reviews.csv")
print(df.head(10)["reviewText"])


print("****************\n")


df["reviewText"]=df["reviewText"].str.lower()#bütün harfler küçültülüyor


#! Punctuations  ( noktalama işaretlerini kaldırma)

df["reviewText"] = df["reviewText"].str.replace('[^\w\s]', '', regex=True)



# ! Numbers  kaldırma 
df['reviewText'] = df['reviewText'].str.replace('\d', '')




#!stopwords

import nltk
from nltk.corpus import stopwords

# Stopword'leri indirin
nltk.download('stopwords')

# İngilizce stopword'leri alın
sw = stopwords.words('english')

# DataFrame'deki 'reviewText' sütunundaki stopword'leri kaldırın
df['reviewText'] = df['reviewText'].apply(
    lambda x: " ".join(word for word in str(x).split() if word.lower() not in sw)
)



'''
Diyelim ki, iki cümlemiz var:

"Bu ürün gerçekten çok iyi."
"Ürün çok kötü."
Eğer stopword'leri kaldırırsak:

İlk cümle: "ürün gerçekten iyi"
İkinci cümle: "ürün kötü"
Burada "bu", "gerçekten" ve "çok" gibi kelimeler çıkarıldığında, cümlenin özeti hala korunmuş olur.
Ancak, durumu daha net bir şekilde görmek için sadece önemli kelimelere odaklanmış oluruz.

Özet
Stopword'ler, genellikle metinlerin anlamını belirgin bir şekilde değiştirmediği için çıkarılırlar.
Bu, analiz süreçlerinin daha verimli ve etkili olmasına yardımcı olur. Ancak, bazı durumlarda,
bağlama bağlı olarak bazı stopword'lerin de anlam taşıyabileceğini unutmamak gerekir.
Bu nedenle, stopword'lerin çıkarılması kararı, spesifik bir uygulamanın ihtiyaçlarına göre verilmelidir.

'''



###############################
#! Rarewords
###############################

'''
Doğal dil işleme (NLP) alanında "rare words" (nadir kelimeler),
belirli bir metin koleksiyonunda veya veri kümesinde çok az kez karşılaşılan kelimeleri ifade eder.
Bu kelimeler genellikle aşağıdaki özelliklere sahiptir:

1. Tanım
Nadir Kelimeler: Belirli bir metin setinde yalnızca birkaç kez veya çok az sayıda geçen kelimelerdir. 
Örneğin, "quasar" veya "cryptocurrency" gibi teknik terimler, genel metinlerde nadir kelimeler olabilir.
2. Özellikleri
Düşük Frekans: Nadir kelimeler, metin içinde düşük frekansla görülür. Genellikle toplam 
kelime sayısının %1'inden daha azını temsil ederler.
Anlam Derinliği: Nadir kelimeler, belirli bir bağlamda derin anlam taşıyabilir.
Ancak, genel dil kullanımında pek yaygın olmadıkları için silebiliriz.




'''
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()



drops = temp_df[temp_df <= 1]

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))






# !Tokenization
'''
Tokenization, doğal dil işleminde metni daha küçük parçalara (token'lara) ayırma sürecidir. 
Bu parçalar genellikle kelimeler, cümleler veya karakterler olabilir. Tokenization, metin analizi,
duygu analizi veya makine öğrenimi modellerinin eğitimi gibi birçok uygulama için temel bir adımdır. 

### Örnek

Metin: "Doğal dil işleme çok ilginçtir!"

Tokenization sonrası: `["Doğal", "dil", "işleme", "çok", "ilginçtir", "!"]` 

Bu örnekte, cümle kelimelere ve bir noktalama işaretine ayrılmıştır.



'''
import nltk
from textblob import TextBlob

# NLTK punkt tokenizer'ını indirin (bir kez çalıştırılması yeterlidir)

#nltk.download("punkt")
#nltk.download("punkt_tab")


# Kelimeleri ayırma
df['reviewWords'] = df["reviewText"].apply(lambda x: TextBlob(x).words)


###############################
#! Lemmatization
###############################

'''


Lemmatization, doğal dil işleme (NLP) alanında kelimelerin kök veya temel 
formuna dönüştürülmesi işlemidir. Bu işlem, kelimenin dilbilgisel ve anlam 
açısından doğru bir şekilde kök haline getirilmesini sağlar. 

### Özellikleri

1. **Anlam Derinliği**: Lemmatization, kelimenin anlamını koruyarak kök formunu bulur.
 Örneğin, "running" kelimesi "run" şeklinde lemmatize edilir.
2. **Dilbilgisel Bilgi**: Lemmatization, kelimenin dilbilgisel özelliklerini (özne, yüklem gibi) dikkate alır.
 Bu nedenle, kelimenin hangi formda olduğunu anlamak için daha fazla bilgi gerektirir.
3. **Kaynaklar**: Lemmatization genellikle dilbilgisi kuralları ve kelime listeleri 
(sözlükler) kullanılarak gerçekleştirilir.

### Örnekler

- "better" → "good"
- "went" → "go"
- "studies" → "study"

### Kullanım

Lemmatization, metin analizi, bilgi erişimi, makine 
öğrenimi ve diğer birçok NLP uygulamasında kelimelerin standartlaştırılması için önemlidir.
 Bu sayede, kelimelerin farklı formları arasında daha iyi bir ilişki kurulabilir ve analizler daha anlamlı hale gelir.
'''

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Gerekli kaynakları indirin (bir kez çalıştırılması yeterlidir)
#nltk.download('wordnet')
#nltk.download('punkt')

# Lemmatizer'ı oluştur
lemmatizer = WordNetLemmatizer()

# Cümleleri lemmatize etme
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))


# İlk 5 sonucu göster
print(df['reviewWords'].head())



#!      metin görşelleştirme=>bu bana metin ve ne üzerine yoğunlaştığı hakkında fikir verir


'''
kelime_frekans=df['reviewText'].apply(lambda x:pd.value_counts(x.split(" "))).sum(axis=0).reset_index()  
kelime_frekans.columns=['kelime','frekans'] 
kelime_frekans=kelime_frekans.sort_values(by='frekans',ascending=False)


print(f"kelime frekans :{kelime_frekans}")


import matplotlib.pyplot as plt

# Frekansı 1000'in üzerinde olan kelimeleri filtrele
frekans_1000_ustu = kelime_frekans[kelime_frekans['frekans'] > 1000]

# Bar grafiği oluştur
plt.figure(figsize=(12, 6))  # Görselin boyutunu ayarla
plt.bar(frekans_1000_ustu['kelime'], frekans_1000_ustu['frekans'], color='skyblue')

# Grafiği etiketle
plt.title('Frekansı 1000 Üzerindeki Kelimeler', fontsize=16)
plt.xlabel('Kelime', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.xticks(rotation=45, ha='right')  # X ekseni etiketlerini döndür
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Grafiği göster
plt.tight_layout()
plt.show()


'''
               #!!!     SENTİMENT MODELLEMESİ==>DUYGU (POZİTİF,NEGATİF Mİ,NÖTR VB.)  BULMA
               #NEVER,HİÇ,ASLA,OLMAZ =>NEGATİF ANLAM TAŞIR


import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer


sia = SentimentIntensityAnalyzer()
print(f"yorum:{sia.polarity_scores("this is the worst and shit animal")}")

#Compound değeri, genel duygu durumunu özetler. -1 ile 1 arasında bir değer alır ve -1 olumsuz, 1 olumlu durumdur ,genelde bunu kullanırız.

# Her reviewText için compound değerini al
her_reviewText_puani = df['reviewText'].apply(lambda x: sia.polarity_scores(x)['compound'])

print(her_reviewText_puani)

df["polarity_score"]=df['reviewText'].apply(lambda x: sia.polarity_scores(x)['compound'])

print(df.columns.tolist())

print(Fore.RED + f'overall:{df["overall"]}.' + Style.RESET_ALL)
#overall (ürün  puanı)  şimdi altta düşük puanlı ama yorumu iyi vermiş(polarity_score 0.9 büyük) kişilere eriş ve yanlış puanmı vermişler bul

# overall'ı 3'ten büyük ve polarity_score 0.90 ten büyük  içeren yorumları seç
filtered_df = df.loc[(df['overall'] > 4) & (df['polarity_score']>0.9)]
print(filtered_df.count)




#!sentiment modelling veriden model çıkarımı




# Sentiment label'larını oluşturma
df["sentiment_label_str"] = df['polarity_score'].apply(lambda x: 'pos' if x > 0 else 'neg')
df["sentiment_label"] = LabelEncoder().fit_transform(df['sentiment_label_str'])



print(df['sentiment_label'])


y=df['sentiment_label']
X=df["reviewText"]
#!   COUNT VECTORS İLE  x (reviewText)  makine öğrenmesi için  uygun matematiksel hale getiririz.
'''
Doğal dil işleme (NLP) alanında metinleri matematiksel hale çevirme işlemleri,
 genellikle kelimelerin veya cümlelerin sayısal temsillerini oluşturmayı içerir. 
 İşte bu sürecin bazı ana yöntemleri:

1. Count Vectors (Frekans Temsilleri)
Tanım: Metindeki her kelimenin frekansını sayarak oluşturulan basit bir temsil.
Yöntem: Her kelime, metindeki toplam kelime sayısına bölünerek normalize edilebilir.
Bu, kelimelerin daha anlamlı bir temsiline olanak tanır.


2. TF-IDF (Term Frequency-Inverse Document Frequency)
Tanım: Kelimenin bir belgede ne sıklıkla geçtiğini (TF) ve bu kelimenin tüm belgelerde 
ne kadar yaygın olduğunu (IDF) dikkate alarak oluşturulan bir temsil.

Yöntem:
TF: Belgedeki kelime sayısı / Toplam kelime sayısı.
IDF: log(Total Belgeler / (1 + Kelimeyi İçeren Belgeler)).
Sonuç olarak, TF ve IDF çarpılarak kelimenin önem derecesi belirlenir.



3. Word Embeddings
Tanım: Kelimeleri çok boyutlu bir uzayda temsil eden vektörlerdir. Kelimeler arasındaki 
benzerlikleri ve ilişkileri daha iyi yakalar.
Yöntemler:
Word2Vec: Kelimeleri belirli bir bağlamda temsil eder. Çalışma prensibi, kelimelerin
komşuluk ilişkilerini öğrenmektir.

GloVe: Global kelime istatistiklerini kullanarak kelimelerin anlamını temsil eder.
Tüm belgelerdeki kelimelerin frekanslarını dikkate alır.

BERT: Bağlamı dikkate alarak kelimeleri temsil eder. Önceden eğitilmiş bir modeldir ve 
cümlelerin anlamını daha iyi kavrar.



'''




# words
# kelimelerin nümerik temsilleri

# characters
# karakterlerin numerik temsilleri

# ngram
a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını gösterir ve feature üretmek için kullanılır"""

print(TextBlob(a).ngrams(3))

#?count vectors===>
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']

# word frekans
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)

vectorizer.get_feature_names_out()
print(Fore.GREEN + f'{vectorizer.get_feature_names_out()}' + Style.RESET_ALL)
X_c.toarray()

print(Fore.GREEN + f'{X_c.toarray()}' + Style.RESET_ALL)



print("--------------------------------------")


# n-gram frekans
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out()
print(f" {vectorizer2.get_feature_names_out()}----------------------------------------------")


X_n.toarray()
print(X_n.toarray())

print(f"*******{df['reviewText'][10:15]}")
X=df["reviewText"]
#bizim datasetimizde count vectorizer 
# word frekans
# Count Vectorizer
vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)
'''
Bu, CountVectorizer'ın metinlerde tanıdığı kelimeleri içerir. Eğer burada beklemediğiniz kelimeler varsa, bu büyük ihtimalle veri setinizdeki belirli bir kelime öbeğinin veya ifadenin sayılmasından kaynaklanıyor.

'''
# Özellik adlarını yazdır
print(Fore.GREEN + f'{vectorizer.get_feature_names_out()}' + Style.RESET_ALL)

# Vektörlerin dizisini yazdır
print(Fore.GREEN + f'{X_count.toarray()}' + Style.RESET_ALL)





#!!!                            TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

# İlk belgenin vektör temsili
print(f"****{X_tf_idf_word[0]}")

# Dense (yoğun) formatta göstermek içi=>İlk belgenin tüm özellikler (kelimeler) için TF-IDF değerlerini gösterir.
print("\nDense format:")
print(X_tf_idf_word[0].toarray())




tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)







#!!  sentiment  modelling machine learning 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate



lg=LogisticRegression()
Lr_model=lg.fit(X_tf_idf_word,y)

cross_val_score=cross_val_score(Lr_model,X_tf_idf_word,y,scoring="accuracy",cv=5).mean()

print(f"Cross Validation Score: {cross_val_score}")



yeniyorum=pd.Series("this is good product perfect")

yeniyorum2=pd.Series("bad shit noworking now brother")

tfyeniyorum=TfidfVectorizer().fit(X).transform(yeniyorum2)

print(f"yeni yorum duygu analizi:{Lr_model.predict(tfyeniyorum)}")


#!  RANDOM FOREST İLE MODELLEME
print(Fore.RED + '................................................' + Style.RESET_ALL)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Count Vectors
rf_model = RandomForestClassifier().fit(X_count, y)
count_cv_score = cross_val_score(rf_model, X_count, y, cv=5, n_jobs=-1).mean()

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
tfidf_word_cv_score = cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean()

# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
tfidf_ngram_cv_score = cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean()

# Sonuçları yazdırma
print("Count Vectors CV Score:", count_cv_score)
print("TF-IDF Word-Level CV Score:", tfidf_word_cv_score)
print("TF-IDF N-GRAM CV Score:", tfidf_ngram_cv_score)


#Hiperparametre optimizasyonu
rf_model = RandomForestClassifier(random_state=17)


f_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X_count, y)



rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count, y)


print(cross_val_score(rf_final, X_count, y, cv=5, n_jobs=-1).mean())


