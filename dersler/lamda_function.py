'''

Python'da lambda fonksiyonu, tek satırlık, kısa ve basit işlemler yapmak için kullanılan anonim
 (ismi olmayan) fonksiyonlardır. Lambda fonksiyonları, özellikle bir fonksiyonu bir kerelik ve 
 hızlıca tanımlayıp kullanmak istediğinizde kullanışlıdır.

Temel Söz Dizimi

#!lambda parametre1, parametre2, ... : ifade
#!lambda: Anonim fonksiyon oluşturmak için kullanılan anahtar kelime.
#!parametre1, parametre2, ...: Fonksiyonun alacağı giriş parametreleri. Normal bir fonksiyondaki gibi olabilir.
#!ifade: Lambda fonksiyonunun döndüreceği tek bir ifade. Birden fazla ifade içeremez.


'''


fonksiyon=lambda x:(x*2)

print(fonksiyon(5)) 


print("-----------------")

toplafonksiyon=lambda x1,x2:(x1+x2)
print(toplafonksiyon(3,4))




