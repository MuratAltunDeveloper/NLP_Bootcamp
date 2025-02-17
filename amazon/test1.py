# Toplama fonksiyonu
def toplama(a, b):
    return a + b

# Çarpma fonksiyonu
def carpma(a, b):
    return a * b

# Ana fonksiyon: İçinde toplama ve çarpma fonksiyonlarını çağırıyor
def hesapla(x, y):
    toplam = toplama(x, y)  # toplama fonksiyonunu çağır
    carpim = carpma(x, y)   # carpma fonksiyonunu çağır
    return toplam, carpim   # İki sonucu döndür

# Fonksiyonu çağırıp sonucu yazdıralım
sonuc_toplam, sonuc_carpim = hesapla(4, 5)
print("Toplam:", sonuc_toplam)
print("Çarpım:", sonuc_carpim)
