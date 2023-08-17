working directory böyle olması lazım kodun directory'si ona göre ayarlı tabiki sizin repo'yu nereye clone'ladığınıza göre vs. değişir, 
![Adsız](https://github.com/volkannnv/Face_Matcher/assets/127948297/2d93bc1c-68af-4025-9ab2-01680e334564) repo'nun içinde repo'yla aynı Face_Matcher klasörü var. scriptler de onun içinde


images klasöründe benim fotoğraflatım var, kendi fotoğraflarınızla denemek isterseniz ister fotoğrafları images klasörüne atıp isimlerini extract ve taken olarak değiştirin ya da fotoğrafı atıp kodun içindeki dosya ismini değiştirin.

dlib ile çalışan 2 ayrı script var, biri face_recognition kullanıyor. diğeri ise pre-trained model kullanıyor. pre-trained model kullanan Face_Matcher_dlib2 daha hızlı

mtcnn kullanan scripti ilk kullanmaya çalıştığınızda pretrained modeli indiriyor 100mb civarı bir dosya. ben ilk denediğimde indirdikten sonra kod hata verdiği için şu anki haliyle çıktı verir mi bilmiyorum ama vermezse tekrar çalıştırınca verir diye umuyorum
