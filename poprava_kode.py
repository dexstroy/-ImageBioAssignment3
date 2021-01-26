import matplotlib.pyplot as plt
import cv2, os
import numpy as np

cascadeFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Zaznava lokacije obraza na sliki
def detectFace(img):
    detectionList = cascadeFace.detectMultiScale(img, 1.05, 5)
    return detectionList


# Local binary patterns
def izracunaj_LBP(slika):
    print("calculating")
    # pretvorba slike v crnobelo
    cb_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

    # prazna slika istih velikosti kot cb_slika
    LBP_slika = np.zeros_like(cb_slika)

    # velikost matrike okoli centralne slikovne tocke
    velikost = 3

    # sprehodim se skozi vse slikovne pike v sliki in racunam LBP
    for vrstica in range(0, slika.shape[0] - velikost):
        for stolpec in range(0, slika.shape[1] - velikost):
            # dobim matriko 3x3
            sosedi = cb_slika[vrstica:vrstica + velikost, stolpec:stolpec + velikost]
            # centralna slikovna pika
            center = sosedi[1,1]

            # primerjam z centralnim pikslom vse elemente
            sosedi_primerjanje = (sosedi >= center) * 1.0

            # matriko pretvorim v vektor
            sosedi_primerjanje_v = sosedi_primerjanje.T.flatten()

            # iz vektorja odstranim centralni piksel
            sosedi_primerjanje_v = np.delete(sosedi_primerjanje_v, 4)

            # izracunam stevilo
            sosedi_primerjanje_v = np.flip(sosedi_primerjanje_v)
            stevilo = 0
            for i in range(0, len(sosedi_primerjanje_v)):
                if sosedi_primerjanje_v[i] >= 1:
                    stevilo += 2**i

            # izracunano stevilo shranim v novo sliko
            LBP_slika[vrstica + 1, stolpec + 1] = stevilo

    # izračunam in vrnem histogram
    feature = LBP_slika.flatten()
    histogram = np.histogram(feature, np.arange(256))[0]
    return histogram / np.sum(histogram)


# naloga
# vsebuje slike osebe, ki jo iščemo v testni množici
train_dir = './train'
# vsebuje osebe za testiraje iskanja
test_dir = './test'

train_persons = os.listdir(train_dir)

for i in range(len(train_persons)):
    train_persons[i] = train_persons[i].split("-")[0]

train_persons_histograms = []

for person in train_persons:
    person_image = cv2.imread(train_dir + "/" + person + '-4.png')
    face_locations = detectFace(person_image)
    x, y, w, h = face_locations[0]
    person_image = person_image[y:y+h, x:x+w]

    histogram = izracunaj_LBP(person_image)
    train_persons_histograms.append(histogram)


# testne slike
# dobi slike nad katerimi se bo izvajala klasifikacija
test_images_names = os.listdir(test_dir)

test_persons_histograms = []
# pretvori obraze na sliki v vektor obraznih karakteristik
for test_images_name in test_images_names:
    person_image = cv2.imread(test_dir + "/" + test_images_name)

    face_locations = detectFace(person_image)
    x, y, w, h = face_locations[0]
    person_image = person_image[y:y + h, x:x + w]

    histogram = izracunaj_LBP(person_image)
    test_persons_histograms.append(histogram)



print("Razdalje: ")
for i in range(len(test_images_names)):
    dist = np.linalg.norm(train_persons_histograms[0] - test_persons_histograms[i])
    print(test_images_names[i] + ": " + str(dist))

