import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import os

# vsebuje slike osebe, ki jo iščemo v testni množici
train_dir = './klas/train2'
# vsebuje osebe za testiraje iskanja
test_dir = './klas/test2'

persons = os.listdir(train_dir)

for i in range(len(persons)):
    persons[i] = persons[i].split("-")[0]

persons_encodings = []

# pretvori obraze na sliki v vektor obraznih karakteristik
for person in persons:
    print('encoding: ' + person)
    person_image = face_recognition.load_image_file(train_dir + "/" + person + '-1.png')
    person_image_encoding = face_recognition.face_encodings(person_image)[0]
    persons_encodings.append(person_image_encoding)

# dobi slike nad katerimi se bo izvajala klasifikacija
test_images_names = os.listdir(test_dir)

test_images_encodings = []
# pretvori obraze na sliki v vektor obraznih karakteristik
for test_images_name in test_images_names:
    print('encoding: ' + test_images_name)
    test_image = face_recognition.load_image_file(test_dir + "/" + test_images_name)
    test_image_encoding = face_recognition.face_encodings(test_image)
    test_images_encodings.append(test_image_encoding)

# klasifikacija, če uvrsti prav je true, drugača false
for i in range(len(test_images_names)):
    if len(test_images_encodings[i]) != 0:
        # dobim listo razdalj ki mi povejo za koliko se obraza razlikujeta
        distances = face_recognition.face_distance(persons_encodings, test_images_encodings[i][0])
        # preveri ali je prišlo do ujemanja, glede na ime datoteke
        # najde najkrajšo razdaljo in preveri ali se ujema s pravo osebo
        mached = str(persons[np.argmin(distances)] == test_images_names[i].split("-")[0])
        
        minimal_distance = distances[np.argmin(distances)]
        # Ce je najmanjsa razdalja vecja od 0.7 potem recem da se ne ujemata
        if minimal_distance > 0.70:
            mached = "False"
        print(test_images_names[i] + ': ' + mached)
    # v primeru da na fotografiji ne najde obraza
    else:
        print(test_images_names[i] + ': False')