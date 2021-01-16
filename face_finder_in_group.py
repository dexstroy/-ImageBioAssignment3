import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

train_dir = './img/train'

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

# skupinska slika
group_image = face_recognition.load_image_file("./img/family_21.jpg")

# izloci obraze iz slike
faces = face_recognition.face_locations(group_image)

# Pretvorim v PIL image format
pil_image = Image.fromarray(group_image)

# Kreiram ImageDraw instanco
draw = ImageDraw.Draw(pil_image)

font = ImageFont.truetype("arial.ttf", 35)

for face in faces:
    top, right, bottom, left = face
    
    color = "red"
    selected_person = "Unknown"
    
    # izreze zaznani obraz iz slike
    face_image = group_image[top:bottom, left:right]

    # vektor obraznih karakteristik iz izlocene slike
    face_encoding = face_recognition.face_encodings(face_image)[0]
    
    # Izracuna razdalje z znamini obrazi
    distances = face_recognition.face_distance(persons_encodings, face_encoding)
    
    # najde najmanjso razdaljo
    minimal_distance = distances[np.argmin(distances)]
    
    # preveri ali je primerna za klasificiranje
    if minimal_distance < 0.70:
        color = "green"
        selected_person = persons[np.argmin(distances)]
    
    # narise kvadrat okoli zaznanega obraza
    draw.rectangle([(right, top), (left, bottom)], outline=color, width=10)
    # narise ime osebe pod kvadratom
    draw.text((left, bottom), selected_person, align ="left", font=font) 
pil_image.show()