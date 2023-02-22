import streamlit as st
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model

kamus_wajah = {0:'Anak Perempuan',1:'Anak Laki-laki',2:'Pria Dewasa',3:'Perempuan Dewasa'}
kamus_outfit = {0:'Outfit Batik',1:'Outfit Bumi',2:'Outfit Kue',3:'Outfit Mamba',4:"Outfit Wedding"}
kamus = {0:'batik.txt',1:'bumi.txt',2:'kue.txt',3:'mamba.txt',4:'wedding.txt'}

model_wajah = load_model('model/face_recognition_0.h5')
model_anak_lk = load_model('model/model_anak_laki.h5')
model_anak_pr = load_model('model/model_anak_pr.h5')
model_dewasa_lk = load_model('model/model_dewasa_laki.h5')
model_dewasa_pr = load_model('model/model_dewasa_pr.h5')

st.set_page_config(page_title="Rekomendasi Outfit")

image_ex = Image.open('assets/fixed.png')
st.title("""Rekomendasi Outfit Berdasarkan Citra Tubuh Manusia""")
st.write("Sistem ini akan menghasilkan rekomendasi outfit yang cocok berdasarkan citra tubuh manusia dengan algoritma Convolutional Neural Network")
col1, col2, col3 = st.columns(3)
col2.image(image_ex, caption="Contoh Input Gambar")

def prediksi_outfit(image, model):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(126,224))/255
    x_test = []
    x_test.append(image)
    x_test = np.array(x_test)
    pred = model.predict(x_test)
    pred = np.argmax(pred,axis=1)
    pred_outfit = [kamus_outfit[i] for i in pred]
    return pred_outfit


def predik_wajah(gbr):
    img_1 = cv2.imread(gbr)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_1,(126,224))
    cropped_image = img[0:126, 0:126]
    cv2.imwrite('assets/crop_face.jpg', cropped_image)
    cropped_image_resize = cv2.resize(cropped_image,(64,64))/255
    x_test = []
    x_test.append(cropped_image_resize)
    x_test = np.array(x_test)
    y_pred = model_wajah.predict(x_test)
    y_pred = np.argmax(y_pred,axis=1)
    pred_wajah = [kamus_wajah[i] for i in y_pred]
    ket = [kamus[i] for i in y_pred]

    if pred_wajah == ['Anak Laki-laki']:
        prediksi_outfit(img_1, model_anak_lk)
        contoh = prediksi_outfit[0] + '-1.jpg'
    elif pred_wajah == ['Anak Perempuan']:
        prediksi_outfit(img_1, model_anak_pr)
        contoh = prediksi_outfit[0] + '-2.jpg'
    elif pred_wajah == ['Pria Dewasa']:
        prediksi_outfit(img_1, model_dewasa_lk)
        contoh = prediksi_outfit[0] + '-1.jpg'
    elif pred_wajah == ['Perempuan Dewasa']:
        prediksi_outfit(img_1, model_dewasa_pr)
        contoh = prediksi_outfit[0] + '-2.jpg'       
    
    with open('assets/'+ket[0]) as f:
        contents = f.read()
    list_hasil = []
    list_hasil.append(predik_wajah[0])
    list_hasil.append(prediksi_outfit[0])
    list_hasil.append(contents)
    list_hasil.append(contoh)
    return list_hasil 

file = st.file_uploader('', type='jpg')
if file is not False:
    proses = predik_wajah(file)
    wajah = proses[0]
    outfit = proses[1]
    caption = proses[2]
    contoh_ = proses[3]
    st.header("Hasil Rekomendasi Outfit")
    cola, colb, colc = st.columns(3)
    img_ = Image.open('assets/crop_face.jpg')
    cola.image(img_, caption=wajah)
    img__ = Image.open(contoh_)
    colc.image(img__, caption=outfit)
    st.success(caption)
else:
    st.warning('HANYA BISA FOTO YANG BERFORMAT JPG')

