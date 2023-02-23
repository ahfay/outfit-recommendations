import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

kamus_wajah = {0:'Anak Perempuan',1:'Anak Laki-laki',2:'Pria Dewasa',3:'Perempuan Dewasa'}
kamus_outfit = {0:'Outfit Batik',1:'Outfit Bumi',2:'Outfit Kue',3:'Outfit Mamba',4:"Outfit Wedding"}
kamus = {0:'batik.txt',1:'bumi.txt',2:'kue.txt',3:'mamba.txt',4:'wedding.txt'}

model_wajah = load_model('model/face_recognition_0.h5')
model_anak_lk = load_model('model/model_outfit_anak_lk.h5')
model_anak_pr = load_model('model/model_outfit_anak_pr.h5')
model_dewasa_lk = load_model('model/model_outfit_dewasa_lk.h5')
model_dewasa_pr = load_model('model/model_outfit_dewasa_pr.h5')

st.set_page_config(page_title="Rekomendasi Outfit")

image_ex = Image.open('assets/fixed.png')
st.title("""Rekomendasi Outfit Berdasarkan Citra Tubuh Manusia""")
st.write("Sistem ini akan menghasilkan rekomendasi outfit yang cocok berdasarkan citra tubuh manusia dengan algoritma Convolutional Neural Network")
col1, col2, col3 = st.columns(3)
col2.image(image_ex, caption="Contoh Input Gambar")

def prediksi_outfit(image, model):
    # convert to grayscale
    image = image.convert("L")

    # resize image
    image = image.resize((126, 224))

    # normalize image
    image = np.array(image)/255.0

    # add batch dimension
    image = np.expand_dims(image, axis=0)

    # make prediction
    pred = model.predict(image)
    pred = np.argmax(pred,axis=1)
    pred_outfit = [kamus_outfit[i] for i in pred]

    return pred_outfit


def predik_wajah(gbr):
    # load image
    img_1 = Image.open(gbr)

    # convert color mode from BGR to RGB
    img_1 = img_1.convert("RGB")

    # resize image
    img = img_1.resize((126,224))

    # crop image
    cropped_image = img.crop((0, 0, 126, 126))

    # save cropped image
    cropped_image.save('assets/crop_face.jpg')

    # resize cropped image
    cropped_image_resize = cropped_image.resize((64, 64))

    # normalize image
    cropped_image_resize = np.array(cropped_image_resize)/255.0
    x_test = []
    x_test.append(cropped_image_resize)
    x_test = np.array(x_test)
    y_pred = model_wajah.predict(x_test)
    y_pred = np.argmax(y_pred,axis=1)
    pred_wajah = [kamus_wajah[i] for i in y_pred]
    ket = [kamus[i] for i in y_pred]

    if pred_wajah == ['Anak Laki-laki']:
        nama = prediksi_outfit(img_1, model_anak_lk)
        contoh = nama[0] + '-1.jpg'
    elif pred_wajah == ['Anak Perempuan']:
        nama = prediksi_outfit(img_1, model_anak_lk)
        contoh = nama[0]  + '-2.jpg'
    elif pred_wajah == ['Pria Dewasa']:
        nama = prediksi_outfit(img_1, model_anak_lk)
        contoh = nama[0]  + '-1.jpg'
    elif pred_wajah == ['Perempuan Dewasa']:
        nama = prediksi_outfit(img_1, model_anak_lk)
        contoh = nama[0] + '-2.jpg'       
    
    with open('assets/'+ket[0]) as f:
        contents = f.read()
    list_hasil = []
    list_hasil.append(predik_wajah[0])
    list_hasil.append(prediksi_outfit[0])
    list_hasil.append(contents)
    list_hasil.append(contoh)
    return list_hasil 

file = st.file_uploader('', type=[ "jpg"])
if file is not None:
    proses = predik_wajah(gbr=file)
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

