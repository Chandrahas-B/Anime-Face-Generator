import streamlit as st
import tensorflow as tf


st.set_page_config(layout="centered")
st.title("Anime Generator")

st.markdown('<style>body{margin:0 auto;text-align:center;}</style>', unsafe_allow_html=True)


model_name = 'gen.hdf5'

latent_vec = 300


@st.cache_resource
def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    print("Model loaded")
    model.summary()
    return model

model = load_model(model_name)

def unnormalize(image):
    image = (image+1)*127.5
    image = image/255
    return image

def generate():
    random_vec = tf.random.normal((1, latent_vec))
    normalized_image = model.predict(random_vec, verbose= 0)
    normalized_image = tf.keras.layers.UpSampling2D(size=(2, 2))(normalized_image).numpy()
    image = unnormalize(normalized_image)
    return image

def col_images():
    st.markdown("""
    <style>
    div{
        gap: 0rem;
    }
    </style>
    """,unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        image = generate()
        st.image(image)
    with col2:
        image = generate()
        st.image(image)
    with col3:
        image = generate()
        st.image(image)


def images():
    with st.spinner('Image is being generated'):
        st.markdown(
            """
            <style>
                button[title^=Exit]+div [data-testid=stImage]{
                    text-align: center;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 100%;
                }
            </style>
            """, unsafe_allow_html=True
        )

    rows = st.columns(3)
    for _ in rows:
        col_images()
        

    
if st.button("Generate Images"):
    images()