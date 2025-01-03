# Import required libraries
import PIL
import streamlit as st
from ultralytics import YOLO
import base64

# Replace the relative path to your weight file
model_path = 'best-final.pt'

# Setting page layout
st.set_page_config(
    page_title="Deteksi Penyakit Daun Tanaman Kentang",  # Setting page title
    page_icon="🌿",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Add background CSS with dim effect for the main page and sidebar
def add_bg_with_dim(image_path):
    """
    Adds a background image with a dim effect using CSS.
    
    Args:
    - image_path (str): Path to the local image file.
    """
    try:
        with open(image_path, "rb") as file:
            encoded_image = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            /* Background for the entire app */
            .stApp {{
                background: linear-gradient(
                    rgba(0, 0, 0, 0.9), 
                    rgba(0, 0, 0, 0.9)
                ),
                url("data:image/jpeg;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            /* Background for the sidebar */
            .css-1d391kg {{
                background: linear-gradient(
                    rgba(0, 0, 0, 0.5),
                    rgba(0, 0, 0, 0.5)
                ),
                url("data:image/jpeg;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            /* Optional: Sidebar text color to ensure contrast */
            .css-1d391kg .sidebar .sidebar-content {{
                color: white;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"Background image not found at {image_path}. Please check the file path.")

# Path to your local background image
add_bg_with_dim("background2.jpeg")  # Replace with your local image file path

# Creating sidebar
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 60)) / 100

# Creating main page heading
st.title("Deteksi Penyakit Daun Tanaman Kentang")
st.caption('Unggah gambar yang ingin dideteksi')
st.caption('Lalu Klik :blue[Detect Objects] button and cek hasil')

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_container_width=True
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image,
                        conf=confidence
                        )
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Detected Image',
                 use_container_width=True
                 )
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")
