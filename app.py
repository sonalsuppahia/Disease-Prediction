import streamlit as st
from keras.models import load_model
import json
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Function to get base64 encoded image
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Inject CSS with base64 background image, title styling, sidebar text color, and custom text color
background_image_path = 'img3.jpg'  # Update with your image path
base64_image = get_base64_encoded_image(background_image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
    }}
    .back-arrow {{
            position: absolute;
            top: -45px;
            left: -260px;
            z-index: 1000;
            border-radius: 50%; /* Round shape */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Shadow effect */
        }}
        .back-arrow img {{
            width: 45px; /* Width of the back arrow icon */
            height: 45px; /* Height of the back arrow icon */
        }}
    .result-box {{
        border: 2px solid #000;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }}
    .title {{
        color: black;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        font-family: Georgia;
    }}
    .custom-text {{
        color: white; /* White color for the custom text */
        font-size: 18px;
        text-align: center;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.2);  /* Black with 20% transparency */
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: white; /* Sidebar header text color */
    }}
    [data-testid="stSidebar"] .css-1b1w7xw, [data-testid="stSidebar"] .css-1b1w7xw input {{
        color: white; /* File uploader text color */
    }}
    </style>
    """,
    unsafe_allow_html=True
)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
back_arrow_image_path = 'AL4.png'

# Encode the icon images
encoded_back_arrow = encode_image_to_base64(back_arrow_image_path)

st.markdown(f'''
<div class="back-arrow">
<a href="http://localhost:8501">
<img src="data:image/png;base64,{encoded_back_arrow}" alt="Back Arrow">
</a>
</div>
''', unsafe_allow_html=True)
# Load models
model_data = load_model("my_keras_model.h5", compile=True)
auth_model = load_model("auth_model.h5", compile=True)

# Load disease information from JSON file
with open("ubaids.json") as f:
    disease_info = json.load(f)
    disease_info = list(disease_info.items())

# Load example images
example_image_paths = ["spe.jpg", "might.jpg"]
example_images = [Image.open(path) for path in example_image_paths]

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array.reshape(1, 224, 224, 3)
    return img_array

def predict_disease(image):
    preprocessed_image = preprocess_image(image)
    auth_result = auth_model.predict(preprocessed_image).argmax()
    if auth_result == 0:
        disease_index = model_data.predict(preprocessed_image).argmax()
        key, observed_disease = disease_info[disease_index]
        disease_name = key
        description = observed_disease["description"]
        symptoms = observed_disease["symptoms"]
        causes = observed_disease["causes"]
        treatment = observed_disease["treatment"]
        suggested_pesticides = ", ".join(observed_disease.get("suggested_pesticides", [])) if treatment.lower() not in ["none", ""] else "N/A"
        return disease_name, description, symptoms, causes, treatment, suggested_pesticides
    else:
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

# Streamlit app
def main():
    st.markdown("<h1 class='title'>Plant Leaf Disease Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='custom-text'>Upload an image of a plant leaf to identify potential diseases and receive treatment and pesticide suggestions.</p>", unsafe_allow_html=True)

    # Sidebar for image upload
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Sidebar for example images
    st.sidebar.header("Example Images")
    cols = st.sidebar.columns(len(example_images))
    example_image_selected = None
    for i, (col, example_image) in enumerate(zip(cols, example_images)):
        with col:
            st.image(example_image, caption=f"Example {i+1}")
            if st.button(f"Load Example {i+1}", key=f"example_{i+1}"):
                # Set the selected example image to be processed
                example_image_selected = example_image

    image = None
    if uploaded_file is not None:
        image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    elif example_image_selected is not None:
        # Convert the example image to bytes and read it into OpenCV
        buffered = io.BytesIO()
        example_image_selected.save(buffered, format="JPEG")
        image = np.asarray(bytearray(buffered.getvalue()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is not None:
        st.image(image, channels="BGR", caption="Uploaded or Example Image")
        disease_name, description, symptoms, causes, treatment, suggested_pesticides = predict_disease(image)
        st.subheader("Results")
        st.markdown(f"<div class='result-box'><b>Disease Name:</b> {disease_name}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'><b>Description:</b> {description}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'><b>Symptoms:</b> {symptoms}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'><b>Causes:</b> {causes}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'><b>Treatment:</b> {treatment}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'><b>Suggested Pesticides:</b> {suggested_pesticides}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
