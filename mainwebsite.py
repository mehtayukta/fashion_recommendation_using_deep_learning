import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pandas as pd
import tensorflow

# Cache used so that loading is faster
@st.cache_resource()
def load_data():
    DATASET_PATH = "images/"
    df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=5000)
    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# Store pickle file in variables for extracting features
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

#pick filenames stored in file
filenames = pickle.load(open('filenames.pkl', 'rb'))

# browse image stored in upload file
def save_uploaded_image(uploaded_image, image_number):
    try:
        with open(os.path.join(os.getcwd(), 'uploads', f'uploaded_image_{image_number}.jpg'), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image {image_number}: {e}")
        return False

# Extracting features of an image
def feature_extraction(image_path, model):
    img = image.load_img(image_path, target_size=(96, 96)) # we have resized our image in folder into 96 * 96
    image_array = image.img_to_array(img)
    dimension_expansion_image = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(dimension_expansion_image)
    results = model.predict(preprocessed_image).flatten()
    normalizartion_result = results / norm(results)
    return normalizartion_result

# Recommending images function
def recommendions(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Streamlit app
def app():
    st.title('Welcome, The Thought to Closet \n Your Personalized Fashion Recommender System')
    # Load an example image
    image_path = "image.jpg"
    image = Image.open(image_path)

# Display the image in Streamlit
    st.image(image, use_column_width=True)

    # Load data
    df = load_data()

    # Create a list of unique product names for selection
    products_list = df['masterCategory'].unique()

    # User selects a product based on different columns
    selected_product = st.selectbox('Select a product:', products_list)

    # Filter the DataFrame based on the selected product
    selected_products_df = df[df['masterCategory'] == selected_product]

    selected_products_df['year'] = selected_products_df['year'].astype(str).str.replace(',', '', regex=False).astype(int)


    # Select and rename specific columns
    column_mapping = {
    'masterCategory': 'Category',
    'subCategory': 'Sub Category',
    'productDisplayName' : 'Product Name',
    'articleType': 'Product Type',
    'baseColour': 'Base colour',
    'season': 'Season',
    'usage': 'Usage',
    'year': 'Year',
    'image' : 'image'
    

        }

    selected_products_df_subset = selected_products_df[['masterCategory', 'subCategory','productDisplayName', 'articleType','baseColour', 'season', 'usage', 'year','image']].rename(columns=column_mapping)

    selected_products_df_subset_img = selected_products_df[['masterCategory', 'subCategory','productDisplayName', 'articleType','baseColour', 'season', 'usage', 'year']].rename(columns=column_mapping)
    
    


# Display the subset of the DataFrame
    


    # Display the selected product information
    st.write(f"Selected Product: {selected_product}")
    st.write(selected_products_df_subset_img)

    # Display top 10 images for the selected product
    st.header(f"Top 10 products based on {selected_product}")

    # Limit to top 10 images for display
    top_10_images_df = selected_products_df_subset.head(10)


    # Create a combo box with the top 10 images
    selected_image = st.selectbox("Select an image:", ["All"] + top_10_images_df['Product Name'].tolist())

    # Check if an image is selected from the combo box
    if selected_image != "All":
        # Display only the selected image
        selected_image_row = top_10_images_df[top_10_images_df['Product Name'] == selected_image]
        image_path = os.path.join("images", selected_image_row.iloc[0]['image'])
        try:
            product_image = Image.open(image_path)
            st.image(product_image.resize((100, 100)), caption=selected_image_row.iloc[0]['Product Name'])
            model = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
            model.trainable = False
            model = tensorflow.keras.Sequential([
                model,
                GlobalMaxPooling2D()
            ])

            features1 = feature_extraction(os.path.join(os.getcwd(), image_path), model)
            indices1 = recommendions(features1, feature_list)

            

            # Display recommendation images
            st.header("Closest Match Similar Products")
            cols = st.columns(5)
            col1, col2, col3, col4, col5 = cols

            with col1:
                st.image(filenames[indices1[0][0]])
                
            with col2:
                st.image(filenames[indices1[0][1]])
            with col3:
                st.image(filenames[indices1[0][2]])
            with col4:
                st.image(filenames[indices1[0][3]])
            with col5:
                st.image(filenames[indices1[0][4]])
            
        except FileNotFoundError:
            st.warning("Image not found.")
    else:
        cols = st.columns(2)
        # Display all top 10 images
        for index, row in top_10_images_df.iterrows():
            image_path = os.path.join("images", row['image'])
            try:
                product_image = Image.open(image_path)
                resized_image = product_image.resize((100, 100))
                cols[index % 2].image(resized_image, caption=row['Product Name'])
            except FileNotFoundError:
                pass

    uploaded_file1 = st.file_uploader("Choose an image", key="image_uploader1")

    if uploaded_file1 is not None:
        if save_uploaded_image(uploaded_file1, 1):
            # Features extraction and recommendation logic
            model = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
            model.trainable = False
            model = tensorflow.keras.Sequential([
                model,
                GlobalMaxPooling2D()
            ])

            features1 = feature_extraction(os.path.join(os.getcwd(), "uploads", "uploaded_image_1.jpg"), model)
            indices1 = recommendions(features1, feature_list)

            image_path = os.path.join(os.getcwd(), "uploads", "uploaded_image_1.jpg")
            product_image = Image.open(image_path)
            st.image(product_image.resize((100, 100)))
            # Display recommendation images
            st.header("Closest Match Similar Products")
            cols = st.columns(5)
            col1, col2, col3, col4, col5 = cols

            with col1:
                st.image(filenames[indices1[0][0]])
            with col2:
                st.image(filenames[indices1[0][1]])
            with col3:
                st.image(filenames[indices1[0][2]])
            with col4:
                st.image(filenames[indices1[0][3]])
            with col5:
                st.image(filenames[indices1[0][4]])
        else:

            st.header("Some error occurred in file upload.")

if __name__ == "__main__":
    app()
