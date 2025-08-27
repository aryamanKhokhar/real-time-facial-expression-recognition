# real-time-facial-expression-recognition

**Facial Expression Recognition**

This project implements a real-time facial expression recognition system using deep learning.
A custom CNN architecture was built from scratch and trained on the Face Expression Recognition Dataset (Kaggle)
.

The system can classify human emotions such as happy, sad, angry, surprised, neutral, and more in real time using OpenCV and Haar Cascade for face detection.

**Dataset**

Dataset: Face Expression Recognition Dataset

Contains ~35,000+ labeled images of different facial expressions across multiple categories.

Images are grayscale and standardized for training.

Steps in the Project
1) **Importing Libraries**

Essential Python libraries such as NumPy, Pandas, TensorFlow/Keras, OpenCV, and Matplotlib are imported for model building, preprocessing, and visualization.

2) **Uploading Files and Setting Directory**

The dataset is loaded and directories are set up for training and validation.

3) **Preparing Dataset: Image Paths and Labels**

Extracts image file paths and corresponding labels (e.g., happy, sad, angry) for further preprocessing.

4) **Feature Extraction**

Images are resized, converted to arrays, and normalized for feeding into the model.

5) **Scaling the Data**

Pixel values are scaled (usually between 0 and 1) to ensure faster and more stable training.

6) **One Hot Encoding the Labels**

Class labels (e.g., happy, angry) are converted into one-hot vectors for categorical classification.

7) **Data Augmentation**

Techniques like rotation, flipping, zooming, and shifting are applied to increase dataset variety and reduce overfitting.

8) **Building and Compiling the Model**

A Convolutional Neural Network (CNN) is built from scratch with layers like Conv2D, MaxPooling, Dropout, and Dense layers.
Compiled using the Adam optimizer and categorical crossentropy loss.

9) **Training the Model**

The model is trained on the dataset with defined epochs and batch size.

10) **Making Predictions**

Model predictions are tested on unseen images and real-time video streams using OpenCV.
Haar Cascade XML is used to detect faces, and the trained CNN predicts the corresponding expression.

**Technologies Used**

Python

TensorFlow / Keras

OpenCV (Haar Cascade for face detection)

NumPy, Pandas, Matplotlib
