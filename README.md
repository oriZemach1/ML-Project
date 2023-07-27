This is my Machine Learning Project with Deep Learning I created as a high school senior.
This project is a basic implementation of OCR (Optical Character Recognition) software.

It takes a picture of handwritten text as Input and outputs the predicted text as a string.

You can find a detailed description of the project (in Hebrew) in the file תיק פרויקט.pdf.
Furthermore, I have created an Android application to demonstrate the result of the project. You can install it with the app-debug.apk file here: https://drive.google.com/drive/u/1/folders/1u_NCAd7_-UanRUjnlqtry-papusEVqrk .
For further instructions please refer to the pdf file.

A brief summary:
- The dataset in this project is the IAM dataset, holding over 100,000 images of handwritten words.
- I used a CNN to LSTM model architecture with a CTC activation to predict the words.
- The results on the test set are ~80% success of entire word predictions and ~91% success of character predictions.
