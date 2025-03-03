import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing import image
from tkinter import Tk, Label, Button, filedialog, Frame, messagebox
from PIL import Image, ImageTk
from sklearn.metrics import precision_score, recall_score, f1_score
import threading

# Load trained model
model = tf.keras.models.load_model('skin_type_model.h5')

# Define class labels
class_labels = ['dry', 'normal', 'oily']

# Global variables for camera handling
cap = None
camera_open = False
video_label = None

# Function to preprocess and predict skin type for one image
def predict_skin_type(img_path):
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        return class_labels[predicted_class_index]
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Function to open camera preview (real-time)
def open_camera_preview():
    global cap, camera_open, video_label

    cap = cv2.VideoCapture(2)  # Camera index (0 for default)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access camera.")
        return

    camera_open = True
    video_label = Label(root)
    video_label.pack(pady=10)

    def update_frame():
        if not camera_open:
            return
        
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((300, 300), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            video_label.imgtk = img_tk
            video_label.configure(image=img_tk)
        video_label.after(30, update_frame)

    update_frame()

# Function to capture from open camera and analyze
def capture_and_predict_from_camera():
    global cap, camera_open

    if cap is None or not camera_open:
        messagebox.showerror("Error", "Camera is not open.")
        return

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image.")
        return

    capture_path = "captured_skin_image.jpg"
    cv2.imwrite(capture_path, frame)

    show_image(capture_path)

    predicted_class = predict_skin_type(capture_path)

    if predicted_class:
        result_label.config(text=f"Predicted Skin Type: {predicted_class}", fg="green")
    else:
        result_label.config(text="Prediction failed.", fg="red")

# Function to close camera preview
def close_camera():
    global cap, camera_open
    if cap:
        cap.release()
        cap = None
    camera_open = False
    if video_label:
        video_label.pack_forget()

# Function for single image prediction
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    predicted_class = predict_skin_type(file_path)

    if predicted_class:
        result_label.config(text=f"Predicted Skin Type: {predicted_class}", fg="green")
        show_image(file_path)
    else:
        result_label.config(text="Prediction failed.", fg="red")

# Function to display image in label
def show_image(img_path):
    img = Image.open(img_path)
    img = img.resize((200, 200), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# # Function to calculate accuracy, precision, recall, F1-score (batch prediction)
# def calculate_folder_accuracy():
#     folder_path = filedialog.askdirectory()
#     if not folder_path:
#         return

#     y_true = []
#     y_pred = []

#     with mlflow.start_run():
#         for class_name in class_labels:
#             class_folder = os.path.join(folder_path, class_name)
#             if not os.path.isdir(class_folder):
#                 continue

#             for img_name in os.listdir(class_folder):
#                 if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
#                     img_path = os.path.join(class_folder, img_name)
#                     predicted_class = predict_skin_type(img_path)
#                     y_true.append(class_name)
#                     y_pred.append(predicted_class)

#         accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
#         precision = precision_score(y_true, y_pred, average='weighted', labels=class_labels)
#         recall = recall_score(y_true, y_pred, average='weighted', labels=class_labels)
#         f1 = f1_score(y_true, y_pred, average='weighted', labels=class_labels)

#         # Log metrics to MLflow
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("precision", precision)
#         mlflow.log_metric("recall", recall)
#         mlflow.log_metric("f1_score", f1)

#         mlflow.tensorflow.log_model(model, "skin_type_model")

#         messagebox.showinfo("Accuracy Report", f"Overall Accuracy: {accuracy:.2f}%\n"
#                                                f"Precision: {precision:.2f}\n"
#                                                f"Recall: {recall:.2f}\n"
#                                                f"F1 Score: {f1:.2f}")

#    mlflow.end_run()

# GUI setup
root = Tk()
root.title("Skin Type Prediction")
root.geometry("500x650")
root.configure(bg="#f0f0f0")

# Header label
header_label = Label(root, text="Skin Type Prediction", font=("Arial", 20, "bold"), bg="#f0f0f0", fg="#333")
header_label.pack(pady=10)

# Image frame
image_frame = Frame(root, bd=2, relief="ridge", bg="white")
image_frame.pack(pady=10)

image_label = Label(image_frame, bg="white")
image_label.pack()

# Result label
result_label = Label(root, text="Predicted Skin Type: ", font=("Arial", 14), bg="#f0f0f0", fg="#555")
result_label.pack(pady=10)

# Buttons frame
button_frame = Frame(root, bg="#f0f0f0")
button_frame.pack(pady=20)

# Upload and Predict Button
upload_button = Button(button_frame, text="Upload Image", command=upload_and_predict, bg="#007BFF", fg="white", font=("Arial", 12), width=15)
upload_button.grid(row=0, column=0, padx=10)

# Camera Control Frame
camera_frame = Frame(root, bg="#f0f0f0")
camera_frame.pack(pady=10)

# Open Camera Button
open_camera_button = Button(camera_frame, text="Open Camera", command=open_camera_preview, bg="#17A2B8", fg="white", font=("Arial", 12), width=15)
open_camera_button.grid(row=0, column=0, padx=5)

# Capture and Predict Button
capture_button = Button(camera_frame, text="Capture & Analyze", command=capture_and_predict_from_camera, bg="#FFC107", fg="black", font=("Arial", 12), width=15)
capture_button.grid(row=0, column=1, padx=5)

# Close Camera Button
close_camera_button = Button(camera_frame, text="Close Camera", command=close_camera, bg="#DC3545", fg="white", font=("Arial", 12), width=15)
close_camera_button.grid(row=1, column=0, columnspan=2, pady=5)

# Footer label
footer_label = Label(root, text="Developed by You", font=("Arial", 10), bg="#f0f0f0", fg="#888")
footer_label.pack(side="bottom", pady=10)

# Start GUI loop
root.mainloop()
