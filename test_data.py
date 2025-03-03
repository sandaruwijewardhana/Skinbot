import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = tf.keras.models.load_model('skin_type_model.h5')

# Log model to MLflow
with mlflow.start_run():
    mlflow.tensorflow.log_model(model, "skin_type_model")

    # Define the class labels (make sure they match the classes used during training)
    class_labels = ['dry', 'normal', 'oily']

    # Function to load and preprocess a single image
    def preprocess_image(img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]
        return img_array

    # Function to evaluate the model on a test dataset directory
    def evaluate_model_on_test_data(model, test_data_dir):
        y_true = []
        y_pred = []

        # Iterate through each class folder
        for label in class_labels:
            class_folder = os.path.join(test_data_dir, label)

            # Skip if folder does not exist (useful for partial test sets)
            if not os.path.isdir(class_folder):
                continue

            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(class_folder, img_name)

                    # Preprocess and predict
                    img_array = preprocess_image(img_path)
                    prediction = model.predict(img_array)
                    predicted_label = class_labels[np.argmax(prediction)]

                    y_true.append(label)
                    y_pred.append(predicted_label)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels))

        # Overall precision, recall, F1-score
        precision = precision_score(y_true, y_pred, average='weighted', labels=class_labels)
        recall = recall_score(y_true, y_pred, average='weighted', labels=class_labels)
        f1 = f1_score(y_true, y_pred, average='weighted', labels=class_labels)

        print(f"Overall Precision: {precision:.2f}")
        print(f"Overall Recall: {recall:.2f}")
        print(f"Overall F1 Score: {f1:.2f}")

        # Log metrics to MLflow
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        return precision, recall, f1

    # Path to test data directory
    test_data_dir = r"D:\CO542\project\SKINBOT\test"

    # Run evaluation
    precision, recall, f1 = evaluate_model_on_test_data(model, test_data_dir)

    # Optionally, save the results to a file
    with open("evaluation_report.txt", "w") as f:
        f.write(f"Overall Precision: {precision:.2f}\n")
        f.write(f"Overall Recall: {recall:.2f}\n")
        f.write(f"Overall F1 Score: {f1:.2f}\n")
    print("\nEvaluation report saved to 'evaluation_report.txt'")

    # Visualize the evaluation metrics using Matplotlib
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Plotting the metrics
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.show()
