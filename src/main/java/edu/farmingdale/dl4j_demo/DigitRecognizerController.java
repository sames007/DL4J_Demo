package edu.farmingdale.dl4j_demo;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.WritableImage;
import javafx.concurrent.Task; // For background processing
import org.jetbrains.annotations.NotNull;

import java.net.URL;
import java.util.ResourceBundle;

public class DigitRecognizerController implements Initializable {

    @FXML
    private Canvas canvas;

    @FXML
    private Button clearBtn;

    @FXML
    private Button recognizeBtn; // Added for the new button

    @FXML
    private Label resultLabel;

    private GraphicsContext gc;
    private double lastX, lastY;

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        gc = canvas.getGraphicsContext2D();
        setupDrawing();
        // You could load the model here in a background task if desired,
        // but TrainCNN.getModel() will load it on first use anyway.
    }

    private void setupDrawing() {
        canvas.setOnMousePressed(e -> {
            lastX = e.getX();
            lastY = e.getY();
        });

        canvas.setOnMouseDragged(e -> {
            gc.setLineWidth(25);  // Set line width for drawing
            gc.strokeLine(lastX, lastY, e.getX(), e.getY());
            lastX = e.getX();
            lastY = e.getY();
        });
    }

    @FXML
    private void handleClear() {
        if (canvas != null && gc != null) {
            gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        }
        if (resultLabel != null) {
            resultLabel.setText("Draw a digit"); // Reset label
        }
    }

    @FXML
    private void handleRecognize() {
        if (canvas.getWidth() == 0 || canvas.getHeight() == 0) {
            resultLabel.setText("Canvas not ready.");
            return;
        }

        // Create a snapshot of the canvas
        WritableImage writableImage = new WritableImage((int) canvas.getWidth(), (int) canvas.getHeight());
        canvas.snapshot(null, writableImage);

        // Disable button and update label during processing
        recognizeBtn.setDisable(true);
        clearBtn.setDisable(true); // Also disable clear during recognition
        resultLabel.setText("Recognizing...");

        // Perform prediction in a background task to keep the UI responsive
        Task<Integer> recognitionTask = new Task<>() {
            @NotNull
            @Override
            protected Integer call() throws Exception {
                // Call the static predict method from TrainCNN
                return TrainCNN.predict(writableImage);
            }
        };

        recognitionTask.setOnSucceeded(_ -> {
            updatePrediction(String.valueOf(recognitionTask.getValue()));
            recognizeBtn.setDisable(false);
            clearBtn.setDisable(false);
        });

        recognitionTask.setOnFailed(_ -> {
            Throwable ex = recognitionTask.getException();
            ex.printStackTrace(); // Log the full error to the console
            // Provide a user-friendly error message
            String errorMessage = "Recognition failed. Ensure the model is trained and available.";
            if (ex != null && ex.getMessage() != null) {
                // Include a snippet of the actual error if helpful
                errorMessage += "\nDetails: " + ex.getMessage().substring(0, Math.min(ex.getMessage().length(), 100)) + "...";
            }
            resultLabel.setText(errorMessage);
            recognizeBtn.setDisable(false);
            clearBtn.setDisable(false);
        });

        new Thread(recognitionTask).start();
    }

    // Method to update the prediction result on the label
    public void updatePrediction(String prediction) {
        if (resultLabel != null) {
            resultLabel.setText("Predicted: " + prediction);
        }
    }
}