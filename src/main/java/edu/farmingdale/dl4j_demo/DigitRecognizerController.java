package edu.farmingdale.dl4j_demo;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;

import java.net.URL;
import java.util.ResourceBundle;

public class DigitRecognizerController implements Initializable {
    @FXML
    private Canvas canvas;
    
    @FXML
    private Button clearBtn;
    
    @FXML
    private Label resultLabel;
    
    private GraphicsContext gc;
    private double lastX, lastY;
    
    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        gc = canvas.getGraphicsContext2D();
        setupDrawing();
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
            resultLabel.setText("Draw a digit");
        }
    }

    // Method to update the prediction result
    public void updatePrediction(String prediction) {
        if (resultLabel != null) {
            resultLabel.setText("Predicted: " + prediction);
        }
    }
}