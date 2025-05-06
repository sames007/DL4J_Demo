package edu.farmingdale.dl4j_demo;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.File;

/**
 * Controller for the JavaFX digit recognizer UI.
 * <p>
 * Handles:
 *  - Loading a pre-trained DL4J model,
 *  - Capturing user‐drawn digits on a Canvas,
 *  - Preprocessing the image,
 *  - Running inference, and
 *  - Displaying the predicted digit.
 */
public class DigitRecognizerController {

    // Logger for debug & error messages
    private static final Logger log = LoggerFactory.getLogger(DigitRecognizerController.class);

    // DL4J neural network that will perform digit classification
    private MultiLayerNetwork model;

    // GraphicsContext for drawing on the JavaFX Canvas
    private GraphicsContext gc;

    // Reference to the Canvas UI element (injected via FXML)
    @FXML private Canvas canvas;

    // Label to show the prediction result to the user
    @FXML private Label resultLabel;

    /**
     * Called automatically by JavaFX after FXML fields are injected.
     * Initializes the model and sets up canvas event handlers for drawing.
     */
    @FXML
    public void initialize() {
        // Attempt to load the trained model from disk
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(new File("mnist-model.zip"));
            log.info("Model loaded successfully.");
        } catch (Exception e) {
            log.error("Could not load model file", e);
        }

        // Get the drawing context of the canvas and clear it
        gc = canvas.getGraphicsContext2D();
        clearCanvas();

        // When the user presses the mouse: begin a new stroke
        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED, e -> {
            gc.beginPath();
            gc.moveTo(e.getX(), e.getY());
            gc.stroke();
        });

        // As the user drags: continue the stroke to current point
        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED, e -> {
            gc.lineTo(e.getX(), e.getY());
            gc.stroke();
        });

        // When the user releases: trigger prediction
        canvas.addEventHandler(MouseEvent.MOUSE_RELEASED, e -> predict());
    }

    /**
     * Clears the drawing canvas and resets the result label.
     * Invoked when the user presses the “Clear” button.
     */
    @FXML
    private void handleClear() {
        clearCanvas();
        resultLabel.setText("Draw a digit");
    }

    /**
     * Fills the canvas with white background and sets brush properties.
     */
    private void clearCanvas() {
        // White background
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        // Black, thick stroke for drawing
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(20);
    }

    /**
     * Captures the canvas content, preprocesses it into a 28×28 grayscale image,
     * feeds it to the neural network, and updates the UI with the predicted digit.
     */
    private void predict() {
        try {
            // Snapshot the JavaFX Canvas into an image
            WritableImage fxImage = canvas.snapshot(null, null);

            // Convert to a BufferedImage for AWT processing
            BufferedImage bImage = SwingFXUtils.fromFXImage(fxImage, null);

            // Create a 28×28 grayscale image
            BufferedImage gray = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            java.awt.Graphics2D g = gray.createGraphics();

            // Draw and scale the user input down to 28×28
            g.drawImage(bImage, 0, 0, 28, 28, null);
            g.dispose();

            // Convert the BufferedImage to an INDArray (shape: [1,1,28,28])
            INDArray input = new NativeImageLoader(28, 28, 1, true)
                    .asMatrix(gray);

            // Normalize pixel values from [0,255] to [0,1]
            new ImagePreProcessingScaler(0, 1).transform(input);

            // Run the model and find the index with highest probability
            int pred = Nd4j.argMax(model.output(input), 1).getInt(0);

            // Display the prediction
            resultLabel.setText("Predicted: " + pred);

        } catch (Exception ex) {
            // Log any errors during preprocessing or inference
            log.error("Prediction failed", ex);
        }
    }
}
