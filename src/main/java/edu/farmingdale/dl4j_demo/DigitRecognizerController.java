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

public class DigitRecognizerController {
    private static final Logger log = LoggerFactory.getLogger(DigitRecognizerController.class);
    private MultiLayerNetwork model;
    private GraphicsContext gc;

    @FXML private Canvas canvas;
    @FXML private Label resultLabel;

    @FXML
    public void initialize() {
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(new File("mnist-model.zip"));
            log.info("Model loaded.");
        } catch (Exception e) {
            log.error("Could not load model", e);
        }

        gc = canvas.getGraphicsContext2D();
        clearCanvas();

        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED, e -> {
            gc.beginPath();
            gc.moveTo(e.getX(), e.getY());
            gc.stroke();
        });
        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED, e -> {
            gc.lineTo(e.getX(), e.getY());
            gc.stroke();
        });
        canvas.addEventHandler(MouseEvent.MOUSE_RELEASED, e -> predict());
    }

    @FXML
    private void handleClear() {
        clearCanvas();
        resultLabel.setText("Draw a digit");
    }

    private void clearCanvas() {
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(20);
    }

    private void predict() {
        try {
            WritableImage fxImage = canvas.snapshot(null, null);
            BufferedImage bImage = SwingFXUtils.fromFXImage(fxImage, null);
            BufferedImage gray = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            java.awt.Graphics2D g = gray.createGraphics();
            g.drawImage(bImage, 0, 0, 28, 28, null);
            g.dispose();

            INDArray input = new NativeImageLoader(28,28,1,true).asMatrix(gray);
            new ImagePreProcessingScaler(0,1).transform(input);

            int pred = Nd4j.argMax(model.output(input), 1).getInt(0);
            resultLabel.setText("Predicted: " + pred);

        } catch (Exception ex) {
            log.error("Prediction failed", ex);
        }
    }
}
