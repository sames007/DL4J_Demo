package edu.farmingdale.dl4j_demo;

import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

/**
 * JavaFX application for training a Convolutional Neural Network (CNN)
 * using DL4J. Provides a simple UI to input the number of training epochs,
 * start the training process, and view log output.
 */
public class TrainCNNApp extends Application {
    private TextArea logArea;         // Area to display training log messages
    private Spinner<Integer> epochSpinner; // Spinner to select number of training epochs
    private Button trainButton;       // Button to trigger training

    /**
     * Main entry point of the application.
     * @param args command-line arguments (not used)
     */
    public static void main(String[] args) {
        launch(args);
    }

    /**
     * Initializes and displays the JavaFX UI.
     * @param stage the primary window
     */
    @Override
    public void start(Stage stage) {
        // Setup the log output area
        logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setWrapText(true);
        logArea.setPrefHeight(400);

        // Spinner to choose the number of epochs (min: 1, max: 50, default: 10)
        epochSpinner = new Spinner<>(1, 50, 10);

        // Button to begin training
        trainButton = new Button("Start Training");
        trainButton.setOnAction(e -> startTraining());

        // Layout for the UI components
        VBox root = new VBox(10,
                new Label("Epochs:"), epochSpinner,
                trainButton,
                new Label("Training Log:"), logArea
        );
        root.setPadding(new Insets(10));
        root.setAlignment(Pos.TOP_CENTER);

        // Set up and show the stage
        stage.setScene(new Scene(root, 500, 600));
        stage.setTitle("Train CNN Model - JavaFX UI");
        stage.show();
    }

    /**
     * Starts the training process in a background thread.
     * Uses a JavaFX Task to perform training and update the log area with progress.
     */
    private void startTraining() {
        // Disable the button to prevent multiple runs
        trainButton.setDisable(true);
        logArea.clear();
        int epochs = epochSpinner.getValue();

        // Define the background task for training
        Task<Void> trainingTask = new Task<>() {
            @Override
            protected Void call() throws Exception {
                // Call the static training method and update log via callback
                TrainCNN.runTraining(epochs, line -> updateMessage(line + "\n"));
                return null;
            }
        };

        // Bind the log area to the task's message updates
        logArea.textProperty().bind(trainingTask.messageProperty());

        // Re-enable the button after training finishes
        trainingTask.setOnSucceeded(e -> {
            logArea.appendText("\n✅ Training complete.\n");
            trainButton.setDisable(false);
        });

        // Handle exceptions that occur during training
        trainingTask.setOnFailed(e -> {
            logArea.appendText("\n❌ Training failed: "
                    + trainingTask.getException().getMessage() + "\n");
            trainButton.setDisable(false);
        });

        // Start training in a separate thread
        new Thread(trainingTask).start();
    }
}
