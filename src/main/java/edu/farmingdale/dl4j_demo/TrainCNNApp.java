package edu.farmingdale.dl4j_demo;

import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.fxml.FXMLLoader;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException; // Import IOException
import java.net.URL;

/**
 * JavaFX application for training a Convolutional Neural Network (CNN)
 * using DL4J. Provides a simple UI to input the number of training epochs,
 * start the training process, and view log output.
 */
public class TrainCNNApp extends Application {
    private TextArea logArea;         // Area to display training log messages
    private Spinner<Integer> epochSpinner; // Spinner to select the number of training epochs
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
        // Create the training window
        Stage trainingStage = createTrainingStage();
        trainingStage.show();

        // Create the digit recognizer window
        createDigitRecognizerStage();
    }

    @NotNull
    private Stage createTrainingStage() {
        Stage stage = new Stage();

        // Set Up the log output area
        logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setWrapText(true);
        logArea.setPrefHeight(400);

        // Spinner to choose the number of epochs (min: 1, max: 50, default: 10)
        epochSpinner = new Spinner<>(1, 50, 10);

        // Button to begin training
        trainButton = new Button("Start Training");
        trainButton.setOnAction(_ -> startTraining());

        // Layout for the UI components
        VBox root = new VBox(10,
                new Label("Epochs:"), epochSpinner,
                trainButton,
                new Label("Training Log:"), logArea
        );
        root.setPadding(new Insets(10));
        root.setAlignment(Pos.TOP_CENTER);

        // Set up and show the stage
        Scene scene = new Scene(root, 500, 600);
        stage.setScene(scene);
        stage.setTitle("Train CNN Model");

        // Apply stylesheet to the training stage as well, if desired
        // Assuming style.css is at the root of resources (e.g., src/main/resources/style.css)
        URL cssUrlTraining = TrainCNNApp.class.getResource("/style.css");
        if (cssUrlTraining != null) {
            scene.getStylesheets().add(cssUrlTraining.toExternalForm());
        } else {
            System.err.println("Stylesheet not found for training stage: /style.css");
        }

        return stage;
    }

    private void createDigitRecognizerStage() {
        try {
            Stage digitStage = new Stage();
            // Use an absolute path from the classpath root
            String fxmlPath = "/digit_recognizer_view.fxml";
            URL fxmlUrl = getClass().getResource(fxmlPath);

            if (fxmlUrl == null) {
                showError("FXML file not found: /edu/farmingdale/dl4j_demo/digit_recognizer_view.fxml. Please check the path.");
                System.err.println("FXML file not found: /edu/farmingdale/dl4j_demo/digit_recognizer_view.fxml or digit_recognizer_view.fxml");
                return; // Can't proceed if FXML is not found
            }
            FXMLLoader fxmlLoader = new FXMLLoader(fxmlUrl);
            Scene scene = new Scene(fxmlLoader.load());

            // Add the stylesheet to the scene
            // Assuming style.css is at the root of resources (e.g., src/main/resources/style.css)
            URL cssUrlRecognizer = TrainCNNApp.class.getResource("/style.css");
            if (cssUrlRecognizer != null) {
                scene.getStylesheets().add(cssUrlRecognizer.toExternalForm());
            } else {
                System.err.println("Stylesheet not found for digit recognizer: /style.css");
            }

            digitStage.setTitle("Digit Recognizer");
            digitStage.setScene(scene);
            digitStage.show();
        } catch (IOException e) { // Catch IOException specifically for fxmlLoader.load()
            e.printStackTrace();
            showError("Error loading digit recognizer FXML: " + e.getMessage());
        } catch (Exception e) { // Catch other potential exceptions
            e.printStackTrace();
            showError("An unexpected error occurred while creating the digit recognizer window: " + e.getMessage());
        }
    }

    private void showError(String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error"); // Simplified title
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
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
            @Nullable
            @Override
            protected Void call() throws Exception {
                // Call the static training method and update the log via callback
                TrainCNN.runTraining(epochs, line -> updateMessage(line + "\n"));
                return null;
            }
        };

        // Bind the log area to the task's message updates
        logArea.textProperty().bind(trainingTask.messageProperty());

        // Re-enable the button after training finishes
        trainingTask.setOnSucceeded(_ -> {
            logArea.appendText("\n✅ Training complete.\n");
            trainButton.setDisable(false);
        });

        // Handle exceptions that occur during training
        trainingTask.setOnFailed(_ -> {
            logArea.appendText("\n❌ Training failed: "
                    + trainingTask.getException().getMessage() + "\n");
            trainButton.setDisable(false);
        });

        // Start training in a separate thread
        new Thread(trainingTask).start();
    }
}