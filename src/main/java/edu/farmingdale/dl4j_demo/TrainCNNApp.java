package edu.farmingdale.dl4j_demo;

import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class TrainCNNApp extends Application {

    private TextArea logArea;
    private Spinner<Integer> epochSpinner;
    private Button trainButton;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) {
        logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setWrapText(true);
        logArea.setPrefHeight(400);

        epochSpinner = new Spinner<>(1, 50, 10);
        trainButton = new Button("Start Training");
        trainButton.setOnAction(e -> startTraining());

        VBox root = new VBox(10,
                new Label("Epochs:"), epochSpinner,
                trainButton,
                new Label("Training Log:"), logArea
        );
        root.setPadding(new Insets(10));
        root.setAlignment(Pos.TOP_CENTER);

        stage.setScene(new Scene(root, 500, 600));
        stage.setTitle("Train CNN Model - JavaFX UI");
        stage.show();
    }

    private void startTraining() {
        trainButton.setDisable(true);
        logArea.clear();
        int epochs = epochSpinner.getValue();

        Task<Void> trainingTask = new Task<>() {
            @Override
            protected Void call() throws Exception {
                TrainCNNWithListener.runTraining(epochs, logArea::appendText);
                return null;
            }

            @Override
            protected void succeeded() {
                logArea.appendText("\nTraining complete.\n");
                trainButton.setDisable(false);
            }

            @Override
            protected void failed() {
                logArea.appendText("\nTraining failed: " + getException().getMessage() + "\n");
                trainButton.setDisable(false);
            }
        };

        new Thread(trainingTask).start();
    }
}