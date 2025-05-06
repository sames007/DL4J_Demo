package edu.farmingdale.dl4j_demo;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import javafx.scene.image.Image;
import javafx.embed.swing.SwingFXUtils;

import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

/**
 * TrainCNN
 * Configures and trains a simple Convolutional Neural Network on
 * the MNIST PNG dataset, logging progress via a provided callback.
 * Also provides utility for loading the model and making predictions.
 */
public class TrainCNN {
    private static final Logger log = LoggerFactory.getLogger(TrainCNN.class);
    private static MultiLayerNetwork modelInstance; // Cached model instance
    private static final String MODEL_PATH = "mnist-model.zip";

    /**
     * Standalone entry point: train for 10 epochs and print logs to stdout.
     */
    public static void main(String[] args) throws Exception {
        runTraining(10, msg -> System.out.println(msg + "\n"));
    }

    /**
     * Gets the trained model. Loads it from disk if not already in memory.
     * @return The MultiLayerNetwork model.
     * @throws IOException If the model file is not found or cannot be loaded.
     */
    public static synchronized MultiLayerNetwork getModel() throws IOException {
        if (modelInstance == null) {
            File modelFile = new File(MODEL_PATH);
            if (!modelFile.exists()) {
                String errorMsg = "Model file not found: " + modelFile.getAbsolutePath() +
                        ". Please train the model first by running the training application.";
                log.error(errorMsg);
                throw new IOException(errorMsg);
            }
            log.info("Loading model from: " + modelFile.getAbsolutePath());
            modelInstance = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            log.info("Model loaded successfully.");
        }
        return modelInstance;
    }

    /**
     * Predicts the digit from a JavaFX Image.
     * @param fxImage The JavaFX Image (expected to be 280x280, black drawing on a white background).
     * @return The predicted digit (0-9).
     * @throws Exception If there's an error during model loading or prediction.
     */
    public static int predict(Image fxImage) throws Exception {
        MultiLayerNetwork model = getModel();

        // 1. Convert JavaFX Image to BufferedImage
        BufferedImage bImage = SwingFXUtils.fromFXImage(fxImage, null);

        // 2. Resize BufferedImage to 28x28 and convert to grayscale
        //    The model expects 28x28, 1 channel (grayscale)
        BufferedImage resizedBImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = resizedBImage.createGraphics();
        // Use smooth scaling
        g2d.drawImage(bImage, 0, 0, 28, 28, null);
        g2d.dispose();

        // 3. Convert resized BufferedImage to INDArray using NativeImageLoader
        //    Height, Width, Channels
        NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
        INDArray imageArray = loader.asMatrix(resizedBImage); // Shape: [1, 1, 28, 28]

        // 4. Normalize pixel values from [0..255] to [0..1]
        //    Our drawing is black (0) on white (255). After scaling, it's black (0) on white (1).
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(imageArray); // Applies normalization in-place

        // 5. Invert colors: MNIST is typically a white digit (~1) on a black background (~0).
        //    Our input (black on white, scaled to 0 on 1) needs to be inverted.
        imageArray = imageArray.rsub(1.0);

        // 6. Make prediction
        INDArray output = model.output(imageArray);

        // 7. Get the predicted class (digit with the highest probability)
        //    output is a row vector of probabilities, argMax along dimension 1 finds the index of max value.
        int[] prediction = output.argMax(1).toIntVector();
        return prediction[0];
    }

    /**
     * Runs the CNN training loop.
     * @param epochs      Number of full passes over the dataset
     * @param logConsumer Consumer that receives one log line per update
     * @throws Exception on file or training errors
     */
    public static void runTraining(int epochs, @NotNull Consumer<String> logConsumer) throws Exception {
        // Image dimensions and number of output classes (digits 0–9)
        final int height = 28, width = 28, channels = 1, outputNum = 10;
        final int batchSize = 64;        // Number of examples per mini‑batch
        final long seed = 1234;          // Random seed for reproducibility

        logConsumer.accept("Setting up data iterators...");
        // Paths to the training and test image folders
        File trainData = new File("src/main/resources/edu/farmingdale/dl4j_demo/mnist_png/training");
        File testData  = new File("src/main/resources/edu/farmingdale/dl4j_demo/mnist_png/testing");

        // Use parent directory names as labels (0, 1, 2, …)
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        Random rng = new Random(seed);

        // Create FileSplits for both training and test sets
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit testSplit  = new FileSplit(testData,  NativeImageLoader.ALLOWED_FORMATS, rng);

        // RecordReaders convert image files into record formats
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        trainRR.initialize(trainSplit);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testSplit);
        DataSetIterator testIter  = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);

        // Normalize pixel values from [0..255] to [0..1]
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter); // Fit on training data
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);
        logConsumer.accept("Data iterators ready.");

        logConsumer.accept("Building network configuration...");
        // Build the CNN architecture
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(0.006, 0.9))               // Learning rate and momentum
                .weightInit(WeightInit.XAVIER) // Using imported WeightInit
                .trainingWorkspaceMode(WorkspaceMode.ENABLED) // If this causes errors, it's likely a deeper native/dependency issue
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED) // Same as above
                .list()
                // First convolution + max‐pool
                .layer(new ConvolutionLayer.Builder(5,5)
                        .nIn(channels).stride(1,1).nOut(20)
                        .activation(Activation.IDENTITY).build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())
                // Second convolution + max‐pool
                .layer(new ConvolutionLayer.Builder(5,5)
                        .stride(1,1).nOut(50)
                        .activation(Activation.IDENTITY).build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())
                // Fully connected layer
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU).nOut(500).build())
                // Output layer with softmax for classification
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nOut(outputNum).build())
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .build();
        logConsumer.accept("Network configuration built.");

        // Initialize the model
        MultiLayerNetwork newModel = new MultiLayerNetwork(conf);
        newModel.init();
        logConsumer.accept("Model initialized.");

        // Setup listeners
        List<TrainingListener> listeners = new ArrayList<>();
        listeners.add(new ScoreIterationListener(100)); // Always log score to console

        // Try to set up UI, but don't let it stop training if it fails
        try {
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            listeners.add(new StatsListener(statsStorage));
            logConsumer.accept("UI Server attached. Access at http://localhost:9000");
        } catch (Exception e) {
            log.warn("Could not start DL4J UI Server. Training will continue without UI. Error: {}", e.getMessage(), e);
            logConsumer.accept("Warning: DL4J UI Server could not start. Check logs for details (e.g., port 9000 might be in use).");
        }
        newModel.setListeners(listeners);


        logConsumer.accept("----- TRAINING START -----");

        // Training loop: fit, evaluate, log accuracy
        for (int i = 0; i < epochs; i++) {
            logConsumer.accept(String.format("Starting Epoch %d/%d...", i + 1, epochs));
            newModel.fit(trainIter);
            trainIter.reset();

            logConsumer.accept(String.format("Evaluating model after Epoch %d...", i + 1));
            Evaluation eval = newModel.evaluate(testIter);
            String line = String.format("Epoch %d complete. Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                    i + 1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1());
            logConsumer.accept(line);
            log.info(line); // Also log to SLF4J
            testIter.reset();
        }

        File modelFile = new File(MODEL_PATH);
        ModelSerializer.writeModel(newModel, modelFile, true);
        logConsumer.accept("Model saved to " + modelFile.getAbsolutePath());
        log.info("Model saved to " + modelFile.getAbsolutePath());
        modelInstance = newModel;
    }
}