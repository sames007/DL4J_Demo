package edu.farmingdale.dl4j_demo;

// DataVec imports for loading and labeling image data
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;

// Iterator to wrap RecordReader into a DataSetIterator
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

// Core DL4J configuration and network classes
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;

// UI server for real‑time training metrics
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.model.stats.StatsListener;

// Evaluation after each epoch
import org.nd4j.evaluation.classification.Evaluation;

// Preprocessing, activation, and loss functions
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

// Java & logging
import java.io.File;
import java.util.Random;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * TrainCNN
 *
 * Configures and trains a simple Convolutional Neural Network on
 * the MNIST PNG dataset, logging progress via a provided callback.
 */
public class TrainCNN {
    private static final Logger log = LoggerFactory.getLogger(TrainCNN.class);

    /**
     * Standalone entry point: train for 10 epochs and print logs to stdout.
     */
    public static void main(String[] args) throws Exception {
        runTraining(10, msg -> System.out.println(msg + "\n"));
    }

    /**
     * Runs the CNN training loop.
     *
     * @param epochs      Number of full passes over the dataset
     * @param logConsumer Consumer that receives one log line per update
     * @throws Exception on file or training errors
     */
    public static void runTraining(int epochs, Consumer<String> logConsumer) throws Exception {
        // Image dimensions and number of output classes (digits 0–9)
        final int height = 28, width = 28, channels = 1, outputNum = 10;
        final int batchSize = 64;        // Number of examples per mini‑batch
        final long seed = 1234;          // Random seed for reproducibility

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
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        // Build the CNN architecture
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(0.006, 0.9))               // Learning rate and momentum
                .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
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

        // Initialize the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Start UI server to visualize training metrics in browser
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        // Attach listeners for both stats and score output every 100 iterations
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(100));

        // Publish start message
        logConsumer.accept("----- TRAINING START -----");

        // Training loop: fit, evaluate, log accuracy
        for (int i = 0; i < epochs; i++) {
            model.fit(trainIter);         // Train on entire training set
            trainIter.reset();            // Reset iterator for next epoch

            Evaluation eval = model.evaluate(testIter);
            String line = String.format("Epoch %d complete. Accuracy: %.4f",
                    i, eval.accuracy());
            logConsumer.accept(line);

            testIter.reset();             // Reset test iterator
        }

        // Save the trained model to disk for later use
        File modelFile = new File("mnist-model.zip");
        ModelSerializer.writeModel(model, modelFile, true);
        logConsumer.accept("Model saved to " + modelFile.getAbsolutePath());
    }
}
