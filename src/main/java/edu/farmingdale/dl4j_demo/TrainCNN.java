package edu.farmingdale.dl4j_demo;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.optimize.listeners.StatsListener;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class TrainCNN {
    private static final Logger log = LoggerFactory.getLogger(TrainCNN.class);

    public static void main(String[] args) throws Exception {
        int height = 28, width = 28, channels = 1, outputNum = 10;
        int batchSize = 64, epochs = 10;
        long seed = 1234;

        File trainData = new File("src/main/resources/edu/farmingdale/dl4j_demo/mnist_png/training");
        File testData  = new File("src/main/resources/edu/farmingdale/dl4j_demo/mnist_png/testing");

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        Random rng = new Random(seed);
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit testSplit  = new FileSplit(testData,  NativeImageLoader.ALLOWED_FORMATS, rng);

        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        trainRR.initialize(trainSplit);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testSplit);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(0.006, 0.9))
                .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .list()
                .layer(new ConvolutionLayer.Builder(5,5)
                        .nIn(channels).stride(1,1).nOut(20)
                        .activation(Activation.IDENTITY).build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())
                .layer(new ConvolutionLayer.Builder(5,5)
                        .stride(1,1).nOut(50)
                        .activation(Activation.IDENTITY).build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2).stride(2,2).build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU).nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nOut(outputNum).build())
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(100));

        log.info("----- TRAINING START -----");
        for (int i = 0; i < epochs; i++) {
            model.fit(trainIter);
            trainIter.reset();
            Evaluation eval = model.evaluate(testIter);
            log.info("Epoch {} complete. Accuracy: {}", i, eval.accuracy());
            testIter.reset();
        }

        File modelFile = new File("mnist-model.zip");
        ModelSerializer.writeModel(model, modelFile, true);
        log.info("Model saved to {}", modelFile.getAbsolutePath());
    }
}
