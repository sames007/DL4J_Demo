# DL4J Digit Recognition Demo

A small Java demo that trains a simple neural network on hand‑drawn digits and shows live training metrics in a browser.
![image](https://github.com/user-attachments/assets/7c32cb76-ee6e-4588-91b1-871f75605a31)

---

## Features

- **Draw‑your‑own digit** in a simple GUI or web canvas  
- **Train a MultiLayerNetwork** on your drawing plus MNIST data  
- **Live training UI** with loss, accuracy, gradient and weight histograms  
- **Automatic CPU/GPU backend** selection via ND4J  
- **Easy setup** with Maven  

---

## Prerequisites

- Java 11 or newer (Oracle JDK or OpenJDK)  
- Maven 3.6+  
- (Optional) NVIDIA GPU + CUDA/cuDNN for faster training  

---

## Dependencies

In your `pom.xml`, include:

```xml
<!-- DL4J core -->
<dependency>
  <groupId>org.deeplearning4j</groupId>
  <artifactId>deeplearning4j-core</artifactId>
  <version>1.0.0-M1.1</version>
</dependency>

<!-- ND4J for CPU (or use nd4j-cuda-xx-platform for GPU) -->
<dependency>
  <groupId>org.nd4j</groupId>
  <artifactId>nd4j-native-platform</artifactId>
  <version>1.0.0-M1.1</version>
</dependency>

<!-- DL4J Training UI -->
<dependency>
  <groupId>org.deeplearning4j</groupId>
  <artifactId>deeplearning4j-ui</artifactId>
  <version>1.0.0-M1.1</version>
</dependency>
