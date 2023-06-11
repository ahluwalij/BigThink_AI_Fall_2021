# Malaria Diagnosis Through Image Classification using Convolutional Neural Networks (CNN)

View the live notebook [here](https://www.jasdeepahluwalia.com/files/notebooks/malaria_detection)

This repository demonstrates a powerful application of Convolutional Neural Networks (CNNs) to medical imaging, specifically to the diagnosis of Malaria through cell images. We built a comprehensive pipeline from data acquisition and preprocessing, to training a deep learning model and evaluating its performance.

## Dataset
The dataset, sourced from the official NIH Website, is a benchmark dataset in the field of medical imaging. It consists of 27,558 cell images labeled as either parasitized (infected) or uninfected, providing a solid foundation for binary image classification tasks.

Access the dataset here: [Malaria Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).

## Frameworks and Libraries
The project harnesses the power of various Python libraries that are fundamental in the Data Science and AI domain:
- **TensorFlow**: An end-to-end open source platform for machine learning.
- **Keras**: A high-level neural networks API, written in Python and capable of running on top of TensorFlow.
- **Numpy**: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- **Seaborn**: A Python data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.

## Methodology
1. **Importing Necessary Libraries**: Importing the necessary computational and visualization libraries to load and analyze the data.

2. **Data Acquisition and Preprocessing**: Downloading the dataset from the internet, loading it into Python using tensorflow's data API and performing necessary preprocessing steps such as shuffling and batch normalization.

3. **Model Building**: Building the Convolutional Neural Network (CNN) using TensorFlow and Keras, defining the structure of the model, and tuning hyperparameters for optimal performance.

4. **Training the Model**: The process of backpropagation and gradient descent to optimize the model's performance on the training data.

5. **Evaluating the Model**: Testing the model's performance on unseen data, examining the confusion matrix, and calculating relevant performance metrics including accuracy, precision, recall, and F1 score.

## Results
The model achieved an impressive accuracy of 85% on the unseen test data. It shows that with deep learning, we can automate the process of detecting Malaria cells, providing a quick and efficient way of diagnosis which could potentially save many lives especially in areas where access to medical services is limited.

## Usage
This project requires Python 3.x and the necessary libraries mentioned above. After installing any missing dependencies, you can simply clone the project and run the .ipynb file in a Jupyter notebook environment.

## Further Reading
For more information about the libraries, functions, and tools used in this project, here are a few resources:
- [Malaria Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)
- [TensorFlow](https://www.tensorflow.org)
- [Keras](https://keras.io)
- [Seaborn](https://seaborn.pydata.org)
- [Numpy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
