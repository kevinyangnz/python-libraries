# TensorFlow
TensorFlow is a library for machine learning and artificial intelligence, utilising linear algebra and statistics. 

It has a collection of APIs for:

- Data processing
- Visualisation
- Model evaluation

Examples of where TensorFlow is used:

- Object detection in MRI scans
- Predictive analysis to sort tweets by time relevance on Twitter
- Recommending music on Spotify
- Fraud detection by PayPal


## References
https://youtu.be/i8NETqtGHms?si=oGUHS-ZMgTeM5bQe

<br> <hr> 

# Keras

Keras is an API which provides an interface for artificial neural networks. Keras is integrated into TensorFlow.

Commonly used for:

- Image recognition
- Natural language processing
- Speech recognition

## References

https://keras.io/about/  
https://www.geeksforgeeks.org/what-is-keras/


# Number Identifier

I have developed a machine learning model using Tensorflow and Keras to accurately classify digits from 0-9, with the numbers_samples folder containing 10 of each digit drawn by myself using Microsoft Paint. Utilizing the image_augmentation.py file, we will apply randomly generated image augmentation to create an additional 990 images per digit. This results in 1,000 images per digit, for a total of 10,000 images passed into the neural network. After all of the additional images have been generated, we can run the number_identifier.py file to train the model and view prediction results.

