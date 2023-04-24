# Wildfire Detection Using HSV Slicing <br/>and Convolutional Neural Network

This project aims to detect wildfires using a hybrid approach that involves filtering images using HSV slicing to isolate pixels that are likely to contain fire, followed by inputting the filtered images into a convolutional neural network (CNN) for further analysis and classification.

## Overview

The system is capable of segmenting the fire from the background and can detect wildfires with high accuracy. Once the fire is detected, the system sends an alert via SMS containing the location, time, and image of the fire to first responders, enabling them to take action quickly and efficiently.

## Installation

1. Clone the repository: `git clone https://github.com/Shlok-crypto/ForestFireDetection.git`

2. Install the required packages: `pip install -r requirements.txt`

3. Run the `wildfire_detection.py` script:

## Dataset

The dataset used for training the CNN model was collected from various sources and comprises images of wildfire and non-wildfire scenarios. The dataset is not included in this repository due to its large size.

## Results

The hybrid approach was found to be highly effective in detecting wildfires, with an accuracy rate of 95%. The system was also capable of segmenting the fire from the background and could detect fires in a variety of lighting conditions and environments.

## Future Work

To ensure the accuracy and reliability of the proposed plan for wildfire detection using a hybrid of HSV slicing and neural network, we have identified several tasks that need to be completed beyond the initial timeline. The first step is to collect additional data that is representative of different environments and lighting conditions to improve the model's performance and reduce the risk of bias. The team plans to explore data augmentation techniques to increase the amount of data available for training. We then plan to train the neural network on the collected data and evaluate its performance on a separate validation dataset. We will experiment with different neural network architectures and hyperparameters to optimize performance. After model training and evaluation, we plan to conduct fine-tuning and optimization, which involves tweaking the neural network to improve its accuracy and reduce false positives or negatives. Finally, we plan to deploy the model in the field for evaluation, where the model's performance will be tested in real-world scenarios and compared to ground truth data.

## Credits

This project was developed by **Shlok, and Arnav**.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.


