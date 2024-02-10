# Stock Market Prediction using LSTM

This notebook contains code for predicting stock prices using LSTM (Long Short-Term Memory) neural networks. The model is trained on historical stock data and then used to make predictions on unseen data.

## Introduction

This notebook demonstrates the use of LSTM neural networks for stock price prediction. It utilizes historical stock data for training and testing the model. The predictions are made for multiple stocks simultaneously.

## How to Use

To use this notebook:

1. Download and Open the notebook.
2. Ensure you have necessary access permissions to download data from Kaggle.
3. Change the csv_path variable to the path where the dataset is downloaded in your machine.
4. Run each cell sequentially to execute the code and see the results.
5. Customize the code as needed for your specific use case.

## Dependencies

This notebook relies on the following libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- TensorFlow (via Keras)

## Data

The data used in this notebook is retrieved from a Kaggle dataset titled "AMZN, DPZ, BTC, NTFX adjusted May 2013-May2019". This dataset includes historical stock prices for Amazon (AMZN), Domino's Pizza (DPZ), Bitcoin (BTC), and Netflix (NFLX) from May 2013 to May 2019. You can access the dataset on Kaggle through the following link: [AMZN, DPZ, BTC, NTFX adjusted May 2013-May2019](https://www.kaggle.com/datasets/hershyandrew/amzn-dpz-btc-ntfx-adjusted-may-2013may2019).

The dataset provides adjusted stock prices, which account for corporate actions such as stock splits and dividends, allowing for more accurate analysis and prediction. It serves as the foundation for training and testing the LSTM model for stock price prediction in this notebook.

## Model Architecture

The LSTM model architecture consists of:

- Two LSTM layers with 50 units each and ReLU activation function.
- One dense output layer for predicting multiple stocks.

## Results

The notebook provides visualizations of true stock prices versus predicted prices for each stock. Additionally, it calculates the Mean Squared Error (MSE) as a measure of model performance.

## Limitations

- Stock market prediction is inherently uncertain and subject to various factors that are difficult to model accurately.
- The model's performance may vary depending on the quality and quantity of the available data.
- This notebook provides a basic implementation and can be further optimized for better results.

## Acknowledgements

- This notebook is for educational purposes and does not constitute financial advice.
