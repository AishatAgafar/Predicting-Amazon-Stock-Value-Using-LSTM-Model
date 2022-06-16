# Amazon-Stock-Prediction
Abstract

The closing price is considered the most accurate evaluation of a stock or other security until trading resumes on the next trading day. Here, we make use of the LSTM model to predict the closing price of Amazon stocks. 

Index Terms—LSTM, Amazon, Closing value, RNN, ANN, Machine learning 

I. INTRODUCTION

Stock market prediction is an act of trying to get some information on the future outlook of a company’s stock or other financial instrument traded on an exchange. Stock market prediction and analysis are difficult because of market volatility and a variety of other dependent and independent variables that influence the value of a certain stock in the market. Closing prices are useful markers for investors to use to assess changes in stock prices over time. The closing figure on one day can be compared to closing figure on the previous day, 30 days earlier, or a year earlier to measure the changes in the market sentiment toward that stock. It is because of the importance of this parameter that many researches have been dedicated towards its forecast. For example, Shen and Shafiq, 2020 [3] collected 2 years data from the Chinese stock market for predicting the stock market price trend. In 2021, Gosh et al [4] compared the effectiveness of both random forests and LSTM in forecasting the out-of-sample directional movements of constituent stocks of the Standard Poor’s 500 (SP 500) from January 1993 till December 2018 for intraday trading. [5] used the artificial neural network (ANN) and the random forest techniques for predicting the next day closing prices for five companies belong to different sectors. In order to improve the prediction accuracy, [6] incorporated the investor sentiment information with deep learning to forecast the market stock index. In this work, our focus is to predict the closing price of Amazon stocks using LSTM model.

II. THE LONG SHORT-TERM MEMORY

The Long-short term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike the standard feedforward neural networks, LSTM has feedback connections. It can process not only single data points (such as images), but also entire sequences of data (such as speech or videos). A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. LSTM networks are suitable for classifying, processing and making decision based on time series. LSTMs were developed to deal with the vanishing gradient problem that can be encountered when training traditional RNNs. 


A. LSTM with a forget gate


    ft = σg(Wfxt + Ufht−1 + bf )
    it = σg(Wixt + Uiht−1 + bi)
    ot = σg(Woxt + Uoht−1 + bo)
    ˜ct = σc(Wcxt + Ucht−1 + bc)
    ct = ft ◦ ct−1 + it ◦ ˜ct
    ht = ot ◦ σh(ct)

where the initial values are co = 0 and ho = 0 and the operator ◦ denotes the Hadamard product (element-wise product). The subscript t indexes the time step. 

Variables:
  1) xt ∈ Rd: input vector to the LSTM unit.
  2) ft ∈ (0, 1)h: forget gate’s activation vector.
  3) i  t ∈ (0, 1)h: input/update gate’s activation vector.
  4) ot ∈ (0, 1)h: output gate’s activation vector.
  5) ht ∈ (−1, 1)h: vector of the LSTM unit.
  6) ˜ct ∈ (−1, 1)h: cell input activation vector.
  7) ct ∈ Rh: cell state vector.
  8) W ∈ Rh×d, U ∈ Rh×h and b ∈ Rh: weight matrices and bias vector parameters which need to be learned during training where the superscripts d and h refer to the number of input features and the number of hidden units, respectively.


Activation functions:
  1) σg : sigmoid function.
  2) σc : hyperbolic tangent function.
  3) σh : hyperbolic tangent function or the identity function.


III. DATA COLLECTION

Amazon’s stock is listed on Yahoo finance (https://finance.yahoo.com) and their value is updated every working day of the stock market. The stock market day does not include Saturdays and Sundays. We are going to use Fig. 1. A typical LSTM model. these data remotely from January 1st 2010 to January 31st 2022. The data has 6 columns without the date column: The opening value of the stock, the highest and lowest values as well as the closing values are listed at the end of each day. The adjusted close value reflects the stock’s value after the dividends have been declared. Furthermore, the total volume of the stocks in the market is provided.


![image](https://user-images.githubusercontent.com/102318984/174089877-ebba1f28-bc6a-43eb-9295-da24728e8540.png)

http://localhost:8888/nbconvert/html/Documents/Python%20file/Amazon%20stock%20prediction-Copy1.ipynb?download=false


A. Target Variable and Features

The variable to be predicted is the adjusted closing price of amazon stock. We choose “Open,” “High,” “Low” and “Volume” as the features to serve as the independent variable for the target variable “Adj Close.”


B. Training and Testing data

We would divide the entire dataset into the training and the testing data. The LSTM model will be trained on the data in the training set and tested for accuracy on the testing set. We are going to use 90% and 10% of the data as the training and testing data, respectively.


C. Building LSTM Model for Amazon Closing Price Prediction

We build a Sequential Keras model with one LSTM layer. The LSTM layer has 32 units and is followed by one dense layer of one neuron. We compile the model using Adam optimizer and mean squared error as the lost function.


IV. RESULTS


![image](https://user-images.githubusercontent.com/102318984/174090397-d799b496-6060-4b14-bd84-5b8d8dfda09f.png)


The graph in Figure 2 shows the true adjusted closing price of Amazon stock in blue and the LSTM-predicted closing value in orange. From the graph, it is clear that the LSTMpredicted values not only follow the pattern of true values but they are very close to the actual value. This demonstrates the power of LSTM model. Given, any Amazon features, one can input them into the LSTM-predicted model to get insights into what the adjusted closing value might look like. This model could help potential investors in Amazon’s stock to make informed decision in evaluating the changes in the stock values over time. In conclusion, stock market predictions are difficult due a number of factors. However, with the advent of machine learning and its algorithms, we can get more and more accurate predictions by adjusting the parameters and perhaps increasing the number of layers in the design of the machine.


REFERENCES


[1] F. A. Gers, J. Schmidhuber, F. Cummins, “Learning to Forget: Continual
Prediction with LSTM,” Neural Computation, 12 (10), 2451 – 2471,
2000.

[2] S. Hochreiter, J. Schmidhuber, “Long short-term memory,” Neural
Computation, 9 (8), 1735 – 1780, 1997.

[3] J. Shen, M. O. Shafiq, “Short-term stock market price trend prediction
using comprehensive deep learning system,” Journal of Big Data, 7:66,
2020.

[4] P. Gosh, A. Neufeld, J.K. Sahoo, “Forcasting directional movements
of stock prices for intraday trading using LSTM and random forests”,
Finance Research Letters, Volume 46, Part A, 2022.

[5] M. Vijh, D. Chandola, V.A. Tikkiwal, and A. Kumar, ”Stock Closing
Price Prediction using Machine Learning Techniques”, Procedia Computer
Science, 167, 599 - 606, 2020.
