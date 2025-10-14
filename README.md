Applied Machine Learning: Predictive Analysis of Financial Time Series
This repository contains the Python code and resources to replicate the research study "Applied Machine Learning: Predictive Analysis of Financial Time Series". The project evaluates the effectiveness of machine learning algorithms, specifically feedforward neural networks, for short-term price prediction of various financial instruments.



The core objective is to identify how different data preparation and modeling techniques impact the predictive accuracy of the models. This is achieved by systematically testing various configurations and measuring their performance through simulated backtesting.



Methodology
The study is implemented in Python and relies on several key libraries for its operations, including yfinance for data retrieval, numpy for mathematical operations, PyTorch for building neural network models, and matplotlib for data visualization. The entire workflow is encapsulated in a single module named project_functions for clarity and ease of use.





The process involves four main stages:


Data Retrieval: Historical time series data for various asset classes (Stocks, Currencies, Cryptocurrencies, and Commodities) are downloaded using the yfinance library. The study analyzes data sampled at different intervals (1 minute, 1 hour, and 1 day) to test the impact of data frequency.




Data Processing & Normalization: Raw time series data is transformed into input/output datasets suitable for model training. A key part of the research involves comparing two normalization techniques:



Min-Max Scaling: Normalizes each input sequence to a range of [0,1].


Logarithmic Difference: Normalizes each input sequence by subtracting the last element from all other elements, preserving information about volatility.


Model Training: The study primarily uses a Feedforward Neural Network implemented with PyTorch. The model is trained on a portion of the dataset, with hyperparameters like the number of epochs and the training data ratio being adjustable.




Performance Evaluation: The model's predictive power is evaluated using a backtesting simulation on unseen data. The cumulative return of a strategy based on the model's predictions is compared against two benchmarks: a simple "buy and hold" strategy and the inverse of the model's strategy. The success of a model is determined by how frequently its strategy outperforms the opposite strategy.


How to Replicate the Study
You can use the code in this repository to replicate the data collection and analysis. The key parameters of the study can be configured to test different scenarios.

The main configurable parameters are:

Instrument: Stocks (Apple, Intel, Amd), Currencies (Euro, Sterling, Yen), Cryptocurrencies (Bitcoin, Ethereum, Solana), and Commodities (Gold, Silver, Oil).

Sampling Interval: 1 Minute, 1 Hour, 1 Day.

Observation Window: 32, 64, 128 data points.

Normalization Technique: min-max or logarithmic difference.

Epochs: Number of training iterations (e.g., 64, 256, 512, etc.).

Key Findings
The research yielded several key insights into the application of neural networks for financial forecasting:


Normalization is Crucial: The logarithmic difference normalization technique proved significantly more effective, with a success rate of p=0.75±0.14 compared to p=0.58±0.16 for min-max scaling.



Data Frequency Matters: Models trained on lower-frequency data (daily) demonstrated more robust predictive capabilities than those trained on high-frequency data (hourly or minute).



Asset Predictability Varies: Cryptocurrencies and stocks were found to be more predictable using this methodology than currencies and commodities. For instance, in one phase of the study, the success rate for cryptocurrencies reached p=0.96±0.07.




Observation Window Length: The length of the input sequence (observation window) did not have a significant impact on model performance, suggesting that the most recent data points hold the most predictive power.


Disclaimer
The results presented in this study are based on an idealized simulation. They do not account for real-world market factors such as transaction costs, slippage, or data latency, which can significantly impact the performance of any trading strategy. This project is for academic and research purposes only and should not be considered financial advice.
