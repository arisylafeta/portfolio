---
date: "2023-11-10"
title: "Portfolio Allocation using FamaFrench5 and K-Means Clustering"
image: "images/post-3/cover.png"
categories: ["Finance", "Data Science & AI"]
draft: false
---
<hr>

#### Introduction

In the financial world, where approximately 5 trillion dollars are invested through factor investing, the quest for optimal portfolio allocation is more pertinent than ever. In this project we will walk through an advanced factor investing strategy. 

{{< notice "note" >}}
  Readers can interact with the code firsthand by visiting the associated Github repo
{{< /notice >}}

{{< button "Github Repo" "https://github.com/arisylafeta/Portfolio-Investing-with-FamaFrench5-and-KMeans/blob/main/FamaFrench5-KMeans.ipynb" >}}
<hr>



#### Data Processing

As in every data science project, this section is the most important but also the most grueling. I will list the important steps, but users are encouraged to visit the repo for a more comprehensive explanation. We've imported the data using the Yahoo Finance Library, getting the NASDAQ100 constitutents from Wikipedia's website. We'll be using a 10 year horizon for our analysis and backtesting.

```python
import yfinance as yf

nas100 = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]

nas100['Ticker'] = nas100['Ticker'].str.replace('.','')
symbols_list = nas100['Ticker'].to_list()

#Download SP500 companies price data
end_date = '2023-09-01'
start_date = '2013-09-01'
df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date,
                 interval='1d').stack()


#Assign multi-index
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()
df
```

 After calculating the daily frequency indicators we'll aggregate the data on a monthly basis to reduce training time. 

 ```python
 #Calculate different time-horizon returns as features
data = (df.drop(columns=['open', 'high', 'low', 'close', 'volume']) # Drop OHLCV columns
          .unstack('ticker')                 # Unstack ticker level to operate on the date level
          .resample('M').last()               # Resample to monthly frequency, taking the last value
          .stack('ticker')                    # Restack the ticker level back
          .dropna())                          # Drop NA values that might arise from resampling

data
 ```
<hr>

##### Feature Engineering
A good set of features is essential for uncovering the patterns within the data and enhancing the predictive power of our models. This process involves producing functions of the raw data to extract new dimensions of information. Thankfully, the financial literature is rich in technical indicators readily available, although only a handful of them actually add relevant information. We will be using the pandas-ta library for common indicators.

{{< collapse "**Garmann Klass Volatility**" >}}
This approach, developed by Robert German and Richard Klass, aims to provide a more accurate measure of volatility by incorporating the range of price movements, thereby offering a comprehensive view of market volatility beyond what is captured by simpler models based solely on closing prices.

```python
#Calculate intraday volatility
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)
```
{{< /collapse >}}
{{< collapse "**Relative Strength Index**" >}}
  The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements, oscillating between 0 and 100. It is a very popular tool for gauging market momentum. 

```python
#Calculate RSI 
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
```
{{< /collapse >}}
{{< collapse "**Bollinger Bands**" >}}
  Bollinger Bands consist of a moving average as the middle band and two standard deviation lines above and below it as the upper and lower bands, respectively. Great for identifying volatility and relative price levels.

```python
#Calculate Bollinger Bands
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])                                                     
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])
```
{{< /collapse >}}
{{< collapse "**Average True Range**" >}}
  
ATR provides a numerical value that takes into account the high, low and close information over a period, representing the true range of prices. Great for measuring market volatility. 

```python
#Calculate Bollinger Bands
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())
df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)
```

{{< /collapse >}}
{{< collapse "**Moving Average Convergence Divergence**" >}}
  
MACD is another great momentum indicator that shows the relationship between a faster versus a slower moving average of an asset.

```python
#Calculate MACD
def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())
df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)
```
{{< /collapse >}}
{{< collapse "**Previous Returns**" >}}
  We will also be adding 1, 2, 3, 6, 9, 12 month returns of assets as features of dataset. Simultanously, to reduce erroneous results from outliers, we will remove 0.5% of extreme observations on either side.

```python
#Calculate different time-horizon returns as features
def calculate_returns(df):
    outlier_cutoff = 0.005 #Exclude top and bottom 0.5% of observations(outliers)
    lags = [1, 2, 3, 6, 9, 12] #Different time-horizon returns
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                              .pct_change(lag)
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1))
    return df    
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

data
```
{{< /collapse >}}





 


<hr>

##### Fama-French 5 Factors

The Fama-French Five-Factor model is an extension of the Fama-French Three-Factor model, introduced by Eugene Fama and Kenneth French to better explain the returns of stocks and portfolios. This model incorporates five key factors: market risk, size of firms (small vs. large), value vs. growth stocks, profitability (robust minus weak), and investment patterns (conservative vs. aggressive). It suggests that the risk and return of a portfolio are influenced not only by market risk but also by these additional factors, which can account for a significant portion of the differences in returns across diversified portfolios. We will be importing the general factor data using pandas_datareader library, and then use an OLS rolling regression to calculate the factor betas for every stock and add them as features. 

```python
import pandas_datareader as pdr

#Download Fama-French 5 factors data
factor_data = pdr.data.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
#Resample to monthly frequency
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
#Merge with 1m return data
factor_data = factor_data.join(data['return_1m']).sort_index()

factor_data
```
```python
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Perform OLS Regression to calculate beta of each factor for each stock.
betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

betas
```
<hr>

##### K-Means Clustering

K-means clustering is a popular unsupervised machine learning algorithm used to partition a dataset into K distinct, non-overlapping clusters based on the similarity of data points. It iteratively assigns each data point to one of K clusters by minimizing the variance within each cluster. 

Our portfolio allocation strategy will be based on investing on assets that are showcasing greater momentum compared to their counterparts. Therefore, we shall cluster our data according to the RSI values. We will split our data into 4 clusters differing stocks with high-selling, selling, buying, and high-buying momentum. The cluster index of every stock will be added as a column to the data.

```python
#Define cluster RSI values (high-selling, selling, buying, high-buying)
target_rsi_values = [30, 45,  55, 70]

#Create initial centroids, fix RSI values, and set other values to 0
initial_centroids = np.zeros((len(target_rsi_values), 18))
initial_centroids[:, 1] = target_rsi_values
initial_centroids
```
```python
from sklearn.cluster import KMeans

#Create a function to get clusters
def get_clusters(df):
    #Apply KMeans clustering to each month of data separately
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_
    return df
data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

data
```
The clusters created are 18-dimensional objects that group the assets based on their momentum. For every month in our dataframe, we will select the assets under the 'high-buying' cluster and place them in a dictionary. Our portfolio, at the beginning of every month, will be allocated according to that dictionary. Below you can find some randomly selected images of the clusters with ATR values on the x-axis, and RSI values on the y-axis.

{{< image class="img-fluid rounded-6" title="image" src="/images/post-3/1.png" alt="element">}}
{{< image class="img-fluid rounded-6" title="image" src="/images/post-3/2.png" alt="element">}}

<hr>

#### Portfolio Optimization

Portfolio optimization is a process used in finance to select the best portfolio out of a set of all possible portfolios that offers the highest expected return for a given level of risk, or equivalently, the lowest risk for a given level of expected return. The father of Modern Portfolio Theory, Harry Markowitz, introduced the concept of the Efficient Frontier- the set of optimal portfolios in the risk and return dimensions. 

We concluded the previous section by creating a list of assets suitable for investing for every month. Now, we will use the PyPortfolioOpt library to optimize the weights in our portfolios, so we lie in the Efficient Frontier

```python
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

#Create a function to optimize weights
def optimize_weights(prices, lower_bound=0):
    #Calculate expected returns 
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    #Calculate covariance matrix
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    #Create an EfficientFrontier object
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')
    #Get the weights that maximize the Sharpe ratio
    weights = ef.max_sharpe()
    
    return ef.clean_weights()
```

We try to fit our Portfolio for every month, and if the algorithm fails we allocate an equal weighing to each asset. Lastly, we calculate the according returns from our portfolio and compare them to a simply buy & hold strategy against the NASDAQ.
```python
# Define a function to calculate portfolio returns based on optimization or equal weighting.
def calculate_portfolio_returns(new_df, fixed_dates, returns_dataframe):
    portfolio_df = pd.DataFrame()

    # Iterate over each start date specified in the 'portfolio_allocator' dictionary.
    for start_date in portfolio_allocator.keys():
        try:
            # Calculate the end date for the current period as the last day of the start date's month.
            end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            # Retrieve the list of tickers to be included in the optimization for the current start date.
            cols = portfolio_allocator[start_date]

            # Determine the date range for optimization: start 12 months before the current start date and end one day before.
            optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
            optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')

            # Extract the adjusted close prices for the tickers and time period identified for optimization.
            optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]

            try:
                # Attempt to calculate optimal weights using a predefined optimization function.
                # The lower bound for weights is set dynamically based on the number of assets.
                weights = optimize_weights(prices=optimization_df,
                                           lower_bound=round(1 / (len(optimization_df.columns) * 2), 3))
                success = True
            except Exception as e:
                # If optimization fails, print an error message and continue with equal weights.
                print(f'Max Sharpe Optimization failed for {start_date} due to {e}, Continuing with Equal-Weights')
                success = False

            # If optimization was not successful, use equal weighting for all tickers.
            if not success:
                equal_weight = 1 / len(cols)
                weights = {ticker: equal_weight for ticker in cols}

            # Convert the weights dictionary into a DataFrame for easier calculations.
            weights_df = pd.DataFrame(weights, index=[0])

            # Retrieve returns data for the current period and tickers.
            temp_df = returns_dataframe.loc[start_date:end_date, cols]
            # Multiply the returns by the weights to calculate weighted returns.
            temp_df = temp_df.mul(weights_df.values, axis=1)
            # Sum the weighted returns across all tickers to get the portfolio's strategy return for each day.
            temp_df['Strategy Return'] = temp_df.sum(axis=1)

            # Append the calculated strategy returns to the portfolio returns DataFrame.
            portfolio_df = pd.concat([portfolio_df, temp_df[['Strategy Return']]], axis=0)

        except Exception as e:
            # If an error occurs during processing, print an error message.
            print(f'Error processing {start_date}: {e}')

    # Remove any duplicate entries in the portfolio returns DataFrame.
    portfolio_df = portfolio_df.drop_duplicates()
    # Return the final portfolio returns DataFrame.
    return portfolio_df
```
{{< image class="img-fluid rounded-6" title="image" src="/images/post-3/output.png" alt="element">}}


<hr>

#### Conclusion 

We've implemented successfully a portfolio allocation strategy that uses components of Arbitrage Pricing Theory and blends them seemlessly with advanced machine learning techniques such as K-Means Clustering to select the assets available for investment. Our strategy seems to beat a buy & hold strategy of the NASDAQ index, although more back-testing is needed to confirm the existence of alpha. There are many backtesting biases that research can unknowingly fall into, so readers are urged to do their own research and implement this strategy at their own risk. Further imporvements to the strategy might include allowing for short-selling, accounting for transaction costs, and using more advanced features.

<hr>
