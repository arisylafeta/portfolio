---
date: "2023-10-03"
title: "Credit Default Probability Estimation under the KMV framework"
image: "images/post-2/1.png"
categories: ["Finance"]
draft: false
---

<hr>

#### Introduction 
Corporate debt is an essential aspect of a company's capital structure, providing vital funds for growth and operations. However, it also introduces risk, as the company must meet its debt obligations to avoid default. This riskiness has led corporations like Moody's to develop sophisticated models to assess and manage credit risk. One such model is the KMV model, named after its creators Kealhofer, McQuown, and Vasicek. In this article, we will delve into the KMV model, explaining its theoretical underpinnings and providing a practical explanation through code. 
<hr>

#### Literature Review

The KMV model is a cornerstone in the field of credit risk assessment. Its development was inspired by the pioneering work of Robert Merton in the early 1970s. Merton's structural model laid the foundation for understanding credit risk by treating a firm's equity as a call option on its assets.

**Ideas proposed by Merton:** <a href="https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1974.tb03058.x">Merton's model</a> was revolutionary in linking the concepts of option pricing to corporate debt. He proposed that the equity of a firm could be viewed as a call option on its assets, with the strike price being the total debt obligation. The value of the firm's equity, therefore, depends on the value and volatility of its assets. 

$$ \displaylines{
     V_{E} = V_{A}*N(d_1) - e^{-rT}XN(d_2)\\\
     d_1 = \frac{ln(\frac{V_{A}}{X}) + (r+ \frac{\sigma^2_A}{2})}{\sigma \sqrt{T}}\\\
     d_2 = d_1 - \sigma_A \sqrt{T}
     }$$

If the value of the assets falls below the debt obligation, the firm defaults, similar to an option expiring out of the money. 

**What is the KMV Model:** <a href="https://www.moodysanalytics.com/-/media/whitepaper/before-2011/12-18-03-modeling-default-risk.pdf">The KMV model</a> is an advanced application of Merton's framework, specifically designed to estimate a firm's probability of default. It calculates the distance to default (DD), which measures how many standard deviations the firm's assets are above the default threshold. The model then translates this distance into a probability of default using the standard normal cumulative distribution function.

{{< image class="img-fluid rounded-6" title="image" src="/images/post-2/2.png" alt="KMV model">}}

The KMV model has been widely adopted in the financial industry due to its ability to provide a quantifiable measure of credit risk. It has influenced the development of various credit risk management tools and has become a standard method for estimating default probabilities in both academic and practical applications.
<hr>

#### Implementation

In this article we will provide you with a simplistic implementation of the KMV model, we will start by parsing the financial data using Yahoo Finance; estimate Asset value and volatility using a quasi-Newton method called the BFGS algorithm, and estimate the probability of default for a company and its implied credit spread.

{{< notice "note" >}}
  Readers can interact with the code firsthand by visiting the associated Github repo
{{< /notice >}}

{{< button "Github Repo" "https://github.com/arisylafeta/kmv/blob/main/kmv.ipynb.ipynb" >}}
<hr>

##### Getting the data

For the purposes of our implementation, we will gather the data using the Yahoo Finance, an easy to use and well maintained library. Specifically we will fetch:

1. Value of Equity - the firm's market capitalization.
2. Stock Volatility - we will approximate volatility by calculating the average standard deviation for a years worth of data.
3. Risk Free Rate - Depicted by CBOE 10 year treasury note
4. Total Debt - to be used for the calculation of default point
5. Total Assets - for use as an initial guess in our estimation

{{< notice "info" >}}
  Real world applications utilize more complex approximations of the underpinning variables. Furthermore, numerous parameters have been ommitted such as leverage and returns.
{{< /notice >}}

 ```python
import yfinance as yf
import pandas as pd

 def get_data(company):

    ticker = yf.Ticker(company)
    rf_rate = yf.Ticker('^TNX').fast_info['previous_close'] #risk free rate
    market_cap = ticker.fast_info['market_cap'] #market capitalization
    balance_sheet = pd.DataFrame(ticker.balance_sheet) #balance sheet
    total_debt = balance_sheet.loc['Total Debt'].iloc[0] #total debt
    total_assets = balance_sheet.loc['Total Assets'].iloc[0] #total assets

    #Calculating the historical volatility
    df = yf.download(company, period='1y') 
    stock_vol= df['Close'].pct_change().rolling(2).std().mean()

    return stock_vol, market_cap, total_debt, total_assets, rf_rate
 ```

##### Estimating Asset Value and Volatility

The KMV model requires the estimation of a firm's asset value and volatility, which are not directly observable from market data. Instead, these parameters are inferred from the observable equity value and equity volatility using an iterative method, such as the Newton-Raphson method or our BFGS algorithm.

**Relations between Equity and Assets:** In the Merton model, the equity of a firm is treated as a call option on its assets. The value of this option, and hence the equity value, is influenced by the underlying asset value. The relationship is given by the Black-Scholes option pricing formula as shown above.

$$V_{E} = V_{A}*N(d_1) - e^{-rT}XN(d_2)$$

and for volatilty the relationship can be derived as:

$$\sigma_E = \frac{V_A}{V_{E}}\sigma_{A}N(d_1) $$


We will start with an intial guess for the Asset Value and Volatility, and calculate the implied Equity Value and Volatility based on the formulas above. Then we will try to minimize iteratively the squared difference between the calculated and fetched values, using stats.optimize.minimize function.

```python
 def Estimate(x):
    asset_val, asset_vol = x

    # Black-Scholes-Merton model
    d1 = (np.log(asset_val/default_point) + (rf_rate + asset_vol**2 / 2) * T) / (asset_vol * np.sqrt(T))
    d2 = d1 - asset_vol * np.sqrt(T)

    #Implementation of the formulas
    equity_val = asset_val*norm.cdf(d1) - np.exp(-rf_rate * T) * default_point * norm.cdf(d2)
    equity_vol = asset_vol*norm.cdf(d1)*asset_val/equity_val
    
    # Residual 1: difference between calculated value and market value of equity
    residual_1 = (stock_val - equity_val)**2 

    # Residual 2: difference between calculated volatility and market volatility
    residual_2 = (stock_vol - equity_vol)**2 
    
    # Objective function combining residuals
    return residual_1 + residual_2

# Initial guesses
initial_guess = [assets, stock_vol]

# Bounds to ensure positive asset value and volatility
bounds = [(1, None), (0.0001, None)]

# Optimization
result = minimize(Estimate, initial_guess, bounds=bounds, method='L-BFGS-B')
asset_val, asset_vol = result.x
```

##### Probability of Default and Implied Credit Spread

Once the asset value and volatility are estimated using the KMV model, the next step is to calculate the Probability of Default (PD) and the Implied Credit Spread (ICS).

**Probability of Default (PD):** The PD is a measure of the likelihood that a firm will default on its debt obligations within a specified time horizon. In the KMV model, the PD is estimated using the Distance to Default (DD), which is a measure of how many standard deviations the firm's assets are above the default point. The formula for the PD is:

$$ PD = N(-DD) = N(-d_2) $$

**Implied Credit Spread (ICS):** The ICS is the additional yield that investors require to compensate for the risk of default. It is the difference in yield between a risky debt instrument and a risk-free bond. The ICS can be estimated from the PD using the following formula:

$$ ICS = - \frac{ln(1-PD)}{T} - \frac{ln(1+r)}{T} $$

```python
def ProbabilityofDefault():
    PoD = norm.cdf(-d2)
    return PoD
    
def ImpliedCreditSpread():
    credit_spread = -1/T * (np.log(1 - ProbabilityofDefault()) + np.log(1+rf_rate))
    return credit_spread
```

#### Conclusion

We've given a simple overview and implementation of estimating probability of default under the KMV framework, readers are encouraged to explore further articles on the topic. Future improvements might include more advanced numerical methods for optimization.