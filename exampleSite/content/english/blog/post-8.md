---
date: "2022-04-13"
title: "Credit Value Adjustment via Monte Carlo Simulation"
image: "images/post-8/cover.png"
categories: ["Finance", 'Credit Risk']
draft: false
---

#### Introduction

In the wake of the 2008 financial crisis, the importance of accurately accounting for the risk of counterparty default in derivative transactions has become paramount. This risk is quantified in a financial metric known as Credit Valuation Adjustment (CVA), which has been incorporated into pricing and risk management practices across the financial industry.

CVA represents the difference between the risk-free portfolio value and the true portfolio value that considers the possibility of a counterparty's default. In essence, it is the market value of counterparty credit risk. This adjustment is crucial when trading over-the-counter (OTC) derivatives, where contracts are not exchanged through a central party and thus expose the parties to the risk of financial loss if the counterparty fails to uphold their end of the contract. The CVA for a portfolio can be expressed by the formula 

$$CVA = (1- RR) \cdot \sum_{i=1}^n EAD_i \cdot P(D_i) \cdot DF_i$$

where
1. RR is the Recovery Rate, the part of the exposure that can be recovered in the event of a default.
2. EAD is the Expected Exposure at default time i.
3. P(D) is the probability of Default at time i.
4. DF is the Discount Factor at time i.

Calculating CVA is a multi-step process that involves modeling future exposures to the counterparty, estimating the probability of default, and discounting these values back to the present value. It requires a robust framework for simulating future market conditions and valuing potential exposures at different points in time. In this project we will present an algorithm to evaluate CVA's on a Portfolio of Interest Swaps using Monte Carlo Simulation 

#### Implementation

For a comprehensive view on the code displayed, please refer to the associated Github Repo. 

 {{< button "Gtihub Repo" "https://github.com/arisylafeta/Simple-CVA-Calculation" >}}

##### Setting Up the Market Data

The first step in any financial simulation is to set up the market data accurately. This includes defining the current interest rates, which are fundamental to discounting future cash flows. In our project, we have used the QuantLib library to create a flat forward yield curve, and will use the EURIBOR 6month rate instead of the SONIA overnight rate, a simplification used for illustrative purposes. Here is how we set it up:

```python
# Setting evaluation date
today = ql.Date(7,4,2022)
ql.Settings.instance().setEvaluationDate(today)

# Define the interest rate
rate = ql.SimpleQuote(0.03)
rate_handle = ql.QuoteHandle(rate)
dc = ql.Actual365Fixed()

# Define the yield term structure
yts = ql.FlatForward(today, rate_handle, dc)
yts.enableExtrapolation()
hyts = ql.RelinkableYieldTermStructureHandle(yts)
t0_curve = ql.YieldTermStructureHandle(yts)
euribor6m = ql.Euribor6M(hyts)
```
<hr>


##### Constructing a Portfolio of Swaps
Interest rate swaps are important tools in the management of financial risk. In these contracts, parties agree to exchange cash flows based on different interest rates—one fixed and one floating—over a specified period. From a CVA perspective, interest rate swaps are significant because they are subject to counterparty credit risk. The fixed nature of the payments in one leg of the swap means that if the counterparty defaults, there's a risk of loss if the market has moved in the non-defaulting party's favor. Thus, the Credit Valuation Adjustment (CVA) quantifies the risk of counterparty default by adjusting the valuation of the swap to reflect the possibility of this loss.

In our code we constructs a plain vanilla interest rate swap, which is the most straightforward type of swap agreement.  

```python
def makeSwap(start, maturity, nominal, fixedRate, index, typ=ql.VanillaSwap.Payer):

    end = ql.TARGET().advance(start, maturity)
    fixedLegTenor = ql.Period("1y")
    fixedLegBDC = ql.ModifiedFollowing
    fixedLegDC = ql.Thirty360(ql.Thirty360.BondBasis)
    spread = 0.0
    fixedSchedule = ql.Schedule(start, end, fixedLegTenor, index.fixingCalendar(), fixedLegBDC,
                                fixedLegBDC, ql.DateGeneration.Backward, False)
    
    floatSchedule = ql.Schedule(start, end, index.tenor(), index.fixingCalendar(), index.businessDayConvention(),
                                index.businessDayConvention(),ql.DateGeneration.Backward, False)
    
    swap = ql.VanillaSwap(typ, nominal,fixedSchedule, fixedRate, fixedLegDC,
                          floatSchedule, index, spread, index.dayCounter())
    
    return swap, [index.fixingDate(x) for x in floatSchedule][:-1]

portfolio = [
    makeSwap(today + ql.Period("2d"), ql.Period("5Y"), 1e6, 0.03, euribor_index),
    makeSwap(today + ql.Period("2d"), ql.Period("4Y"), 5e5, 0.03, euribor_index)
]

```
<hr>

##### Monte Carlo Evaluation of Exposure

The Monte Carlo simulation method plays a pivotal role in the evaluation of potential future exposures (PFE) in a portfolio of financial instruments. This probabilistic approach involves simulating numerous interest rate scenarios to capture a wide range of possible future market conditions and assess the impact on the portfolio's exposure over time.


{{< image class="img-fluid rounded-100" title="image" src="/images/post-8/1.png" alt="element">}}

By running thousands of simulations, each representing a possible future state of the market, we obtain a distribution of exposure values at each time step. These simulations take into account the stochastic nature of interest rates.

<hr>

##### Pricing the Swaps

The next step is the valuation of the portfolio under different market conditions. The portfolio consists of a collection of interest rate swaps, each with its own set of cash flows and sensitivities to market interest rates. To assess the portfolio's value under each scenario, we've simulated the evolution of interest rates and now we calculate the Net Present Value (NPV) of each swap at various points in time.

**The "npv_cube"  Data Structure**

We introduce a three-dimensional array, npv_cube, to store the NPVs. Its dimensions correspond to the number of scenarios (N), the number of time points in our evaluation grid (len(date_grid)), and the number of swaps in our portfolio (len(portfolio)). This structure allows us to track the value of each swap across each scenario and time point.

**Pricing Engine and Discount Curve**


For each simulated scenario and at each time point, we perform the following steps:

{{< collapse "1. Set Evaluation Date" >}}
  We update the evaluation date in QuantLib's settings to the current date in the simulation grid. This ensures that all valuations are consistent with the 'as of' date of each scenario.
{{< /collapse >}}

{{< collapse "2. Construct Discount Curve" >}}
 We build a discount curve, yc, using the zero-coupon bond rates generated from the Monte Carlo simulation. This curve is essential for discounting future cash flows to their present value.
{{< /collapse >}}

{{< collapse "3. Enable Extrapolation" >}}
We allow the discount curve to extrapolate beyond the last specified date to accommodate valuations at any point within the simulation horizon.
{{< /collapse >}}

{{< collapse "4. Link Yield Term Structure" >}}
  The discount curve is then linked to the yield term structure handle, hyts, which is used by the pricing engine to discount cash flows.
{{< /collapse >}}

{{< collapse "5. Fixing Dates and Rates" >}}
  If the current date is a valid fixing date for the floating rate index (e.g., EURIBOR), we record the simulated fixing rate. This rate is used to determine the floating cash flows for the swaps.
{{< /collapse >}}

{{< collapse "6. Calculate NPV" >}}
  Finally, we calculate the NPV for each swap in the portfolio using the configured pricing engine, which uses the discount curve we've established. The results are stored in npv_cube.
{{< /collapse >}}


```python
#Price the portfolio of swaps under each scenario
npv_cube = np.zeros((N,len(date_grid), len(portfolio)))
for p in range(0,N):
    for t in range(0, len(date_grid)):
        date = date_grid[t]
        ql.Settings.instance().setEvaluationDate(date)
        ycDates = [date, 
                   date + ql.Period(6, ql.Months)] 
        ycDates += [date + ql.Period(i,ql.Years) for i in range(1,11)]
        yc = ql.DiscountCurve(ycDates, 
                              zero_bonds[p, t, :], 
                              ql.Actual365Fixed())
        yc.enableExtrapolation()
        hyts.linkTo(yc)
        if euribor6m.isValidFixingDate(date):
            fixing = euribor6m.fixing(date)
            euribor6m.addFixing(date, fixing)
        for i in range(len(portfolio)):
            npv_cube[p, t, i] = portfolio[i][0].NPV()
    ql.IndexManager.instance().clearHistories()
ql.Settings.instance().setEvaluationDate(today)
hyts.linkTo(yts)
```
```python
# Calculate the discounted npvs
discounted_cube = np.zeros(npv_cube.shape)
for i in range(npv_cube.shape[2]):
    discounted_cube[:,:,i] = npv_cube[:,:,i] * discount_factors
```
{{< image class="img-fluid rounded-100" title="image" src="/images/post-8/2.png" alt="element">}}

<hr>

##### Calculate Potential Future Exposure and CVA

Potential Future Exposure (PFE) is a measure of the maximum expected credit exposure over a specified time period at a given confidence interval. We calculate PFE for each time point in our simulation grid

```python
# Calculate the expected exposure
E = portfolio_npv.copy()
E[E<0]=0
EE = np.sum(E, axis=0)/N

# Calculate the PFE curve (95% quantile)
PFE_curve = np.percentile(E, 95, axis=0)
```

Based on this data we can calculate and display important variables for the nature of our portfolio such as the hazard rate, default probability and default density 

{{< image class="img-fluid rounded-100" title="image" src="/images/post-8/3.png" alt="element">}}

Finally from this data we simply calculate the CVA based on the formula showcased in the introduction. 

```python
# Calculation of the CVA
recovery = 0.4
CVA = (1-recovery) * np.sum(dEE[1:] * dPD)
CVA
```



