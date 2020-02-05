# Experimentation using ML for market prediction
#### Why use ML in market prediction 
Market data is incredibly noisy and extremely hard to predict. 
One can view the market as one massive multivariate function with many confounding factors. 
Because of the massive amount of factors it is practically impossible for a human to predict the direction the market, 
or an individual stock, will go. This is where machine learning can come into play. In theory, it is possible to build a 
model that could predict, with relatively high accuracy, the direction of the market; of course, given enough
data, complexity, and computing resources. 

#### What this is: 
This was an attempt at using ML for market prediction. A variety of attempts were made using CNN, RNN, TCN, and LSTM. 
In addition, the data had been modified in many different ways, adding different indicators (standard parameters and optimized parameters). 
However, attempting to use indicators never served great use because there are many indicators that would signal bullish, while others signal bearish, and of course other signal neutral.  
Removing indicators did not seem to affect the the model in any significant way. Sometimes it helped some times it produced many false positives. 
The greatest improvement on performance seemed to be increasing the time interval, which helped de-noise the data (I found 15min to hourly were most effective).
However, this is varies greatly on the trading strategies being used. However, seeing as the market is net-negative (contrary to popular belief), the best results were just barely better than random guesses.

#### What could be improved 
As any active day or swing trader would know, there is a must know analytical technique know as multi timeframe analysis. 
This is basically the idea of looking at data in different time aggregations to reveal both shorter and longer term trends (for example using 5min charts along side 30min charts).
 
Ideally, several models, operating on different time frames could be used. And have some logic like the following: 
```
model5 = someMLmodel(timeInterval="5min")
model30 = someMLmodel(timeInterval="30min")
modelDaily = someMLmodel(timeInterval="daily")
positionSize = 0.0


while inMarketHours:    
    newData = getNewData()

    modelDaily.update(newData)
    trendingDaily = modelDaily.predictTrending()
    if trendingDaily="bull": 
        model30.update(newdata)
        trending30 = model30.predict()
        positionSize = .25   # this is arbirtray  
        if trending30="bull": 
            model5.update(newData)
            trending5 = model5.preidct()  
            positionSize = .5
            if trending5="bull": 
                positionSize = 1
    elif trendingDaily="bear": 
        model30.update(newdata)
        trending30 = model30.predict()
        positionSize = -0.25   # this is arbirtray  
        if trending30="bear": 
            model5.update(newData)
            trending5 = model5.preidct()  
            positionSize = -0.5
            if trending5="bear": 
                positionSize = -1
    else: 
        positionSize = 0
```
Basically, you could use position sizing to mitigate risk taking. If the stock seems like its bullish across multiple time scales,
it is more likely to go in your favor. Taking smaller positions with less certainty is also to help reduce risk (this is similar to just continuously averaging and index fund)

#### Problems/Challenges
The most obvious problem with this is the returns. They arent great. However, this is to be expected given the randomness of the data. 

However, there are more technical problems that arise. 
* Fills.
    * You are not guaranteed to be filled and there is slippage.
        * Slippage is a problem with ML because of speed.   
* Volume
    * Volume is limited. How scalable is it
        * Forex has greater volume as most trades are from big banks
    * How much until you start effecting the market (flash crashes from HFT)
* Returns/Risk
    * Are there Stops in place so the model doesnt go AWAL. 
    * Do the additional returns justify the risk (Risk/Reward, MAE, MFE, etc)
* Computational Expensive
    * Requires a lot of resources to train

#### What I have learned
Machine learning, in the way many seem to view it, is not that ideal. For technical trading, it is far to complex to actually yield high returns with proper risk management. 
Furthermore, it does not yield a significantly higher return than typical algorithmic trading. Despite the modern
computing capabilities, It seems that the biggest thing in trading is proper risk management and an 'edge'. ML learning models 
may achieve this 'edge', however, you could end up testing and trying model after model and achieve nothing in a life span. 

However, I do believe that ML remains useful in text analytics. When it comes to technical trading, its very numeric and 
algorithms can be designed to have an "edge", coupled with a trading strategy with proper risk management. However, when 
text is not quite the same. Furthermore, text analysis is useful for longer term trades (speculative/fundamental investing). 
In addition, a machine can read 1000s of articles before a human could, allowing it to read new earning reports, legal documents,
research papers, and much more before humans could properly interpret it.    

Ultimately, I found that ML models are not great for technical trading for the following reasons: 
* Computationally expensive
* Slow (Slippage)
* Doesnt really have an 'edge'

If I were to use ML for stock market prediction, it would be used either for text analysis or to optimize an algorithm, 
that is supervised ML. 

#### Some Additional Notes
This was written last year or so, I never saved all the failed attempts. But there were MANNNNNNNYYYYY failed attempts, including this one

It baffles me how many people attempt to develop ML models to predict the stock market with very minimal knowledge of how 
it works in the first place. 

Suggestions for those who want to try: 
* Learn how the market works
    * Paper Trade (You dont have to be good, just understand)
    * Learn risk management
* Develop an Algorithm with at least Neutral returns (Surprisingly, this is better than 90% traders)
    * Ideally positive returns
* Dont get caught up in absurd returns. If your model says it yield 1000%/yr returns, your model is bad... Sorry
    * High end may be 2%/day (500%/yr), beyond that you have to be taking absurd amounts of risk.
        * Understand this is volume limited. Returns like this may work with a few million at best.
* Look for places that offer high volume, liquidity is important. 
* Use Java/C/GoLang/etc rather than Python
    * Despite Python's ease, implement scripts should be done in Java or a lower level language
        * Speed is important
        * Reliability is important
        * Resource management is important
    * Pythons great for quick testing / trial and error.   


#### This is just my 2cents. Just like with anyone else, take my advice with a grain of salt and figure these things out yourself.

# Best of luck, and make sure to have fun!!! 
