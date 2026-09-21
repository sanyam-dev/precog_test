from pdb import run
import numpy as np
from utils import Utils
import pandas as pd

class Price:
    def __init__(self, open, high, low, close, volume):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __repr__(self):
        return f"Price(open={self.open}, high={self.high}, low={self.low}, close={self.close}, volume={self.volume})"

class Asset:
    def __init__(self, price):
        self.price = price
        self.weight = 0

    def update_weight(self, weight):
        self.weight = weight

    def __repr__(self):
        return f"Asset(price={self.price.__repr__()})"

class State:
    def __init__(self, date, capital, assets):
        self.date = date
        self.capital = capital
        self.assets = assets
        self.done = False

    def update_capital(self, capital):
        self.capital = capital
    
    def update_assets(self, assets):
        self.assets = assets
    
    def update_date(self, date):
        self.date = date
    
    def get_capital(self):
        return self.capital
    
    def get_assets(self):
        return self.assets
    
    def get_date(self):
        return self.date
    
    def set_done(self, done):
        self.done = done
    
    def get_done(self):
        return self.done

    
    def __repr__(self):
        return f"State(date={self.date}, capital={self.capital}, assets={self.assets})"

class Simluation:
    '''
    Simulation class for the trading environment
    '''
    def __init__(self, DF, initial_capital=100000, trasaction_cost=0.001):
        self.DF = DF
        self.tr_ = None
        self.te_ = None
        self.transaction_cost = 0.001
        self.initial_capital = initial_capital
        # State at every time step captures capital left with the agent, the current day OLHCV of every stock 
        self.state = State(date=self.DF.index[0], capital=self.initial_capital, assets=[Asset(Price(open = 0, close = 0, high = 0, low = 0, volume = 0)) for _ in range(100)]) 
        self.time_step = 0
        self.done = False
        self.transaction_cost = trasaction_cost

    def reset(self):
        self.DF = self.DF.copy()
        self.tr_ = self.tr_.copy() if self.tr_ is not None else None
        self.te_ = self.te_.copy() if self.te_ is not None else None
        self.done = False
        self.info = None
        self.state = State(date=self.DF.index[0], capital=self.initial_capital, assets=[Asset(Price(open = 0, close = 0, high = 0, low = 0, volume = 0)) for _ in range(100)]) 
        self.time_step = 0
        return self.state, 0, False, {}
    
    def step(self, action):
        current_step = self.time_step
        self.time_step += 1

        #check if simulation is done
        if self.time_step >= len(self.DF):
            self.state.set_done(True)
        
        if self.state.get_done():
            return self.state, self.get_reward(), self.state.get_done(), {}
        
        
        # calculate transaction cost
        weights = np.asarray(action, dtype=float)
        previous_weight = np.array([asset.weight for asset in self.state.get_assets()], dtype=float)
        ttc_cost = self.transaction_cost * np.sum(np.abs(weights - previous_weight))
        return_ = self.get_return(weights)
        net_return = return_ - ttc_cost

        # update state
        current_capital = self.state.get_capital() * (1 + net_return)
        self.state.update_capital(current_capital)
        assets = self.state.get_assets()
        asset_names = self.DF.columns.get_level_values(0).unique()
        row = self.DF.iloc[self.time_step]

        for asset, asset_name, weight in zip(assets, asset_names, weights):
            values = row[asset_name]
            asset.price = Price(
                open=values["Open"],
                high=values["High"],
                low=values["Low"],
                close=values["Close"],
                volume=values["Volume"],
            )
            asset.update_weight(float(weight))

        self.state.update_assets(assets)
        self.state.update_date(self.DF.index[self.time_step])

        # calculate reward
        reward = self.get_reward()
        print("State: ", self.state)
        print("Reward: ", reward)
        print("Done: ", self.state.get_done())
        print("Info: ", {})
        print("--------------------------------")

        return self.state, reward, self.state.get_done(), {}

    def get_reward(self):
        
        return 0

    def get_return(self, weights):
        close = self.DF.xs("Close", axis=1, level=1)
        current_close = close.iloc[self.time_step].to_numpy(dtype=float)
        previous_close = close.iloc[self.time_step - 1].to_numpy(dtype=float)

        asset_returns = current_close / previous_close - 1
        return float(weights @ asset_returns)

def load_test_data():
    df = pd.read_csv(
        './test.csv',
        header=[0, 1],
        index_col=0,
        parse_dates=[0],
        float_precision="round_trip",
    )
    df.index.name = "Date"
    return df

def run_simulation(sim):
    state, _, _, _ = sim.reset()
    while not state.get_done():
        action = np.random.randn(100)
        state, reward, done, info = sim.step(action)