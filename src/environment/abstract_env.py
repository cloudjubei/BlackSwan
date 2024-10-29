from matplotlib.figure import Figure
from abc import abstractmethod
import pandas as pd
import gymnasium
from gymnasium import spaces
from src.conf.env_config import EnvConfig
from src.data.abstract_dataprovider import AbstractDataProvider


class AbstractEnv(gymnasium.Env):
    def __init__(self, env_config: EnvConfig, data_provider: AbstractDataProvider, device: str):
        super().__init__()
        self.env_config = env_config
        self.data_provider = data_provider
        self.device = device

        self.reward_model = "percent_profit"
        self.current_step = 0

        self.id = self.get_id()

    # observations_contain: List[str]
    def get_id(self) -> str:
        return f'{self.env_config.type}_{self.env_config.amount}_{len(self.env_config.observations_contain)}_-{self.env_config.transaction_fee * 100 if self.env_config.transaction_fee is not None else 0}%_TP{self.env_config.take_profit if self.env_config.take_profit is not None else 0}_TR{self.env_config.trailing_take_profit if self.env_config.trailing_take_profit is not None else 0}_SL{self.env_config.stop_loss if self.env_config.stop_loss is not None else 0}'

    def get_timesteps(self) -> int:
        return self.data_provider.get_timesteps()
    
    def setup(self, reward_model: str, multipliers):
        self.reward_model = reward_model
        self.reward_multipliers = multipliers

    def get_price(self, step: int) -> float:
        return self.data_provider.get_price(step)

    @abstractmethod
    def create_action_space(self) -> spaces.Discrete:
        pass

    @abstractmethod
    def create_observation_space(self) -> spaces.Box:
        pass

    @abstractmethod
    def take_action(self, action) -> bool:
        pass

    @abstractmethod
    def calculate_cumulative_return(self) -> float:
        pass

    @abstractmethod
    def calculate_sharpe_ratio(self, risk_free_rate: int) -> float:
        pass

    @abstractmethod
    def plot_portfolio_value(self) -> Figure:
        pass

    @abstractmethod
    def log_metrics_over_time(self, prefix: str, log_remote: bool) -> None:
        pass

    @abstractmethod
    def get_run_state(self):
        pass
    
    @abstractmethod
    def get_raw_df_for_plotting(self) -> pd.DataFrame:
        pass
