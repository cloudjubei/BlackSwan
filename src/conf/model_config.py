from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelRLConfigSearch:
    model_name: List[str] = field(default_factory=[]), # possible ["ppo", "a2c", "dqn"]
    reward_model: List[str] = field(default_factory=[]) # possible ["combo_all", "combo_actions", "combo_old"],
    learning_rate: List[float] = field(default_factory=[]) 
    batch_size: List[int] = field(default_factory=[]) 
    buffer_size: List[int] = field(default_factory=[]) 
    gamma: List[float] = field(default_factory=[]) 
    tau: List[float] = field(default_factory=[])
    exploration_fraction: List[float] = field(default_factory=[]) 
    exploration_final_eps: List[float] = field(default_factory=[]) 
    learning_starts: List[int] = field(default_factory=[]) 
    train_freq: List[int] = field(default_factory=[]) 
    gradient_steps: List[int] = field(default_factory=[]) 
    target_update_interval: List[int] = field(default_factory=[]) 
    max_grad_norm: List[float] = field(default_factory=[]) 
    optimizer_class: List[str] = field(default_factory=[])
    optimizer_eps: List[float] = field(default_factory=[]) 
    optimizer_weight_decay: List[float] = field(default_factory=[]) 
    optimizer_centered: List[bool] = field(default_factory=[])
    optimizer_alpha: List[float] = field(default_factory=[]) 
    optimizer_momentum: List[float] = field(default_factory=[]) 
    activation_fn: List[str] = field(default_factory=[])
    net_arch: List[List[int]] = field(default_factory=[])
    custom_net_arch: List[List[str]] = field(default_factory=[])

    episodes: List[int] = field(default_factory=[])

    reward_multiplier_combo_noaction: List[float] = field(default_factory=[])
    reward_multiplier_combo_positionprofitpercentage: List[float] = field(default_factory=[])
    reward_multiplier_combo_buy: List[float] = field(default_factory=[])
    reward_multiplier_combo_sell: List[float] = field(default_factory=[])

    reward_multiplier_combo_sell_profit: List[float] = field(default_factory=[])
    reward_multiplier_combo_sell_profit_prev: List[float] = field(default_factory=[])
    reward_multiplier_combo_sell_perfect: List[float] = field(default_factory=[])
    reward_multiplier_combo_sell_drawdown: List[float] = field(default_factory=[])
    reward_multiplier_combo_buy_profit: List[float] = field(default_factory=[])
    reward_multiplier_combo_buy_perfect: List[float] = field(default_factory=[])
    reward_multiplier_combo_buy_profitable_offset: List[int] = field(default_factory=[])
    reward_multiplier_combo_buy_profitable: List[float] = field(default_factory=[])
    reward_multiplier_combo_buy_drawdown: List[float] = field(default_factory=[])
    reward_multiplier_combo_hold_profit: List[float] = field(default_factory=[])
    reward_multiplier_combo_hold_drawdown: List[float] = field(default_factory=[])

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None
@dataclass
class ModelRLConfig:
    model_name: str
    reward_model: str
    learning_rate: float = 0.0001
    batch_size: int = 32
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 1.0
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    learning_starts: int = 50000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10000
    max_grad_norm: float = 10
    optimizer_class: str = 'Adam'
    optimizer_eps: float = 0.00000001
    optimizer_weight_decay: float = 0
    optimizer_centered: bool = False
    optimizer_alpha: float = 0.99
    optimizer_momentum: float = 0
    activation_fn: str = 'ReLU'
    net_arch: List[int] = field(default_factory=[])
    custom_net_arch: List[str] = field(default_factory=[])

    episodes: int = 1

    reward_multiplier_combo_noaction: float = 0
    reward_multiplier_combo_positionprofitpercentage: float = 0
    reward_multiplier_combo_buy: float = 0
    reward_multiplier_combo_sell: float = 0
    
    reward_multiplier_combo_sell_profit: float = 0,
    reward_multiplier_combo_sell_profit_prev: float = 0,
    reward_multiplier_combo_sell_perfect: float = 0,
    reward_multiplier_combo_sell_drawdown: float = 0,
    reward_multiplier_combo_buy_profit: float = 0,
    reward_multiplier_combo_buy_perfect: float = 0,
    reward_multiplier_combo_buy_profitable_offset: int = 0,
    reward_multiplier_combo_buy_profitable: float = 0,
    reward_multiplier_combo_buy_drawdown: float = 0,
    reward_multiplier_combo_hold_profit: float = 0,
    reward_multiplier_combo_hold_drawdown: float = 0,

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None
@dataclass
class ModelLSTMConfigSearch:
    # loss_fn: List[str] = field(default_factory=[]), # possible ["bce", "mse", "r2", "sqm"]
    learning_rate: List[float] = field(default_factory=[]) 
    weight_decay: List[float] = field(default_factory=[]) 
    episodes: List[int] = field(default_factory=[])

    hidden_size: List[int] = field(default_factory=[])
    layers: List[int] = field(default_factory=[])
    lstm_dropout: List[float] = field(default_factory=[]) 
    extra_dropout: List[float] = field(default_factory=[]) 
    first_activation: List[str] = field(default_factory=[]), # possible ["relu", "lrelu", "sigmoid"]
    last_activation: List[str] = field(default_factory=[]), # possible ["relu", "lrelu", "sigmoid"]
@dataclass
class ModelLSTMConfig:
    loss_fn: str = 'mse'
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    episodes: int = 1

    hidden_size: int = 16
    layers: int = 1
    lstm_dropout: float = 0.5
    extra_dropout: float = 0.5
    first_activation: str = 'lrelu'
    last_activation: str = 'sigmoid'

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None
@dataclass
class ModelMLPConfigSearch:
    # loss_fn: List[str] = field(default_factory=[]), # possible ["bce", "mse", "r2", "sqm"]
    learning_rate: List[float] = field(default_factory=[]) 
    weight_decay: List[float] = field(default_factory=[]) 
    episodes: List[int] = field(default_factory=[])

    hidden_dim1: List[int] = field(default_factory=[])
    hidden_dim2: List[int] = field(default_factory=[])
    first_activation: List[str] = field(default_factory=[]), # possible ["relu", "lrelu", "sigmoid"]
    last_activation: List[str] = field(default_factory=[]), # possible ["relu", "lrelu", "sigmoid"]
@dataclass
class ModelMLPConfig:
    loss_fn: str = 'mse'
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    episodes: int = 1

    hidden_dim1: int = 64
    hidden_dim2: int = 32
    first_activation: str = 'relu'
    last_activation: str = 'relu'

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None
@dataclass
class ModelTimeConfigSearch:
    time_buy: List[int] = field(default_factory=[1200])
    time_sell: List[int] = field(default_factory=[1400])
@dataclass
class ModelTimeConfig:
    time_buy: int
    time_sell: int
@dataclass
class ModelTechnicalConfigSearch:
    buy_indicator: List[str] = field(default_factory=[])
    buy_amount_threshold: List[float] = field(default_factory=[])
    sell_indicator: List[str] = field(default_factory=[])
    sell_amount_threshold: List[float] = field(default_factory=[])
    buy_amount_is_multiplier: bool = False
    buy_is_price_check: bool = False
    buy_is_down_check: bool = True
    sell_amount_is_multiplier: bool = False
    sell_is_price_check: bool = False
    sell_is_up_check: bool = True
@dataclass
class ModelTechnicalConfig:
    buy_indicator: str
    buy_amount_threshold: float
    sell_indicator: str
    sell_amount_threshold: float
    buy_amount_is_multiplier: bool = False
    buy_is_price_check: bool = False
    buy_is_down_check: bool = True
    sell_amount_is_multiplier: bool = False
    sell_is_price_check: bool = False
    sell_is_up_check: bool = True
@dataclass
class ModelConfigSearch:
    model_type: str # possible ["hodl", "rl", "technical", "time"]
    model_rl: ModelRLConfigSearch | None = None
    model_lstm: ModelLSTMConfigSearch | None = None
    model_mlp: ModelMLPConfigSearch | None = None
    model_technical: ModelTechnicalConfigSearch | None = None
    model_time: ModelTimeConfigSearch | None = None
@dataclass
class ModelConfig:
    model_type: str # possible ["hodl", "rl", "technical", "time"]
    iterations_to_pick_best: int = 10
    model_rl: ModelRLConfig | None = None
    model_lstm: ModelLSTMConfig | None = None
    model_mlp: ModelMLPConfig | None = None
    model_technical: ModelTechnicalConfig | None = None
    model_time: ModelTimeConfig | None = None
    
    def is_deep(self) -> bool:
        return self.model_type == "rl"
    def is_hodl(self) -> bool:
        return self.model_type == "hodl"
    
# duel_dqn seems to get much better rewards than dqn, yet much lower profits on combo_actions_diff
# drawdown + profit percentage seems to be an ok indicator -> the model definitely learns something
# profit_percentage2 + profit_percentage3 + profit_percentage4 -> are all ok indicators, the model learns from them
# buy_sell_signal + buy_sell_signal2 -> are ok

# indicators1 - ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi5", "rsi10", "rsi15"]
# indicators2 - ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30"]
# indicators3 - ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "bollinger15Low"]
# indicators4 - ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30"]
# indicators5 - ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30"]
# indicators6 - ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "cci10"]
# indicators7 - ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "disparityIndex7", "disparityIndex10"]
# indicators8 - ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "sortinoRatio30"]

# indicators5 + indicators3 -> the best?

# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi5", "rsi15", "choppiness30"] -> extra rsi5
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi5", "rsi10", "rsi15", "choppiness30"] -> extra rsi15
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi5", "rsi10", "rsi15", "bollinger15Low"] -> extra bollinger15Low
# BAD -> ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "rsi15", "choppiness30"] -> extra rsi15
# BAD -> ["kallman15", "kallman30", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30"] -> extra kallman30
# BAD -> ["timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10"] -> no choppiness30
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30"] -> extra rsi10
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30", "bollinger15Low"] -> extra bollinger15Low
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30", "donchianChannels5Low"] -> extra donchianChannels5Low
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30", "donchianChannels10High"] -> extra donchianChannels10High
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "rsi10", "choppiness30", "bollinger10Mid"] -> extra bollinger10Mid
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "bollinger10Mid"] -> extra bollinger10Mid
# BAD -> ["kallman15", "timeseriesMomentum7", "closenessTo1000", "closenessTo10000", "meanReversion10", "meanReversion15", "choppiness30", "bollinger10Mid", "bollinger15Low"] -> extra bollinger15Low
# BAD -> ad5-ad30
# BAD -> volatility5-30
# ALL OF cci improve results, so test later with them
# turbulenceIndex10 improves
# disparityIndex10,7 improves
# volatilityVolume30+5,7 improves
# Williams ALL improve
# StochasticOscillator - only 30 doesn't improve
# SortinoRatio - ALL improve
# SharpeRatio - ALL improve


# BatchNorm1d single with single dropout improves things a lot
# Dropout0.5 seems too strong, but had single great results -> experiment more later
# DropConnectLinear got some great results, placing it at the beginning can lead to no learning, but also great results, at the beginning it seems to improve things -> experiment more
# LayerNorm seems to improve slightly, just 1 is good enough (at the beginning rather than end)
# GRULocal improves
# ScaledDotProductAttention improves
# AdditiveAttention improves
# BatchNorm1d + -> start
# ResidualBlock + -> near the start
# LayerNorm -> improves $, but not sure
# BAD -> MultiHeadAttention
# BAD -> SelfAttention
# BAD -> weight_norm
# BAD -> spectral_norm
# BAD -> GRUFull
# BAD -> LSTFull
# BAD -> LSTFullN
# BAD -> LSTMLocalN
# BAD -> GRULocal2+4
# BAD -> Dense

# MIN lr1 x reward_model 4/6
# MIN lr2 x reward_model 4/6
# MIN custom_net_arch9 2/8
# MIN model_name 7/10
# custom_net_arch2 3/7
# custom_net_arch3 2/7

# custom_net_arch4 1/7 #READY

# TODO: min combo profitable
# TODO: custom_net_arch + indicators1-8
# TODO: custom_net_arch more

# TODO:
    # @ApiProperty() trades_number: string = "0"
    # @ApiProperty() asset_volume_quote: string = "0"
    # @ApiProperty() asset_volume_taker_base: string = "0"
    # @ApiProperty() asset_volume_taker_quote: string = "0"

# and then min tests

# TODO: optimizer optim

model_rl_h = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        # model_name= ["ppo", "reppo", "trpo", "a2c", "ars", "ars-mlp", "qrdqn", "dqn", "duel-dqn"], 
        model_name= ["duel-dqn-custom"],
        # model_name= ["munchausen-dqn-custom"], # TODO: check custom
        # model_name= ["munchausen-dqn"],
        # model_name= ["rainbow-dqn", "iqn"], # very slow (12it/s, 10it/s), but can work very well
        # model_name= ["duel-dqn-lstm"], # can have some very good results with hidden size 2,4

        # reward_model= ["combo_all", "profit_percentage3", "profit_percentage4", "profit_all", "profit_all2", "combo"],
        reward_model= ["combo_all"],

        learning_rate= [0.0001],
        # batch_size= [512, 8, 1024],
        batch_size= [512],
        # buffer_size = [10, 100_000, 1_000_000],
        buffer_size = [1_000_000],
        # gamma = [0.99, 0.95, 0.85],
        gamma = [0.99],
        # tau = [0.99, 0.9, 0.7, 0],
        tau = [0.99],
        # exploration_fraction = [0.1, 0.05, 0.99],
        exploration_fraction = [0.1],
        # exploration_final_eps = [0.01, 0.2, 0.25],
        exploration_final_eps = [0.2],
        # learning_starts= [15_000, 20_000, 25_000],
        learning_starts= [20_000],
        # train_freq= [16, 64, 2],
        train_freq= [16],
        # gradient_steps= [-1, 2],
        gradient_steps= [-1],
        # target_update_interval= [15_000, 20_000, 10_000],
        target_update_interval= [15_000],
        # max_grad_norm= [0.01, 10000],
        max_grad_norm= [0.01],

        # optimizer_class = ['AdamW', 'NAdam', 'RMSprop'], # TODO check with finetuning 
        optimizer_class = ['RMSprop'],
        # optimizer_class = ['NAdam'],
        # optimizer_class = ['AdamW'],
        # optimizer_class = ['default'],
        # TODO: 'DGWO' + 'LBFGS' -> both require closure passed to them
        # optimizer_eps = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2],
        optimizer_eps = [0.5],

        # optimizer_weight_decay = [0, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        optimizer_weight_decay = [0.00000001],
        # optimizer_centered = [True, False],
        optimizer_centered = [True],
        # optimizer_alpha = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.99],
        optimizer_alpha = [0.9],
        # optimizer_momentum = [0.0001, 0.1, 0.000001],
        optimizer_momentum = [0.0001], 

        # activation_fn = ['LogSigmoid', 'CELU', 'PReLU', 'ReLU6', 'Softsign'],
        activation_fn= ['CELU'],
        # net_arch= [[512,256,64,32], [256,256,256,256,256,256], [1024,1024], [2048,512,64], [256,256,256,256], [256,256,256,256,256], [1024,256,64,16], [512,256,128,64,32]],
        net_arch= [[512,256,64,32]],
        
        # custom_net_arch= [[""]],
        
        # custom_net_arch= [
        #     ["Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"], # ALWAYS HAS TO BE 1 more Linear+act than custom_net_arch number
        # ],

        # custom_net_arch1= [
        #     ["Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "Dropout", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "activation_fn", "Dropout", "Linear"],
        
            # latest OK one with good Win ratio
        #     ["BatchNorm1d", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        
        #     ["Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        
        # less OK but still good
        #     ["BatchNorm1d", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "Dropout05", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "Dropout05", "activation_fn", "Linear"],
        #     ["Linear", "LayerNorm", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        # ],
        # custom_net_arch2= [
        #     ["BatchNorm1d", "Linear", "LayerNorm", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "LayerNorm", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "BatchNorm1d", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "BatchNorm1d", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "ScaledDotProductAttention", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "BatchNorm1d", "Linear", "LayerNorm", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"]
        # ],
        # custom_net_arch3= [
        #     ["BatchNorm1d", "Linear", "activation_fn", "Dropout", "Linear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "ResidualBlock", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "ResidualBlock", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "ResidualBlock", "Linear"],
        # ],
        custom_net_arch= [ #TODO
            ["DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
            ["BatchNorm1d", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
            ["Linear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "Linear"],
            ["BatchNorm1d", "Linear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "Linear"],
            ["DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "DropConnectLinear", "activation_fn", "Linear"],
            ["BatchNorm1d", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "DropConnectLinear", "activation_fn", "Linear"],
        ],
        

        # medium= [
        #     ["ResidualBlock", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        # ],
        # bad= [
        #     ["LayerNorm", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "BatchNorm1d", "ScaledDotProductAttention", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "ScaledDotProductAttention", "BatchNorm1d", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "Dropout1", "ScaledDotProductAttention", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "ScaledDotProductAttention", "Dropout1", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "BatchNorm1d", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "BatchNorm1d", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "activation_fn", "BatchNorm1d", "Linear"]
        #     ["Linear", "activation_fn", "LayerNorm", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "LayerNorm", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "activation_fn", "LayerNorm", "Linear"],
        #     ["Linear", "activation_fn", "ResidualBlock", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "activation_fn", "ResidualBlock", "Linear"]
        #     ["Linear", "ScaledDotProductAttention", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "ScaledDotProductAttention", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "ScaledDotProductAttention", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "activation_fn", "ScaledDotProductAttention", "Linear"],
        #     ["MultiHeadAttention", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["AdditiveAttention", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"], # TAKES WAY TOO LONG
        #     ["SelfAttention", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"], # TAKES WAY TOO LONG
        #     ["GRULocal", "activation_fn", "Linear", "activation_fn", "Linear"], # TAKES WAY TOO LONG
        #     ["LSTMLocalN", "activation_fn", "Linear", "activation_fn", "Linear"], # TAKES WAY TOO LONG
        #     ["DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "activation_fn", "Linear", "activation_fn", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear", "activation_fn", "DropConnectLinear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Linear", "Dropout", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "Dropout", "ResidualBlock", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "ResidualBlock", "Dropout", "activation_fn", "Linear"],
        #     ["ResidualBlock", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "BatchNorm1d", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "DropConnectLinear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "activation_fn", "ResidualBlock", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Dropout", "Linear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Dropout", "Linear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "DropConnectLinear", "activation_fn", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "DropConnectLinear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "Dropout", "activation_fn", "DropConnectLinear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["Linear", "BatchNorm1d", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["Linear", "Dropout", "BatchNorm1d", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["Linear", "Dropout1", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "Dropout2", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Dropout05", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Dropout1", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Dropout2", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Dropout", "Linear", "ResidualBlock", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "ScaledDotProductAttention", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "ScaledDotProductAttention", "activation_fn", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["DropConnectLinear", "ScaledDotProductAttention", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "BatchNorm1d", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "Dropout", "BatchNorm1d", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "Dropout05", "activation_fn", "Linear", "Dropout05", "activation_fn", "Linear", "Dropout05", "activation_fn", "Linear", "Dropout05", "activation_fn", "Linear"],
        #     ["LayerNorm", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["LayerNorm", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["LayerNorm", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "LayerNorm", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "Dropout", "LayerNorm", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "Dropout", "activation_fn", "LayerNorm", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Dropout", "LayerNorm", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["LayerNorm", "BatchNorm1d", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "LayerNorm", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "BatchNorm1d", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["BatchNorm1d", "Linear", "activation_fn", "Dropout05", "Linear", "activation_fn", "Dropout05", "Linear", "activation_fn", "Dropout05", "Linear", "activation_fn", "Dropout05", "Linear"],
        #     ["Linear", "activation_fn", "Dropout05", "Linear", "activation_fn", "Dropout05", "Linear", "activation_fn", "Dropout05", "Linear", "activation_fn", "Dropout05", "Linear"],
        #     ["Linear", "Dropout", "activation_fn", "Linear", "Dropout", "activation_fn", "Linear", "Dropout", "activation_fn", "Linear", "Dropout", "activation_fn", "Linear"],
        #     ["Linear", "Dropout1", "activation_fn", "Linear", "Dropout1", "activation_fn", "Linear", "Dropout1", "activation_fn", "Linear", "Dropout1", "activation_fn", "Linear"],
        #     ["Linear", "Dropout2", "activation_fn", "Linear", "Dropout2", "activation_fn", "Linear", "Dropout2", "activation_fn", "Linear", "Dropout2", "activation_fn", "Linear"],
        # ],

        episodes= [1],

        reward_multiplier_combo_noaction= [-1],
        reward_multiplier_combo_positionprofitpercentage= [10],
        reward_multiplier_combo_buy= [0.1],
        reward_multiplier_combo_sell= [1000],

        # reward_multiplier_combo_sell_profit= [1000, 100, 10], #
        reward_multiplier_combo_sell_profit= [1000],
        # reward_multiplier_combo_sell_profit_prev= [100, 10, 1, 0], #
        reward_multiplier_combo_sell_profit_prev= [100],
        # reward_multiplier_combo_sell_perfect= [0.001, 0, 0.0001], #
        reward_multiplier_combo_sell_perfect= [0.001],
        # reward_multiplier_combo_sell_drawdown= [0, 0.1, 0.001], #
        reward_multiplier_combo_sell_drawdown= [0],
        # reward_multiplier_combo_buy_profit= [1, 0.1, 1000], #
        reward_multiplier_combo_buy_profit= [1],
        # reward_multiplier_combo_buy_perfect= [0.001, 0.1], #
        reward_multiplier_combo_buy_perfect= [0.001],
        
        # reward_multiplier_combo_buy_profitable_offset= [5, 10], #
        reward_multiplier_combo_buy_profitable_offset= [5],
        # reward_multiplier_combo_buy_profitable= [0.1, 0.01, 0.001], #
        reward_multiplier_combo_buy_profitable= [0.1],

        # reward_multiplier_combo_buy_drawdown= [0.001, 0.1, 10], #
        reward_multiplier_combo_buy_drawdown= [0.001],
        # reward_multiplier_combo_hold_profit= [10, 1, 100], #
        reward_multiplier_combo_hold_profit= [10],
        # reward_multiplier_combo_hold_drawdown= [0.0001, 0.01, 0.001, 1], #
        reward_multiplier_combo_hold_drawdown= [0.0001],


        # checkpoints_folder='checkpoints/',
        # checkpoint_to_load='rl_dqn_combo_actions_diff_0.0001_10000_256_1000000_0.9_Adamax_CELU_[64, 64, 64, 64]_1_1717946749.513737'
    )
)

model_rl_min = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        model_name= ["duel-dqn-custom"],
        # model_name= ["ppo", "reppo", "trpo", "a2c", "ars", "ars-mlp", "qrdqn", "dqn", "duel-dqn"], 

        # reward_model= ["combo_all", "profit_percentage3", "profit_percentage4", "profit_all", "profit_all2", "combo"],
        # reward_model= ["combo_all", "profit_percentage3", "profit_percentage4", "profit_all", "profit_all2"],
        reward_model= ["combo_all"],
        # reward_model= ["combo"], #TODO:
 
        # learning_rate= [0.00000001, 0.000001, 0.0001],
        # learning_rate= [0.0001],
        learning_rate= [0.0000001],
        batch_size= [512],
        # buffer_size = [10, 100_000, 1_000_000],
        buffer_size = [1_000_000],
        gamma = [0.99],
        tau = [0.99],
        exploration_fraction = [0.1],
        exploration_final_eps = [0.2],
        # learning_starts= [20_000],
        learning_starts= [50_000],
        train_freq= [16],
        gradient_steps= [-1],
        target_update_interval= [15_000],
        # max_grad_norm= [0.01, 10000],
        max_grad_norm= [0.01],
        optimizer_class = ['RMSprop'],

         optimizer_eps = [0.5],
        optimizer_weight_decay = [0.00000001],
        optimizer_centered = [True],
        optimizer_alpha = [0.9],
        optimizer_momentum = [0.0001], 
        activation_fn= ['CELU'],
        net_arch= [[512,256,64,32]],
        
        custom_net_arch= [
            ["Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear", "activation_fn", "Linear"],
        ],
        # custom_net_arch9= [
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "Dropout", "activation_fn", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Dropout", "Linear", "activation_fn", "Linear"],
        #     ["ScaledDotProductAttention", "Linear", "activation_fn", "Linear", "Dropout", "activation_fn", "Linear"],
        #     ["Linear", "Dropout", "activation_fn", "Linear", "ResidualBlock", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "Dropout", "ResidualBlock", "activation_fn", "Linear"],
        #     ["Linear", "activation_fn", "Linear", "ResidualBlock", "Dropout", "activation_fn", "Linear"],
        # ],

        episodes= [1],

        reward_multiplier_combo_noaction= [-1],
        reward_multiplier_combo_positionprofitpercentage= [10],
        reward_multiplier_combo_buy= [0.1],
        reward_multiplier_combo_sell= [1000],

        reward_multiplier_combo_sell_profit= [1000],
        reward_multiplier_combo_sell_profit_prev= [100],
        reward_multiplier_combo_sell_perfect= [0.001],
        reward_multiplier_combo_sell_drawdown= [0],
        reward_multiplier_combo_buy_profit= [1],
        reward_multiplier_combo_buy_perfect= [0.001],

        # reward_multiplier_combo_buy_profitable_offset= [5, 10, 30, 60, 120, 600], #
        reward_multiplier_combo_buy_profitable_offset= [5],
        reward_multiplier_combo_buy_profitable= [0.1],
        reward_multiplier_combo_buy_drawdown= [0.001],
        reward_multiplier_combo_hold_profit= [10],
        reward_multiplier_combo_hold_drawdown= [0.0001],
    )
)

model_hodl = ModelConfigSearch(
    model_type= "hodl"
)
model_time = ModelConfigSearch(
    model_type= "time",
    model_time= ModelTimeConfigSearch(
        time_buy=[600, 700, 800, 900, 1000, 1100, 1200, 1300],
        time_sell=[1400, 1500, 1600, 1700, 1800, 1900, 2000]
    )
)
model_time2 = ModelConfigSearch(
    model_type= "time",
    model_time= ModelTimeConfigSearch(
        time_buy=[1000,1100,1200,1300,1350],
        time_sell=[1400,1401,1405,1410,1430,1500,1505,1600]
    )
)
model_technical_bollinger= ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["bollinger20Low"],
        buy_amount_threshold= [1.01, 1.05, 1.1, 1.2, 1.3, 1.5],
        buy_amount_is_multiplier= True,
        buy_is_price_check= True,
        sell_indicator= ["bollinger20High"],
        sell_amount_threshold= [0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        sell_amount_is_multiplier= True,
        sell_is_price_check= True,
    )
)
model_technical_kallmanfilter5 = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["kallman5Momentum"],
        buy_amount_threshold= [0.01, 0.1, 0.6, 1.0, 5.0],
        sell_indicator= ["kallman5Momentum"],
        sell_amount_threshold= [-0.01, -0.1, -0.6, -1.0, -5.0],
    )
)
model_technical_kallmanfilter14 = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["kallman14Momentum"],
        buy_amount_threshold= [0.01, 0.1, 0.6, 1.0, 5.0],
        sell_indicator= ["kallman14Momentum"],
        sell_amount_threshold= [-0.01, -0.1, -0.6, -1.0, -5.0],
    )
)
model_technical_kallmanfilter20 = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["kallman20Momentum"],
        buy_amount_threshold= [0.01, 0.1, 0.6, 1.0, 5.0],
        sell_indicator= ["kallman20Momentum"],
        sell_amount_threshold= [-0.01, -0.1, -0.6, -1.0, -5.0],
    )
)

model_time_test = ModelConfigSearch(
    model_type= "time",
    model_time= ModelTimeConfigSearch(
        time_buy=[1200],
        time_sell=[1400]
    )
)
model_technical_kallmanfilter_test = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["kallman20Momentum"],
        buy_amount_threshold= [0.01],
        sell_indicator= ["kallman20Momentum"],
        sell_amount_threshold= [-0.01],
    )
)
model_technical_bollinger_test = ModelConfigSearch(
    model_type= "technical",
    model_technical= ModelTechnicalConfigSearch(
        buy_indicator= ["bollinger20Low"],
        buy_amount_threshold= [1.3],
        buy_amount_is_multiplier= True,
        buy_is_price_check= True,
        sell_indicator= ["bollinger20High"],
        sell_amount_threshold= [0.7],
        sell_amount_is_multiplier= True,
        sell_is_price_check= True,
    )
)

model_rl_comboall_adamax = ModelConfigSearch(
    model_type= "rl",
    model_rl= ModelRLConfigSearch(
        model_name= ["dqn"],
        reward_model= ["combo_all"],
        learning_rate= [0.01],
        batch_size= [128],
        buffer_size = [1_000],
        gamma = [0.995],
        tau = [0],
        exploration_fraction = [0.05],
        exploration_final_eps = [0.1],
        learning_starts= [10_000],
        train_freq= [8],
        gradient_steps= [-1],
        target_update_interval= [10_000],
        max_grad_norm= [1000],
        optimizer_class = ['Adamax'],
        optimizer_eps = [0.001],
        optimizer_weight_decay = [0.00001],
        optimizer_centered = [True],
        optimizer_alpha = [0.9],
        optimizer_momentum = [0.0001], 
        activation_fn= ['CELU'],
        net_arch= [[64,64]],
        custom_net_arch= [[""]],

        episodes= [1],

        reward_multiplier_combo_noaction= [0.1],
        reward_multiplier_combo_positionprofitpercentage= [0.001],
        reward_multiplier_combo_buy= [1],
        reward_multiplier_combo_sell= [100],

        reward_multiplier_combo_sell_profit= [100],
        reward_multiplier_combo_sell_profit_prev= [100],
        reward_multiplier_combo_sell_perfect= [0.01],
        reward_multiplier_combo_sell_drawdown= [0.01],
        reward_multiplier_combo_buy_profit= [100],
        reward_multiplier_combo_buy_perfect= [0.01],
        reward_multiplier_combo_buy_profitable_offset= [5],
        reward_multiplier_combo_buy_profitable= [0.01],
        reward_multiplier_combo_buy_drawdown= [0.01],
        reward_multiplier_combo_hold_profit= [100],
        reward_multiplier_combo_hold_drawdown= [0.01],

        # checkpoints_folder='trials/',
        # checkpoint_to_load='rl_dqn_combo_all_0.01_0.995_Adamax_CELU_[64, 64]_1_1716187086.833314'
        checkpoints_folder='checkpoints/',
        checkpoint_to_load='rl_dqn_combo_all_0.005_1000_4_1000000_0.995_Adamax_CELU_[64, 64, 64, 64]_1_1716768803.222076'
    )
)

@dataclass
class ModelRLConfig:
    model_name: str
    reward_model: str
    learning_rate: float = 0.0001
    batch_size: int = 32
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 1.0
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    learning_starts: int = 50000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10000
    max_grad_norm: float = 10
    optimizer_class: str = 'Adam'
    optimizer_eps: float = 0.00000001
    optimizer_weight_decay: float = 0
    optimizer_centered: bool = False
    optimizer_alpha: float = 0.99
    optimizer_momentum: float = 0
    activation_fn: str = 'ReLU'
    net_arch: List[int] = field(default_factory=[])
    custom_net_arch: List[str] = field(default_factory=[])

    episodes: int = 1

    reward_multiplier_combo_noaction: float = 0
    reward_multiplier_combo_positionprofitpercentage: float = 0
    reward_multiplier_combo_buy: float = 0
    reward_multiplier_combo_sell: float = 0

    reward_multiplier_combo_sell_profit: float = 0,
    reward_multiplier_combo_sell_profit_prev: float = 0,
    reward_multiplier_combo_sell_perfect: float = 0,
    reward_multiplier_combo_sell_drawdown: float = 0,
    reward_multiplier_combo_buy_profit: float = 0,
    reward_multiplier_combo_buy_perfect: float = 0,
    reward_multiplier_combo_buy_profitable_offset: int = 0,
    reward_multiplier_combo_buy_profitable: float = 0,
    reward_multiplier_combo_buy_drawdown: float = 0,
    reward_multiplier_combo_hold_profit: float = 0,
    reward_multiplier_combo_hold_drawdown: float = 0,

    progress_bar: bool = True
    checkpoints_folder: str = 'checkpoints/'
    checkpoint_to_load: str | None = None

def get_models_simple():
    # return [model_hodl, model_time, model_technical_bollinger, model_technical_kallmanfilter5, model_technical_kallmanfilter14, model_technical_kallmanfilter20]
    return [model_hodl, model_time_test, model_technical_kallmanfilter_test, model_technical_bollinger_test]

def get_models_rl_h():
    return [model_hodl, model_rl_h]
def get_models_rl_min():
    return [model_hodl, model_rl_min]

def get_models_all():
    return [model_hodl]