import logging

from src.data.abstract_dataprovider import AbstractDataProvider

import hydra
from src.conf.structured_config import Config
from src.conf.data_config import DataConfig
from src.conf.model_config import ModelConfig
from src.data.data_factory import create_provider
from src.data.data_utils import fill_missing_timestamps, plot_actions_data, plot_actions_data_non_trade
from src.environment.env_factory import create_environment
from src.model.model_factory import create_model, get_model_combinations
from src.environment.abstract_env import AbstractEnv
from src.model.abstract_model import AbstractModel
import pandas as pd
import numpy as np
import time

def run_training(config: Config, model: AbstractModel, env: AbstractEnv, data_provider: AbstractDataProvider):
    print(f"Model {model.id} env: {env.id} data: {data_provider.id} is training...")

    # TODO:
    # capture time
    env.setup(model.get_reward_model(), model.get_reward_multipliers())
    # env.reset()

    model.train(env)    

    if config.show_render and model.show_train_render():
        env.render()

    # env.log_metrics_over_time(f'train', not config.local_only)

    print(f'Model finished learning with total reward: {env.total_reward}')
    return env.get_run_state()

def run_testing(config: Config, model: AbstractModel, env: AbstractEnv, deterministic: bool = True):
    print("Testing the model...")
    
    # TODO:
    # capture time
    env.setup(model.get_reward_model(), model.get_reward_multipliers())
    model.test(env, deterministic)

    if config.show_render:
        env.render()
    # env.log_metrics_over_time('test', not config.local_only)

    print("Model finished testing.")
    return env.get_run_state()

def run_model(config: Config, model_config: ModelConfig, data_provider: AbstractDataProvider, env_train: AbstractEnv, env_test: AbstractEnv, states: list):
    
    print("Creating model...")
    model = create_model(model_config, env_train, config.device)
    if not model.is_pretrained():
        run_training(config, model, env_train, data_provider)

    if model.has_deterministic_test():
        state_test = run_testing(config, model, env_test, True)
        states.append([data_provider.id, env_test.id, model.id] + state_test)
        
def get_model_configs_count(config: Config) -> int:
    count = 0
    for model_config_search in config.model_configs:
        model_config_combinations = get_model_combinations(model_config_search)
        count += len(model_config_combinations)
    return count

# MAC                                                   -> 170 it/s + 245 it/s SBX + 24 it/s rainbow + 12 it/s iqn
# AMD Ryzen 5 5600X 6-core + RTX 3060                   -> 385 it/s
# 11th Gen Core™ i5-11400F + RTX 3080 Ti                -> 480 it/s + 358 it/s SBX + 90 it/s rainbow + 207 it/s iqn
# AMD Ryzen Threadripper PRO 5995WX 16-core + RTX 4090  -> 440 it/s + 235 it/s SBX + 74 it/s rainbow + 224 it/s iqn
# 490 it/s vs 385 it/s vs 250 it/s MAC

@hydra.main(version_base=None, config_name="config_simple")
# @hydra.main(version_base=None, config_name="config_simple_other")
def main(config: Config) -> None:

    pd.options.display.float_format = '{:.3f}'.format
    np.set_printoptions(formatter={'float_kind':'{:.4f}'.format})

    result_states = list()

    run_count_total = len(config.data_configs) * len(config.env_configs) * get_model_configs_count(config)
    run_count = 0

    for data_config in config.data_configs:
        print("Preparing train data...")
        data_provider_train = create_provider(data_config, data_config.train_data_paths, data_config.fidelity_input, data_config.fidelity_run, data_config.layers, data_config.buyreward_maxwait, data_config.buyreward_percent)

        print("Preparing test data...")
        data_provider_test = create_provider(data_config, data_config.test_data_paths, data_config.fidelity_input_test, data_config.fidelity_run_test, data_config.layers_test, data_config.buyreward_maxwait_test, data_config.buyreward_percent_test)

        for env_config in config.env_configs:
            print("Creating train environment...")
            env_train = create_environment(env_config, data_provider_train, config.device)

            print("Creating testing environment...")
            env_test = create_environment(env_config, data_provider_test, config.device)

            print(f'Run 0/{run_count_total} Started')

            for model_config_search in config.model_configs:
                model_config_combinations = get_model_combinations(model_config_search)
                for model_config in model_config_combinations:
                    run_count += 1
                    if (not model_config.is_deep() and not model_config.is_hodl()):
                        print(f'Run {run_count}/{run_count_total} Skipped')
                        continue

                    max_iterations = model_config.iterations_to_pick_best if model_config.is_deep() else 1
                    for i in range(0, max_iterations):
                        print(f'Iteration {i+1}/{max_iterations}')
                        run_model(config, model_config, data_provider_train, env_train, env_test, result_states)
                        # if (not model_config.is_hodl()):
                        #     plot_actions_data_non_trade(env_test)
                            # plot_actions_data(env_test)
                    
                    print(f'Run {run_count}/{run_count_total} Complete')

    # df_results = pd.DataFrame(result_states, columns=["Data", "Env", "Name", "Rewards", "$", "%", "Trade$", "CompoundTrade$", "Wins", "Losses", "Win%", "Avg$Win", "Max$Win", "Min$Win", "Avg$Loss", "Max$Loss", "Min$Loss", "Avg$Trade", "Fees$", "Volume$", "#trades", "SLs", "SL$", "Multipliers"])
    df_results = pd.DataFrame(result_states, columns=["Data", "Env", "Name", "Rewards", "F1", "Ratio", "Acc%", "Prec%", "Rec%", "-Rec%", "AvgStreak", "MaxStreak", "Totals", "Multipliers"])
    results_name = f'results_{time.time()}.csv'
    df_results.to_csv(results_name, index=False)  

    # df_results = df_results.drop(columns=["Data", "Env", "%", "CompoundTrade$", "Avg$Win", "Max$Win", "Min$Win", "Avg$Loss", "Max$Loss", "Min$Loss", "Avg$Trade", "Fees$", "Volume$", "SLs", "SL$", "Multipliers"])
    df_results = df_results.drop(columns=["Data", "Env", "Acc%", "Prec%", "Rec%", "-Rec%", "AvgStreak", "MaxStreak", "Multipliers"])
    pd.set_option('display.max_rows', None)  # None means unlimited rows
    # pd.set_option('display.max_columns', None)  # None means unlimited columns
    print(df_results)
    print(f'results_name: {results_name}')

    return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error encountered: {e}")
