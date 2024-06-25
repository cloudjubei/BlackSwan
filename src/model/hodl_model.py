from src.model.abstract_model import AbstractModel
from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv

class HodlModel(AbstractModel):

    def get_id(self, config: ModelConfig):
        return f'{self.config.model_type}'
    
    def show_train_render(self):
        return True
    
    def is_pretrained(self):
        return True
    
    def train(self, env: AbstractEnv):
        env.reset()

        take_profit = env.env_config.take_profit
        stop_loss = env.env_config.stop_loss
        env.env_config.take_profit = None
        env.env_config.stop_loss = None

        env.step(1)
        for _ in range(env.get_timesteps() - 2):
            env.step(0)
        env.step(2)

        env.env_config.take_profit = take_profit
        env.env_config.stop_loss = stop_loss

    def test(self, env: AbstractEnv, deterministic: bool = True):
        return self.train(env)

    def get_action(self, env: AbstractEnv, obs):
        return None

