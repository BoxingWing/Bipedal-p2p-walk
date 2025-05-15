from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

@configclass
class BipedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 1
    num_steps_per_env = 40  # 16 The number of steps per environment per update.
    max_iterations = 8000
    save_interval = 100
    experiment_name = "loong_walk_p2p_s1"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",  # 'scalar' or 'log'
        actor_hidden_dims=[512, 256, 128], #[512,256,128]
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class BipedPPORunnerCfgS2(RslRlOnPolicyRunnerCfg):
    seed = 1
    num_steps_per_env = 40  # 16 The number of steps per environment per update.
    max_iterations = 8000
    save_interval = 200
    experiment_name = "loong_walk_p2p_s2"
    resume = True
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",  # 'scalar' or 'log'
        actor_hidden_dims=[512, 256, 128], #[512,256,128]
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
