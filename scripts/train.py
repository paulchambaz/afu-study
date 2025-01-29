from afu.agents.sac import SACAgent
from afu.environments.mountaincar import MountainCarEnv
from omegaconf import OmegaConf


def run_sac(sac):
    cfg = sac.cfg
    logger = sac.logger

    # Get initial entropy coefficient
    ent_coef = cfg.algorithm.init_entropy_coef
    tau = cfg.algorithm.tau_target

    # Setup agents and critics
    t_actor = TemporalAgent(sac.train_policy)
    t_q_agents = TemporalAgent(Agents(sac.critic_1, sac.critic_2))
    t_target_q_agents = TemporalAgent(
        Agents(sac.target_critic_1, sac.target_critic_2)
    )

    # Setup optimizers
    actor_optimizer = setup_optimizer(cfg.actor_optimizer, sac.actor)
    critic_optimizer = setup_optimizer(
        cfg.critic_optimizer, sac.critic_1, sac.critic_2
    )
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)

    # Training loop
    for rb in sac.iter_replay_buffers():
        # Critic update
        critic_optimizer.zero_grad()
        critic_loss_1, critic_loss_2 = compute_critic_loss(
            cfg,
            rb.get("reward"),
            rb.get("must_bootstrap"),
            t_actor,
            t_q_agents,
            t_target_q_agents,
            rb,
            ent_coef,
        )
        (critic_loss_1 + critic_loss_2).backward()
        critic_optimizer.step()

        # Actor update
        actor_optimizer.zero_grad()
        actor_loss = compute_actor_loss(ent_coef, t_actor, t_q_agents, rb)
        actor_loss.backward()
        actor_optimizer.step()

        # Update entropy coefficient if using auto mode
        if entropy_coef_optimizer:
            entropy_coef_optimizer.zero_grad()
            action_logprobs = rb["action_logprobs"][0].detach()
            entropy_coef_loss = -(
                log_entropy_coef.exp() * (action_logprobs + sac.target_entropy)
            ).mean()
            entropy_coef_loss.backward()
            entropy_coef_optimizer.step()
            ent_coef = log_entropy_coef.exp().item()

        # Soft update target networks
        soft_update(sac.critic_1, sac.target_critic_1, tau)
        soft_update(sac.critic_2, sac.target_critic_2, tau)

        # Log metrics
        logger.add_log(
            "critic_loss", (critic_loss_1 + critic_loss_2).item(), sac.nb_steps
        )
        logger.add_log("actor_loss", actor_loss.item(), sac.nb_steps)
        logger.add_log("entropy_coef", ent_coef, sac.nb_steps)

        # Simple evaluation every 1000 steps
        if sac.nb_steps % 1000 == 0:
            eval_reward = evaluate_policy(sac.eval_policy, sac.eval_env)
            logger.add_log("eval_reward", eval_reward, sac.nb_steps)
            print(f"Step {sac.nb_steps}: Eval reward = {eval_reward}")

    return sac


def evaluate_policy(policy, env, episodes=5):
    """Quick and dirty evaluation - just average reward over a few episodes"""
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy(
                obs, stochastic=False
            )  # Deterministic actions for evaluation
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
    return total_reward / episodes


def main():
    # Create environment
    env = MountainCarEnv(
        render_mode=None
    )  # None for training, "human" for visualization

    # Set basic SAC parameters
    params = {
        "algorithm": {
            "seed": 42,
            "n_envs": 1,
            "n_steps": 100,
            "batch_size": 256,
            "learning_starts": 1000,
            "discount_factor": 0.99,
            "entropy_mode": "auto",
            "init_entropy_coef": 0.1,
            "tau_target": 0.005,
            "architecture": {
                "actor_hidden_size": [64, 64],
                "critic_hidden_size": [64, 64],
            },
        },
        "actor_optimizer": {
            "classname": "torch.optim.Adam",
            "lr": 3e-4,
        },
        "critic_optimizer": {
            "classname": "torch.optim.Adam",
            "lr": 3e-4,
        },
        "entropy_coef_optimizer": {
            "classname": "torch.optim.Adam",
            "lr": 3e-4,
        },
    }

    # Create agent and start training
    cfg = OmegaConf.create(params)
    sac_agent = SACAgent(env.state_dim, env.action_dim, cfg)
    run_sac(sac_agent)

    # Test final policy
    env = MountainCarEnv(render_mode="human")  # Create new env with rendering
    eval_reward = evaluate_policy(sac_agent.eval_policy, env, episodes=3)
    print(f"Final evaluation reward: {eval_reward}")
    env.close()


if __name__ == "__main__":
    main()
