import torch  # type: ignore
from bbrl.utils.logger import TFLogger  # type: ignore
from bbrl.workspace import Workspace  # type: ignore

from afu.agents.sac import ActorAgent, CriticAgent
from afu.environments.cartpole import create_env_agent


def main():
    logger = TFLogger("runs/sac_cartpole")

    env = create_env_agent()
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    actor = ActorAgent(obs_size, [64, 64], act_size)
    critic1 = CriticAgent(obs_size, [256, 256], act_size)
    critic2 = CriticAgent(obs_size, [256, 256], act_size)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=3e-4)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=3e-4)

    workspace = Workspace()

    for epoch in range(100):
        actor(workspace, t=0, n_steps=1)
        env.step(workspace)

        critic1_opt.zero_grad()
        critic2_opt.zero_grad()
        critic1(workspace, t=0)
        critic2(workspace, t=0)

        reward = workspace.get("env/reward")
        q1_value = workspace.get("q_value")
        q2_value = workspace.get("q_value")

        critic_loss = -torch.mean(q1_value + q2_value)
        critic_loss.backward()
        critic1_opt.step()
        critic2_opt.step()

        actor_opt.zero_grad()
        actor(workspace, t=0)
        actor_loss = -torch.mean(workspace.get("action_logprobs"))
        actor_loss.backward()
        actor_opt.step()

        logger.add_scalar("critic_loss", critic_loss.item(), epoch)
        logger.add_scalar("actor_loss", actor_loss.item(), epoch)


if __name__ == "__main__":
    main()
