from afu.agents.sac import ActorAgent, CriticAgent


def main():
    print("Hello, world")
    actor = ActorAgent()
    critic = CriticAgent()

    actor.forward()
    critic.forward()


if __name__ == "__main__":
    main()
