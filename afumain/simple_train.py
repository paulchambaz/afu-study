import subprocess
import argparse
import os

env_step_limit = {"Ant": 3, "HalfCheetah": 5, "Hopper": 3, "Humanoid": 3,
                  "InvertedDoublePendulum": 1, "Reacher": 1, "Walker2d": 3}
algos = {"SAC": "train_sac.py", "TD3": "train_td3.py", 
         "AFU": "train_afu.py", "IQL": "train_afu.py",
         "SQL": "train_afu.py", "EQL": "train_afu.py"}
commands = []

def get_command(algo, env, rho=0.2, variant="alpha", hyperparam=1., cpu_option=False):
    output_list = [os.path.join(".", "trainers", algos[algo])]
    if algo == "AFU":
        agent_name = algo + "_" + str(rho)
    elif algo == "IQL" or algo == "SQL" or algo == "EQL":
        agent_name = algo + "_" + str(hyperparam)
    else:
        agent_name = algo
    output_list.append("--save_dir")
    output_list.append("./results/" + agent_name)
    output_list.append("--env_id")
    output_list.append(env + "-v4")
    output_list.append("--num_agent_steps")
    output_list.append(str(env_step_limit[env] * 10 ** 6))
    output_list.append("--eval_interval")
    output_list.append("5000")
    output_list.append("--num_eval_episodes")
    output_list.append("1")
    if algo == "AFU" or algo == "IQL" or algo == "SQL" or algo == "EQL":
        output_list.append("--rho")
        output_list.append(str(rho))
        output_list.append("--variant")
        output_list.append(variant)
        output_list.append("--alg")
        output_list.append(algo)
        output_list.append("--hyperparam")
        output_list.append(str(hyperparam))
    if cpu_option:
        output_list.append("--cpu")
    return output_list

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Ant")
    p.add_argument("--algo", type=str, default="AFU-alpha")
    p.add_argument("--param", type=float, default=0.2)
    p.add_argument("--cpu", action="store_true", default=False)
    arguments = p.parse_args()

    assert arguments.env in [
        "Ant", "HalfCheetah", "Hopper", "Humanoid", "InvertedDoublePendulum", "Reacher", 
        "Walker2d"], \
        'The parameter [--env] must be "Ant", "HalfCheetah", "Hopper", "Humanoid", ' \
        '"InvertedDoublePendulum", "Reacher", or "Walker2d".'
    
    assert arguments.algo in [
        "AFU-alpha", "AFU-beta", "SAC", "TD3", "IQL", "SQL", "EQL"], \
        'The parameter [--algo] must be "AFU-alpha", "AFU-beta", "SAC", "TD3", ' \
        '"IQL", "SQL", or "EQL".'
    
    variant="alpha"

    if arguments.algo == "AFU-alpha":
        arguments.algo = "AFU"
    
    if arguments.algo == "AFU-beta":
        arguments.algo = "AFU"
        variant = "beta"

    rho = 0.2
    if arguments.algo == "AFU":
        rho = arguments.param

    hyperparam=1.
    if arguments.algo == "IQL" or arguments.algo == "SQL" or arguments.algo == "EQL":
        hyperparam = arguments.param

    command_list = ["python"] + get_command(
        arguments.algo,
        arguments.env,
        rho=rho,
        variant="alpha",
        hyperparam=hyperparam,
        cpu_option=arguments.cpu)

    print(' '.join(command_list))
    subprocess.run(
        command_list
    )
