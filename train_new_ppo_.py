def run_training1(name_of_agent, max_eps, write_to_file=False):
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario3'
    sg = DroneSwarmScenarioGenerator(num_drones=num_drones, maximum_steps=500)
    cyborg = CybORG(sg, 'sim')
    wrapped_cyborg = PettingZooParallelWrapper(env=cyborg)
    observation = wrapped_cyborg.reset()
    
    if technique == 'PPO':
#         print(wrapped_cyborg.observation_space(f"blue_agent_0").shape[0]-(2*(num_drones-1)+2))
#         print(wrapped_cyborg.observation_space(f"blue_agent_0"))
        print(wrapped_cyborg.get_action_space('blue_agent_0'))

#         agents = {f"blue_agent_{agent}": PPO(wrapped_cyborg.observation_space(f"blue_agent_{agent}").shape[0], len(wrapped_cyborg.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, True, 'CybORG\Evaluation\submission\Models\\5110.pth') for agent in range(18)}
#         agents = {f"blue_agent_{agent}": PPO(wrapped_cyborg.observation_space(f"blue_agent_{agent}").shape[0], len(wrapped_cyborg.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, True, None) for agent in range(num_drones)}
        agents = {f"blue_agent_{agent}": PPO(wrapped_cyborg.observation_space(f"blue_agent_{agent}").shape[0]-(0*(num_drones-1)+0), len(wrapped_cyborg.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, False, None) for agent in range(num_drones)}
#         agents = {f"blue_agent_{agent}": PPO(wrapped_cyborg.observation_space(f"blue_agent_{agent}").shape[0]-(3*(num_drones-1)+2), len(wrapped_cyborg.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, False, None) for agent in range(num_drones)}
        print(f'Using agents {agents}, if this is incorrect please update the code to load in your agent')
        if write_to_file:
            file_name = str(inspect.getfile(CybORG))[:-7] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S")
            print(f'Saving evaluation results to {file_name}_summary.txt and {file_name}_full.txt')
        start = datetime.now()

        print(f'using CybORG v{cyborg_version}, {scenario}\n')

        manual_actions = []
        total_reward = []
        actions_log = []
        obs_log = []
        timestep = 0
        update_timestep = 200
        total_reward_n = 0
        mean_total_reward = 0
        for i in range(10000):     # Episode
            observation = wrapped_cyborg.reset()
#             print(type(observation))
#             print(observation)
#             new_observation = remove_elements_from_dict_values(observation, num_drones)
            new_observation = observation
#             print(new_observation)

            r = []
            a = []
            o = []
            for j in range(1000):    # Steps
                timestep += 1
                actions = {agent_name: agent.get_action(new_observation[agent_name], agent.memory) for agent_name, agent in agents.items() if agent_name in wrapped_cyborg.agents}
#                 print("÷ction = ", j, actions)
                manual_actions = []

#                 for key, value in actions.items():
#                 for key, value in actions.items():
#                     print('episode, step, key, obs, new_obs, actions')
#                     print("Ep: ", i, ", Step: ", j)
#                     print(key)
#                     print(observation[key])
#                     print_Obs(key, observation[key], agents[key])
#                     print(new_observation[key])
#                     print('policy actions = ', actions[key])
#                     print_Actions(wrapped_cyborg.get_action_space('blue_agent_0'))
#                     temp = int(input("Please enter action index: "))
#                     manual_actions.append(temp)
                
#                 print("manual actions = ", manual_actions)
                observation, rew, done, info = wrapped_cyborg.step(actions)
#                 print(j, rew)
    
#                 rew = {key: -value for key, value in rew.items()}
#                 print(rew, done)
#                 new_observation = remove_elements_from_dict_values(observation, num_drones)
#                 print(done)
                new_observation = observation

                for agent_name, agent in agents.items():
                    if agent_name in actions:
                        if agent_name in wrapped_cyborg.agents:
                            agent.memory.rewards.append(rew[agent_name])
#                             print("agent.memory.rewards = ", agent.memory.rewards)
                            agent.memory.is_terminals.append(done[agent_name])
                        else:
                            agent.memory.rewards.append(-1)
                            agent.memory.is_terminals.append(True)
                r.append(mean(rew.values()))
                if all(done.values()):
#                     with open(file_name+'_1.txt', 'w') as data:
#                         data.write(f'R:{r}\n')
                    print("steps = ", j)
                    print('all done')
                    break
                if write_to_file:
                    a.append({agent_name: wrapped_cyborg.get_action_space(agent_name)[actions[agent_name]] for agent_name in actions.keys()})
                    o.append({agent_name: observation[agent_name] for agent_name in observation.keys()})
#                 print("rew.values = ", rew.values())
#                 print("r = ", r)
                
                if timestep % update_timestep == 0:
                    for agent_name, agent in agents.items():
                        if agent_name in wrapped_cyborg.agents:
                            agent.update()
                            agent.memory.clear_memory()
                    timestep = 0
#             total_reward.append(sum(r))
            total_reward_n += 1
            if write_to_file:
                actions_log.append(a)
                obs_log.append(o)
            mean_total_reward += (1/total_reward_n)*(sum(r)-mean_total_reward)   # for debugging mem overflow error
#             mean_total_reward /= total_reward_n    # mem overflow error seen
            print(i, ": ", mean_total_reward, sum(r))
        ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i))
        torch.save(agent.policy.state_dict(), ckpt)
        print('Checkpoint saved')
        end = datetime.now()
        difference = end-start
        print(
#             f'Average reward is: {mean_total_reward} with a standard deviation of {stdev(total_reward)}')
            f'Average reward is: {mean_total_reward} ')
        print(f'file took {difference} amount of time to finish evaluation')
        if write_to_file:
            with open(file_name+'_summary.txt', 'w') as data:
                data.write(f'CybORG v{cyborg_version}, {scenario}\n')
                data.write(
                    f'author: Jay, technique: {name_of_agent}\n')
                data.write(
                    f'Average reward is: {mean_total_reward} with a standard deviation of')
                data.write(f'Using agents {agents}')

            with open(file_name+'_full.txt', 'w') as data:
                data.write(f'CybORG v{cyborg_version}, {scenario}\n')
                data.write(
                    f'author: Jay, technique: {name_of_agent}\n')
                data.write(
                    f'mean: {mean_total_reward}, standard deviation {mean_total_reward}\n')
                for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                    data.write(
                        f'actions: {act},\n observations: {obs} \n total reward: {sum_rew}\n')
