import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def build_mdp(gamma=0.95, theta=1e-6):
    mdp = {}
    mdp["grid_size"] = (4, 4)
    states = []
    for x in range(4):
        for y in range(4):
            if (x, y) == (3, 3):
                states.append((x, y, 0, 0))  # goal state
            elif (x, y) in [(0, 1), (0, 2)]:
                states.append((x, y, 0, 1))  # fire states
            elif (x, y) in [(2, 1), (2, 2)]:
                states.append((x, y, 1, 0))  # water states
            else:
                states.append((x, y, 0, 0))  # normal states
    
    mdp["states"] = states
    mdp["actions"] = [0, 1, 2, 3]  # 0: up, 1: down, 2: left, 3: right
    mdp["goal_state"] = (3, 3, 0, 0)
    mdp["gamma"] = gamma
    mdp["theta"] = theta

    mdp["transitionProbabilities"] = {
        0: [0.8, 0.0, 0.1, 0.1],  # up:80%, left: 10%, right:10%
        1: [0.0, 0.8, 0.1, 0.1],  # down:80%, left: 10%, right:10%
        2: [0.1, 0.1, 0.8, 0.0],  # left:80%, up: 10%, down:10%
        3: [0.1, 0.1, 0.0, 0.8],  # right:80%, up: 10%, down:10%
    }

    return mdp

def get_rewards(state):
    x, y, water, fire = state
    if (x, y) == (3, 3):
        return 100  # goal state
    elif fire == 1:
        return -10  # fire penalty
    elif water == 1:
        return -5  # water penalty
    else:
        return -1  # movement cost

def get_next_states(state, action, mdp):
    next_states = {}
    x, y, water, fire = state
    for i in range(4):
        prob = mdp["transitionProbabilities"][action][i]
        if i == 0:  # up
            nx, ny = x, max(0, y - 1)
        elif i == 1:  # down
            nx, ny = x, min(3, y + 1)
        elif i == 2:  # left
            nx, ny = max(0, x - 1), y
        elif i == 3:  # right
            nx, ny = min(3, x + 1), y

        if (nx, ny) == (3, 3):
            newWater, newFire = 0, 0
        else:
            newWater = 1 if (nx, ny) in [(2, 1), (2, 2)] else 0
            newFire = 1 if (nx, ny) in [(0, 1), (0, 2)] else 0

        next_state = (nx, ny, newWater, newFire)

        if next_state in next_states:
            next_states[next_state] += prob
        else:
            next_states[next_state] = prob

    return next_states

def value_iteration(mdp):
    V = {s: 0 for s in mdp["states"]}
    policy = {s: 0 for s in mdp["states"]}
    iteration = 0

    while True:
        delta = 0
        newV = V.copy()

        for state in mdp["states"]:
            if state == mdp["goal_state"]:
                continue

            action_values = []

            for action in mdp["actions"]:
                next_states = get_next_states(state, action, mdp)
                value = sum(prob * (get_rewards(s) + mdp["gamma"] * V[s])for s, prob in next_states.items())
                action_values.append(value)
            
            best_value = max(action_values)
            best_action = np.argmax(action_values)

            newV[state] = best_value
            policy[state] = best_action

            delta = max(delta, abs(V[state] - newV[state]))
        
        V = newV
        iteration += 1

        print(f"Iteration {iteration}: max change = {delta:.6f}")

        # convergence check
        if delta < mdp["theta"]:
            break
    
    return V, policy

def policy_iteration(mdp):
    policy = {}
    for state in mdp["states"]:
        if state == mdp["goal_state"]:
            policy[state] = None
        else:
            policy[state] = np.random.choice(mdp["actions"])
    V = {s: 0 for s in mdp["states"]}
    iteration = 0
    while True:
        iteration += 1

        # policy evaluation
        while True:
            delta = 0
            for state in mdp["states"]:
                if state == mdp["goal_state"]:
                    continue
                old_v = V[state]
                a = policy[state]
                next_states = get_next_states(state, a, mdp)
                V[state] = sum(prob * (get_rewards(s) + mdp["gamma"] * V[s]) for s, prob in next_states.items())
                delta = max(delta, abs(old_v - V[state]))
            if delta < mdp["theta"]:
                break
        
        # policy improvement
        policy_stable = True
        for state in mdp["states"]:
            if state == mdp["goal_state"]:
                continue
            old_action = policy[state]
            action_values = []
            for action in mdp["actions"]:
                next_states = get_next_states(state, action, mdp)
                value = sum(prob * (get_rewards(s) + mdp["gamma"] * V[s]) for s, prob in next_states.items())
                action_values.append(value)
            best_action = np.argmax(action_values)
            policy[state] = best_action
            if best_action != old_action:
                policy_stable = False
        
        print(f"Policy Iteration {iteration} done")
        if policy_stable:
            break

    return V, policy

def simulate_episode(policy, mdp, T = 100):
        episode = []
        current_state = (0,0,0,0)
        steps = 0

        while current_state != mdp["goal_state"] and steps < T:
            action = policy[current_state]
            next_states = get_next_states(current_state, action, mdp)

            keys = list(next_states.keys())
            probs = list(next_states.values())
            idx = np.random.choice(len(keys), p=probs)
            next_state = keys[idx]

            reward = get_rewards(next_state)

            episode.append((current_state, action, reward))
            current_state = next_state
            steps += 1

        episode.append((mdp["goal_state"], None, get_rewards(mdp["goal_state"])))
        return episode

def dagger(mdp, expert_policy, N, T=100):

    D_states = []  # store [x, y, water, fire]
    D_actions = []  # store expert action

    # learned_policy
    learned_policy = {s: np.random.choice(mdp["actions"]) for s in mdp["states"]}

    # accuracy of learned policy
    acc_list = [] 

    # DAGGER iterations
    for i in range(1, N+1):
        print(f"\nDAGGER iteration {i}")

        trajectory = simulate_episode(learned_policy, mdp, T)
        
        for (s, _, _) in trajectory:
            D_states.append(list(s))
            D_actions.append(expert_policy[s])
        
        clf = DecisionTreeClassifier()
        clf.fit(D_states, D_actions)

        # update the learned policy
        test_states = [list(s) for s in mdp["states"]]
        pred_actions = clf.predict(test_states)
        learned_policy = {s: a for s, a in zip(mdp["states"], pred_actions)}

        # evaluate the accuracy
        expert_labels = [expert_policy[s] for s in mdp["states"]]
        acc = accuracy_score(expert_labels, pred_actions)
        acc_list.append(acc)
        print(f"Iteration {i}: Accuracy = {acc:.4f}")
    
    return learned_policy, acc_list


def main():
    # Part A: value iteration & policy iteration
    start_state = (0, 0, 0, 0)
    
    print("=== Part A: Baseline Solutions ===")
    # value iteration with gamma = 0.3
    mdp_03 = build_mdp(gamma=0.3, theta=1e-6)
    V_vi_03, policy_vi_03 = value_iteration(mdp_03)
    print("\nFinal state value function V from Value Iteration (gamma=0.3):")
    for state, value in sorted(V_vi_03.items()):
        print(f"  State {state}: Value {value:.2f}")
    print(f"Objective Value (gamma=0.3): {V_vi_03[start_state]:.2f}")
    
    # value iteration with gamma = 0.95
    mdp_095 = build_mdp(gamma=0.95, theta=1e-6)
    V_vi_095, policy_vi_095 = value_iteration(mdp_095)
    print("\nFinal state value function V from Value Iteration (gamma=0.95):")
    for state, value in sorted(V_vi_095.items()):
        print(f"  State {state}: Value {value:.2f}")
    print(f"Objective Value (gamma=0.95): {V_vi_095[start_state]:.2f}")
    
    # policy iteration with gamma = 0.95
    V_pi, policy_pi = policy_iteration(mdp_095)
    print("\nFinal state value function V from Policy Iteration (gamma=0.95):")
    for state, value in sorted(V_pi.items()):
        print(f"  State {state}: Value {value:.2f}")
    print(f"Objective Value (Policy Iteration, gamma=0.95): {V_pi[start_state]:.2f}")
    
    # simulate episode (use value iteration)
    episode = simulate_episode(policy_vi_095, mdp_095)
    print("\nEpisode (using Value Iteration policy, gamma=0.95):")
    for t, (state, action, reward) in enumerate(episode):
        print(f"Step {t}: State {state}, Action {action}, Reward {reward}")
    
    # Part B: DAGGER
    print("\n=== Part B: DAGGER Experiment ===")
    # gamma = 0.95
    mdp = build_mdp(gamma=0.95, theta=1e-6)
    # expert policy from value iteration
    V_expert, expert_policy = value_iteration(mdp)
    print("\nExpert policy obtained from Value Iteration (gamma=0.95):")
    for state, action in sorted(expert_policy.items()):
        print(f"  State {state}: Expert action {action}")
    
    N_list = [5, 10, 20, 30, 40, 50]
    acc_results = {}
    
    for N in N_list:
        print(f"\nRunning DAGGER with N = {N}")
        learned_policy, acc_list = dagger(mdp, expert_policy, N, T=100)
        acc_results[N] = acc_list[-1]
    
    print("\nFinal accuracy for different N:")
    for N in N_list:
        print(f"  N = {N}: Accuracy = {acc_results[N]:.4f}")
    
    plt.figure()
    plt.plot(N_list, [acc_results[N] for N in N_list], marker='o')
    plt.xlabel('Number of DAGGER Iterations (N)')
    plt.ylabel('Accuracy')
    plt.title('DAGGER: Accuracy vs N')
    plt.show()

if __name__ == "__main__":
    main()