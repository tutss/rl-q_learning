import gym

def main():
    env = gym.make('LunarLander-v2')

    n_games = 100

    for _ in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, info = env.step(action)
            score += reward
            # env.render()
        print(f"Score: {score:.1f}")

main()
