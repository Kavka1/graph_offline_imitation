import d4rl
import gym


if __name__ == "__main__":
    for agent in [
        # 'hopper', 
        # 'halfcheetah', 
        'ant', 
        'walker2d'
    ]:
        for dataset in [
                'random', 
                'expert', 
                'medium', 
                # 'medium-expert', 'medium-replay', 'full-replay'
            ]:
            for version in ['v2']:
                env = gym.make(f'{agent}-{dataset}-{version}')
                env.get_dataset()