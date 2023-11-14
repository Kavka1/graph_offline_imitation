# Register environment classes here
from gym.envs import register
from gym.spaces import Box
import numpy as np

NEW_REGISTERED_ENV = {}

try:
    import d4rl
except ImportError:
    print("[research] skipping d4rl, package not found.")


try:
    from .kitchen import KitchenGoalConditionedWrapper, KitchenDropGoalWrapper
    import adept_envs
    register(
        id="kitchen-all-v0",
        entry_point="graph_offline_imitation.envs.kitchen:KitchenAllV0",
        max_episode_steps=280,
        reward_threshold=1.0,
    )
    register(
        id='kitchen-kettle_microwave_bottomburner_hinge-v0',
        entry_point='graph_offline_imitation.envs.kitchen:KitchenKettleMicrowaveBottomBurnerHingeV0',
        max_episode_steps=280,
        reward_threshold=1.0
    )

    NEW_REGISTERED_ENV.update(
        {
            'kitchen': [
                'kitchen-all-v0',
                'kitchen-kettle_microwave_bottomburner_hinge-v0'
            ]
        }
    )

except ImportError:
    print(
        "[research] Could not import Franka Kitchen envs. Please install the Adept Envs via the instructions in Play to"
        " Policy"
    )


try:
    # Register environment classes here
    # Register the DM Control environments.
    from dm_control import suite
    # Custom DM Control domains can be registered as follows:
    # from . import <custom dm_env module>
    # assert hasattr(<custom dm_env module>, 'SUITE')
    # suite._DOMAINS['<custom dm_env module>'] = <custom dm_env module>

    new_dmc_tasks = []

    # Register all of the DM control tasks
    for domain_name, task_name in suite._get_tasks(tag=None):
        # Import state domains
        ID = f"{domain_name.capitalize()}{task_name.capitalize()}-v0"
        register(
            id=ID,
            entry_point="graph_offline_imitation.envs.dm_control:DMControlEnv",
            kwargs={
                "domain_name": domain_name,
                "task_name": task_name,
                "action_minimum": -1.0,
                "action_maximum": 1.0,
                "action_repeat": 1,
                "from_pixels": False,
                "flatten": True,
                "stack": 1,
            },
        )
        new_dmc_tasks.append(ID)

        # Import vision domains as specified in DRQ-v2
        ID = f"{domain_name.capitalize()}{task_name.capitalize()}-vision-v0"
        camera_id = dict(quadruped=2).get(domain_name, 0)
        register(
            id=ID,
            entry_point="graph_offline_imitation.envs.dm_control:DMControlEnv",
            kwargs={
                "domain_name": domain_name,
                "task_name": task_name,
                "action_repeat": 2,
                "action_minimum": -1.0,
                "action_maximum": 1.0,
                "from_pixels": True,
                "height": 84,
                "width": 84,
                "camera_id": camera_id,
                "flatten": False,
                "stack": 3,
            },
        )
        new_dmc_tasks.append(ID)
    
    NEW_REGISTERED_ENV.update({'dmc': new_dmc_tasks})
    # Cleanup extra imports
    del suite
except ImportError:
    print("[research] Skipping dm_control, package not found.")