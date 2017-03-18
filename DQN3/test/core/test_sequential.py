import pytest
from test.mocks import Sequential, Identity, Skip, Random
from test.fixtures import *


class TestSequential:

    def test_add_type_args(self, task):
        policy = Sequential(task)
        policy.add(Skip, 42)
        assert policy.steps[-1]._amount == 42
        policy.add(Skip, amount=13)
        assert policy.steps[-1]._amount == 13

    def test_need_interface_to_add(self, task):
        policy = Sequential(task)
        policy.add(Skip, 42)
        policy.add(Random)
        with pytest.raises(Exception):
            policy.add(Identity)

    def test_reward_not_none(self, env, policy):
        policy.begin_episode(0, True)
        observ = env.reset()
        while observ is not None:
            for step in policy.steps:
                if not step.step:
                    continue
                assert step.reward is not None
            action = policy.observe(observ)
            reward, observ = env.step(action)
            policy.receive(reward, observ is None)
        policy.end_episode()

    def test_receive_last_reward(self, env, policy):
        policy.begin_episode(0, True)
        observ = env.reset()
        while observ is not None:
            action = policy.observe(observ)
            reward, observ = env.step(action)
            for step in policy.steps:
                step.reward = None
            policy.receive(reward, observ is None)
        policy.end_episode()
        for step in policy.steps:
            if step.step is None:
                break
            assert step.reward is not None

    def test_time_steps(self, env, policy):
        policy.begin_episode(0, True)
        timestep = 0
        observ = env.reset()
        while observ is not None:
            action = policy.observe(observ)
            reward, observ = env.step(action)
            policy.receive(reward, observ is None)
            actual, references = [], []
            reference = timestep
            steps = policy.steps[:]
            while steps:
                step = steps.pop(0)
                if step.step is None:
                    break
                actual.append(step.step)
                references.append(reference)
                if isinstance(step, Skip):
                    reference //= step._amount
                if hasattr(step, 'steps'):
                    steps = step.steps + steps
            print('Actual:   ', actual)
            print('Reference:', references)
            assert actual == references
            timestep += 1
        policy.end_episode()
