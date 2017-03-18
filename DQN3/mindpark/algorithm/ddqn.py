import numpy as np
import tensorflow as tf
import mindpark as mp
import mindpark.part.preprocess
import mindpark.part.approximation
import mindpark.part.network
import mindpark.part.replay


class DDQN(mp.Algorithm, mp.step.Experience):

    """
    Algorithm: Double Deep Q-Network (DDQN)
    Paper: Deep Reinforcement Learning with Double Q-learning
    Authors: Hasselt, Guez, Silver. 2015
    PDF: https://arxiv.org/pdf/1509.06461v3.pdf
    """

    @classmethod
    def defaults(cls):
        preprocess = 'dqn_2015'
        preprocess_config = dict(frame_skip=4)
        network = 'dqn_2015'
        replay_capacity = 1e5  # 1e6
        start_learning = 5e4
        epsilon = dict(
            from_=1.0, to=0.1, test=0.05, over=1e6, offset=start_learning)
        batch_size = 32
        sync_target = 2500
        initial_learning_rate = 2.5e-4
        optimizer = tf.train.RMSPropOptimizer
        optimizer_config = dict(decay=0.95, epsilon=0.1)
        return mp.utility.merge_dicts(super().defaults(), locals())

    def __init__(self, task, config):
        mp.Algorithm.__init__(self, task, config)
        self._parse_config()
        self._preprocess = self._create_preprocess()
        mp.step.Experience.__init__(self, self._preprocess.above_task)
        self._model = mp.model.Model(self._create_network)
        self._target = mp.model.Model(self._create_network)
        self._target.weights = self._model.weights
        self._sync_target = mp.utility.Every(
            self.config.sync_target, self.config.start_learning)
        print(str(self._model))
        self._learning_rate = mp.utility.Decay(
            self.config.initial_learning_rate, 0, self.task.steps)
        self._cost_metric = mp.Metric(self.task, 'dqn/cost', 1)
        self._sync_target_metric = mp.Metric(self.task, 'dqn/sync_target', 1)
        self._learning_rate_metric = mp.Metric(
            self.task, 'dqn/learning_rate', 1)
        self._memory = self._create_memory()

    def end_epoch(self):
        super().end_epoch()
        if self.task.directory:
            self._model.save(self.task.directory, 'model')

    def perform(self, observ):
        return self._model.compute('qvalues', state=observ)

    def experience(self, observ, action, reward, successor):
        action = action.argmax()
        self._memory.push(observ, action, reward, successor)
        if self.task.step < self.config.start_learning:
            return
        self._train_network()

    @property
    def policy(self):
        # TODO: Why doesn't self.task work here?
        policy = mp.Sequential(self._preprocess.task)
        policy.add(self._preprocess)
        policy.add(self)
        return policy

    def _train_network(self):
        self._model.set_option(
            'learning_rate', self._learning_rate(self.task.step))
        self._learning_rate_metric(self._model.get_option('learning_rate'))
        observ, action, reward, successor = \
            self._memory.batch(self.config.batch_size)
        return_ = self._estimated_return(reward, successor)
        cost = self._model.train(
            'cost', state=observ, action=action, return_=return_)
        self._cost_metric(cost)
        if self._sync_target(self.task.step):
            self._target.weights = self._model.weights
            self._sync_target_metric(True)
        else:
            self._sync_target_metric(False)

    def _estimated_return(self, reward, successor):
        terminal = np.isnan(successor.reshape((len(successor), -1))).any(1)
        successor = np.nan_to_num(successor)
        assert np.isfinite(successor).all()
        # NOTE: Swapping the models below seems to work similarly well.
        future = self._target.compute('qvalues', state=successor)
        choice = self._model.compute('choice', state=successor)
        future = choice.choose(future.T)
        future[terminal] = 0
        return_ = reward + self.config.discount * future
        assert np.isfinite(return_).all()
        return return_

    def _create_memory(self):
        observ_shape = self._preprocess.above_task.observs.shape
        shapes = observ_shape, tuple(), tuple(), observ_shape
        memory = mp.part.replay.Random(self.config.replay_capacity, shapes)
        memory.log_memory_size()
        return memory

    def _create_preprocess(self):
        policy = mp.Sequential(self.task)
        preprocess = getattr(mp.part.preprocess, self.config.preprocess)
        policy.add(preprocess, self.config.preprocess_config)
        policy.add(mp.step.EpsilonGreedy, **self.config.epsilon)
        return policy

    def _create_network(self, model):
        learning_rate = model.add_option(
            'learning_rate', self.config.initial_learning_rate)
        model.set_optimizer(self.config.optimizer(
            learning_rate=learning_rate,
            **self.config.optimizer_config))
        network = getattr(mp.part.network, self.config.network)
        observs = self._preprocess.above_task.observs.shape
        actions = self._preprocess.above_task.actions.shape[0]
        mp.part.approximation.q_function(model, network, observs, actions)

    def _parse_config(self):
        if self.config.start_learning > self.config.replay_capacity:
            raise KeyError('Why not start learning after the buffer is full?')
        if self.config.start_learning < self.config.batch_size:
            raise KeyError('Must collect at least one batch before learning.')
        self.config.start_learning *= self.config.preprocess_config.frame_skip
        self.config.sync_target *= self.config.preprocess_config.frame_skip
        self.config.epsilon.over *= self.config.preprocess_config.frame_skip
