import os, copy
import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        """builds model using configuration
        config: dict
            key, value pairs are parameter names and values"""

        self.config = copy.deepcopy(config)

        if 'random_seed' in config:
            self.random_seed = self.config['random_seed']

        self.result_dir = self.config['result_dir']
        self.max_iter = self.config['max_iter']
        self.lr = self.config['lr']
        self.nb_units = self.config['nb_units']

        # Now the child Model needs some custom parameters, to avoid any
        # inheritance hell with the __init__ function, the model
        # will override this function completely
        self.set_agent_props()

        self.graph = self.build_graph(tf.Graph())

        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=50,
            )

        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)

        # This function is not always common to all models, that's why it's again
        # separated from the __init__ one
        self.init()

    def set_agent_props(self):
        pass

    def get_best_config(self):
        # This function is here to be overriden completely.
        # It returns a dictionary used to update the initial configuration (see __init__)
        return {}

    @staticmethod
    def get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to other processes
        # so they can independently generate random configuration of the current model
        raise Exception('The get_random_config function must be overridden by the agent')

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overridden by the agent')

    def infer(self):
        raise Exception('The infer function must be overridden by the agent')

    def learn_from_epoch(self):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overridden by the agent')

    def train(self, save_every=1):
        for epoch_id in range(0, self.max_iter):
            self.learn_from_epoch()

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and epoch_id % save_every == 0:
                self.save()

    def save(self):
        global_step_t = tf.train.get_global_step(self.graph)
        global_step, episode_id = self.sess.run([global_step_t, self.episode_id])
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))
        self.saver.save(self.sess, self.result_dir + '/agent-ep_' + str(episode_id), global_step)

        # I always keep the configuration that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:

            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def infer(self):
        # This function is usually common to all your models
        pass
