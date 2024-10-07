from cluster import load_data, fragment_signals, scale_data, denormalize_data, apply_tsne, plot_tsne
import numpy as np
import json
import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow import data as tfdata
from tensorflow import float32
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Add, Conv1D, Dropout, Attention
from tensorflow import ones_like, zeros_like, sqrt, reduce_mean
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.nn import moments as nn_moments 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization


seq_len = 200
model_call = "randomnormal"
clip_norm = 1.0
n_seq = 1
threshold_improvement = 0.03
last_n_epochs = 100
k = 10
k_g = 5
train_gen_first = 100
k_d = 1
hidden_dim = 24
gamma = 1
saving_interval = 25
noise_dim = 40
dim = 64
batch_size = 64

log_step = 10
learning_rate = 5e-5 #changed learning rate from 5e-4
disc_learning_rate = 1e-5
train_steps = 1000
starting_gen_epoch =1000
clip_value = 10
prev_d_loss = None
prev_g_loss = None
gan_args = batch_size, learning_rate, noise_dim, 24, 2, (0, 1), dim



def save_signals_to_txt(file_name, signals):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)  # Create directories if they don't exist
    with open(file_name, 'w') as file:
        for signal in signals:
            line = ' '.join(map(str, signal))
            file.write(line + '\n\n')

def log_gradients(gradients):
    gradients_log = []
    for i, grad in enumerate(gradients):
        if grad is not None:
            grad_norm = tf.norm(grad)
            gradients_log.append(f"Layer {i}: Gradient Norm = {grad_norm.numpy()}\n")  
    return gradients_log

def save_gradient_logs(gradients_log, gradients_file=f'models/gradients/gradients_{train_steps}_{k}_withBatchNorm_{model_call}.txt'):
    directory = os.path.dirname(gradients_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(gradients_file, 'a') as file:
        for log in gradients_log:
            file.write(log)

def save_training_ratios(k_d, k_g, filename="training_ratios.json"):
    """
    Save the training ratios k_d and k_g to a JSON file.
    """
    ratios = {"k_d": k_d, "k_g": k_g}
    with open(filename, "w") as file:
        json.dump(ratios, file)
def load_training_ratios(filename="training_ratios.json"):
    """
    Load the training ratios k_d and k_g from a JSON file.
    """
    with open(filename, "r") as file:
        ratios = json.load(file)
    return ratios["k_d"], ratios["k_g"]

# Example usage
k_d, k_g = load_training_ratios()


def net(model, n_layers, hidden_units, output_units, net_type='GRU'):
    regularizer = l2(0.01) 

    # Add Conv1D layers
    for i in range(1, 4):
        model.add(Conv1D(filters=hidden_units, kernel_size=3, padding='same', 
                         activation='relu', 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),  # Random Normal Initialization
                         name=f'Conv1D_{i}'))
        model.add(Dropout(0.2)) 

    if net_type == 'GRU':
        for i in range(n_layers):
            model.add(GRU(units=hidden_units,
                          return_sequences=True,
                          kernel_regularizer=regularizer,
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),  # Random Normal Initialization
                          name=f'GRU_{i + 1}'))
            if (i + 1) % 2 == 0:
                model.add(BatchNormalization())
            model.add(Dropout(0.2))  
    else:
        for i in range(n_layers):
            model.add(LSTM(units=hidden_units,
                           return_sequences=True,
                           kernel_regularizer=regularizer,
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),  # Random Normal Initialization
                           name=f'LSTM_{i + 1}'))
            if (i + 1) % 2 == 0:
                model.add(BatchNormalization())
            model.add(Dropout(0.2)) 

    model.add(Dense(units=output_units,
                    activation='sigmoid',
                    kernel_regularizer=regularizer,
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),  # Random Normal Initialization
                    name='OUT'))
    
    return model

class Generator(Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.net_type = net_type
        self.model = self.build_model()

    def build_model(self):
        model = Sequential(name='Generator')
        return net(model, n_layers=10, hidden_units=self.hidden_dim, output_units=self.hidden_dim, net_type=self.net_type)

class Discriminator(Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.net_type = net_type
        self.model = self.build_model()

    def build_model(self):
        model = Sequential(name='Discriminator')
        return net(model, n_layers=5, hidden_units=self.hidden_dim, output_units=1, net_type=self.net_type)

class Recovery(Model):
    def __init__(self, hidden_dim, n_seq):
        super(Recovery, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_seq = n_seq
        self.model = self.build_model()

    def build_model(self):
        recovery = Sequential(name='Recovery')
        return net(recovery, n_layers=3, hidden_units=self.hidden_dim, output_units=self.n_seq)

class Embedder(Model):
    def __init__(self, hidden_dim):
        super(Embedder, self).__init__()
        self.hidden_dim = hidden_dim
        self.model = self.build_model()

    def build_model(self):
        embedder = Sequential(name='Embedder')
        return net(embedder, n_layers=3, hidden_units=self.hidden_dim, output_units=self.hidden_dim)

class Supervisor(Model):
    def __init__(self, hidden_dim):
        super(Supervisor, self).__init__()
        self.hidden_dim = hidden_dim
        self.model = self.build_model()

    def build_model(self):
        model = Sequential(name='Supervisor')
        return net(model, n_layers=2, hidden_units=self.hidden_dim, output_units=self.hidden_dim)

def train_test_split_no_shuffle(data, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data
    
def preprocess():
        my_data, _, _ = load_data()
        my_data_frag_with_overlap = fragment_signals(my_data, 200, 25)
        scaled_data, scaler = scale_data(my_data_frag_with_overlap)
        train_data, test_data = train_test_split_no_shuffle(scaled_data, test_size=0.2)
        X = np.expand_dims(train_data, axis=-1)
        X_test = np.expand_dims(test_data, axis = -1)
        return X,X_test, scaler

class TimeGAN():
    def __init__(self, model_parameters, hidden_dim, seq_len, n_seq, gamma, scaler,model_dir = "models"):
        self.seq_len = seq_len
        self.batch_size, self.lr, self.beta_1, self.beta_2, self.noise_dim, self.data_dim, self.layers_dim = model_parameters
        self.n_seq = n_seq
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.scaler = scaler
        self.model_dir = model_dir
        self.define_gan()
    
    def load_or_create_model(self, model_name, model_instance, loss_type='mse'):
        # Use the file path with .keras extension as models are saved in .keras format
        model_path = os.path.join(self.model_dir, f'{model_name}_10_{model_call}.keras')
        
        if os.path.exists(model_path):
            print(f"Loading existing model: {model_name}")
            # Load the model from the .keras file
            model_instance = tf.keras.models.load_model(model_path)
            # Compile the model after loading
            model_instance.compile(optimizer=Adam(learning_rate=self.lr), loss=loss_type)
        else:
            print(f"Creating new model: {model_name}")
            
        return model_instance

    
    def save_models(self):
        # Save models with .keras extension
        self.generator_aux.save(os.path.join(self.model_dir, f'generator_aux_10_{model_call}.keras'))
        self.supervisor.save(os.path.join(self.model_dir, f'supervisor_10_{model_call}.keras'))
        self.discriminator.save(os.path.join(self.model_dir, f'discriminator_10_{model_call}.keras'))
        self.recovery.save(os.path.join(self.model_dir, f'recovery_10_{model_call}.keras'))
        self.embedder.save(os.path.join(self.model_dir, f'embedder_10_{model_call}.keras'))
        
        print("Models have been saved with .keras extension.")


    def define_gan(self):
        self.generator_aux = self.load_or_create_model('generator_aux', Generator(self.hidden_dim).model)
        self.supervisor = self.load_or_create_model('supervisor', Supervisor(self.hidden_dim).model)
        self.discriminator = self.load_or_create_model('discriminator', Discriminator(self.hidden_dim).model)
        self.recovery = self.load_or_create_model('recovery', Recovery(self.hidden_dim, self.n_seq).model)
        self.embedder = self.load_or_create_model('embedder', Embedder(self.hidden_dim).model)

        X = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RealData')
        Z = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RandomNoise')

        # AutoEncoder
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        self.autoencoder = Model(inputs=X, outputs=X_tilde)

        # Adversarial Supervised Architecture
        E_Hat = self.generator_aux(Z)
        H_hat = self.supervisor(E_Hat)
        Y_fake = self.discriminator(H_hat)

        self.adversarial_supervised = Model(inputs=Z, outputs=Y_fake, name='AdversarialSupervised')

        # Adversarial architecture in latent space
        Y_fake_e = self.discriminator(E_Hat)

        self.adversarial_embedded = Model(inputs=Z, outputs=Y_fake_e, name='AdversarialEmbedded')

        # Synthetic data generation
        X_hat = self.recovery(H_hat)

        self.generator = Model(inputs=Z, outputs=X_hat, name='FinalGenerator')

        # Final discriminator model
        Y_real = self.discriminator(H)

        self.discriminator_model = Model(inputs=X, outputs=Y_real, name="RealDiscriminator")


        # Loss tf.functions
        self._mse=MeanSquaredError()
        self._bce=BinaryCrossentropy()


class TimeGAN(TimeGAN):
    def __init__(self, model_parameters, hidden_dim, seq_len, n_seq, gamma, scaler, model_dir="models"):
        super().__init__(model_parameters, hidden_dim, seq_len, n_seq, gamma, scaler, model_dir)
    
    
    @tf.function
    def train_autoencoder(self, x, opt):
        with tf.GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss_0 = 10 * sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    @tf.function
    def train_supervisor(self, x, opt):
        with tf.GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            g_loss_s = self._mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        var_list = self.supervisor.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return g_loss_s

    @tf.function
    def train_embedder(self,x, opt):
        with tf.GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss = 10 * sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)
    
    def adjust_discriminator_output(self,y_real, y_fake, y_fake_e,randomness_factor=0.01):
        # Adjust the outputs based on their probabilities
        adjusted_y_real = tf.where(y_real < 0.5 + randomness_factor, 
                                    tf.random.uniform(tf.shape(y_real)), 
                                    y_real)
        adjusted_y_fake = tf.where(y_fake > 0.5 - randomness_factor, 
                                    tf.random.uniform(tf.shape(y_fake)), 
                                    y_fake)
        adjusted_y_fake_e = tf.where(y_fake_e > 0.5 - randomness_factor, 
                                    tf.random.uniform(tf.shape(y_fake_e)), 
                                    y_fake_e)
        
        return adjusted_y_real, adjusted_y_fake, adjusted_y_fake_e

    def discriminator_loss(self, x, z):
        y_real = self.discriminator_model(x)
        print("Y_real:",y_real.numpy())
        y_fake = self.adversarial_supervised(z)
        print("Y_fake:",y_fake.numpy())
        y_fake_e = self.adversarial_embedded(z)
        print("Y_fake_e:",y_fake_e.numpy())
        # Adjust outputs with randomness
        adjusted_y_real, adjusted_y_fake, adjusted_y_fake_e = self.adjust_discriminator_output(y_real, y_fake, y_fake_e)

        # print("Adjusted Y Real:", adjusted_y_real.numpy())
        # print("Adjusted Y Fake:", adjusted_y_fake.numpy())
        # print("Adjusted Y Fake E:", adjusted_y_fake_e.numpy())

        discriminator_loss_real = self._bce(y_true=tf.ones_like(adjusted_y_real),  
                                            y_pred=adjusted_y_real)
        discriminator_loss_fake = self._bce(y_true=tf.zeros_like(adjusted_y_fake), 
                                            y_pred=adjusted_y_fake)
        discriminator_loss_fake_e = self._bce(y_true=tf.zeros_like(adjusted_y_fake_e),  
                                            y_pred=adjusted_y_fake_e)
        correct_real = tf.reduce_mean(tf.cast(tf.equal(tf.round(adjusted_y_real), tf.ones_like(y_real)), tf.float32))
        correct_fake = tf.reduce_mean(tf.cast(tf.equal(tf.round(adjusted_y_fake), tf.zeros_like(y_fake)), tf.float32))
        correct_fake_e = tf.reduce_mean(tf.cast(tf.equal(tf.round(adjusted_y_fake_e), tf.zeros_like(y_fake_e)), tf.float32))
        correct_list= [correct_real, correct_fake, correct_fake_e]

        return discriminator_loss_real ,discriminator_loss_fake , self.gamma * discriminator_loss_fake_e, correct_list

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = nn_moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = nn_moments(x=y_pred, axes=[0])
        g_loss_mean = reduce_mean(abs(y_true_mean - y_pred_mean))
        g_loss_var = reduce_mean(abs(sqrt(y_true_var + 1e-6) - sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    @tf.function
    def train_generator(self, x, z, opt):
        with tf.GradientTape() as tape:
            y_fake = self.adversarial_supervised(z)
            #print("Y_fake:",y_fake.numpy())
            generator_loss_unsupervised = self._bce(y_true=ones_like(y_fake),
                                                    y_pred=y_fake)

            y_fake_e = self.adversarial_embedded(z)
            #print("Y_fake_e:",y_fake.numpy())
            generator_loss_unsupervised_e = self._bce(y_true=ones_like(y_fake_e),
                                                      y_pred=y_fake_e)
            h = self.embedder(x)
            #print("H:",h.numpy())
            h_hat_supervised = self.supervisor(h)
            #print("H_hat:",h_hat_supervised.numpy())
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_hat = self.generator(z)
            #print("X: ",x.numpy() )
            #print("X_hat:",x_hat.numpy())
            generator_moment_loss = self.calc_generator_moments_loss(x, x_hat)

            generator_loss = (generator_loss_unsupervised +
                              generator_loss_unsupervised_e +
                              100 * sqrt(generator_loss_supervised) +
                              100 * generator_moment_loss)

        var_list = self.generator_aux.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        # Clip gradients by norm
        clipped_gradients = [tf.clip_by_norm(grad, clip_norm) for grad in gradients if grad is not None]
        #gradients_log = log_gradients(clipped_gradients)
        #print("Clipped gradients:", clipped_gradients, "Versus unclipped ones:", gradients)
        opt.apply_gradients(zip(clipped_gradients, var_list))
        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    #@tf.function
    def train_discriminator(self, x, z, opt):
        with tf.GradientTape() as tape:
            discriminator_loss_real ,discriminator_loss_fake , discriminator_loss_fake_e, correct_list = self.discriminator_loss(x, z)
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake + discriminator_loss_fake_e

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        # Clip gradients by norm
        clipped_gradients = [tf.clip_by_norm(grad, clip_norm) for grad in gradients if grad is not None]
        opt.apply_gradients(zip(clipped_gradients, var_list))
        return discriminator_loss_real ,discriminator_loss_fake , discriminator_loss_fake_e, correct_list

    def get_batch_data(self, data, n_windows):
        # Create the TensorFlow dataset and shuffle it
        data = tf.cast(data, dtype=tf.float32)
        data_iter = iter(tf.data.Dataset.from_tensor_slices(data)
                                    .shuffle(buffer_size=n_windows)
                                    .batch(batch_size)
                                    .repeat())
        
        return data_iter

    def _generate_noise(self):
        while True:
            yield np.random.uniform(low=0, high=1.0, size=(self.seq_len, self.n_seq))

    def get_batch_noise(self):
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=float32)
                                .batch(self.batch_size)
                                .repeat())

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in range(steps, desc='Synthetic data generation'):
            Z_ = next(self.get_batch_noise())
            records = self.generator(Z_)
            data.append(records)
        return np.array(np.vstack(data))
    
    def generate_samples(self, n_samples=1000):
        generated_samples = []
        total_steps = n_samples // self.batch_size + (1 if n_samples % self.batch_size != 0 else 0)
        
        for _ in range(total_steps):
            Z_ = next(self.get_batch_noise())
            generated_batch = self.adversarial_supervised(Z_)
            generated_samples.append(generated_batch)

        return np.array(np.vstack(generated_samples))[:n_samples]
        

    def save_generated_signals(self):
        generated_signals = self.generate_samples(n_samples = self.batch_size)
        generated_samples_list = [generated_signals[i].tolist() for i in range(batch_size)]
        generated_samples_list = np.squeeze(generated_samples_list, axis=-1)
        # for signal in generated_samples_list[0]:
        #     print(signal)
        min_value = self.scaler.data_min_
        max_value = self.scaler.data_max_
        range_value = self.scaler.data_range_

        # What near 0 (e.g., 0.001) is mapped to
        almost_zero_mapped = 0.001 * range_value + min_value

        # What near 1 (e.g., 0.999) is mapped to
        almost_one_mapped = 0.999 * range_value + min_value
        denormalized_generated_list = denormalize_data(generated_samples_list, self.scaler)
        save_signals_to_txt(f'models/samples/generated_samples_{train_steps*k}_{k}_{model_call}.txt', denormalized_generated_list)

def adjust_training_ratio(generator_loss, discriminator_loss_real,discriminator_loss_fake, k_g, k_d, step):
    """
    Adjust the training ratio of generator (k_g) and discriminator (k_d) based on their losses.
    """
    if step <= 50:
        k_g = 5
        k_d = 1
    elif discriminator_loss_fake < 0.5:
        k_g = min(10, k_g+1)
        k_d = max(1, k_d -1)
    elif discriminator_loss_real > 1.0:
        k_d = 5
        k_g = 1
    else:
        k_d = 1
        k_g = 1

    return k_g, k_d

def plot_losses(losses):
    generator_losses, discriminator_losses_real, discriminator_losses_fake = losses
    loss_indices = range(0, len(generator_losses))
    plt.figure(figsize=(14, 8))

    # Plot each loss with its respective indices
    plt.plot(loss_indices, generator_losses, label='Generator Loss', color='green')
    plt.plot(loss_indices, discriminator_losses_real, label='Discriminator Loss for Real Samples', color='red')
    plt.plot(loss_indices, discriminator_losses_fake, label='Discriminator Loss for Generated Samples', color='blue')

    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Time')
    plt.legend()
    plt.savefig(f"models/plots/losses{train_steps}_{k}_withBatchNorm_{model_call}.png")


data,test_data, scaler = preprocess()
print(data[0:10])
synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1, scaler = scaler)
losses_file = f'losses_500_10_withBatchNorm_{model_call}.pkl'
if os.path.exists(losses_file):
    with open(losses_file, 'rb') as file:
        generator_losses, discriminator_losses_real, discriminator_losses_fake = pickle.load(file)
else:
    generator_losses, discriminator_losses_real, discriminator_losses_fake = [], [], []

autoencoder_opt = Adam(learning_rate=learning_rate)
progress_file = "progress.txt"
if os.path.exists(progress_file):
    with open(progress_file, 'r') as file:
        lines = file.readlines()
        completed_steps_embedder = 0
        completed_steps_supervisor = 0
        completed_steps_joint = 0

        # Parse available lines
        if len(lines) > 0:
            completed_steps_embedder = int(lines[0].strip())
        if len(lines) > 1:
            completed_steps_supervisor = int(lines[1].strip())
        if len(lines) > 2:
            completed_steps_joint = int(lines[2].strip())

for step in tqdm(range(completed_steps_embedder + 1, train_steps + 1), desc='Embedding network training'):
    X_ = next(synth.get_batch_data(data, n_windows=len(data)))
    step_e_loss_t0 = synth.train_autoencoder(X_, autoencoder_opt)
    completed_steps_embedder +=1
    if step % saving_interval == 0:
        synth.save_models()
supervisor_opt = Adam(learning_rate=learning_rate)
for step in tqdm(range(completed_steps_supervisor + 1, train_steps + 1), desc='Supervised network training'):
    X_ = next(synth.get_batch_data(data, n_windows=len(data)))
    step_g_loss_s = synth.train_supervisor(X_, supervisor_opt)
    completed_steps_supervisor +=1
    if step % saving_interval == 0 or completed_steps_supervisor == train_steps:
        synth.save_models()
generator_opt = Adam(learning_rate=learning_rate)
embedder_opt = Adam(learning_rate=learning_rate)
discriminator_opt = Adam(learning_rate=disc_learning_rate)
step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss_real = step_d_loss_fake = 0
k_d, k_g = load_training_ratios()
# Your training loop
for step in tqdm(range(completed_steps_joint + 1, starting_gen_epoch + 1), desc='Joint networks training'):
    # Train generator k_g times
    for _ in range(k_g):
        X_ = next(synth.get_batch_data(data, n_windows=len(data)))
        #tf.print("Real batch data (first 5 samples):", X_[:5])
        Z_ = next(synth.get_batch_noise())
        #print(f"Shape of real data (X_): {X_.shape}")
        #print(f"Shape of noise data (Z_): {Z_.shape}")
        #tf.print("Fake batch data (first 5 samples):", Z_[:5])
        step_g_loss_u, step_g_loss_s, step_g_loss_v = synth.train_generator(X_, Z_, generator_opt)
        
        # Append generator losses
        generator_losses.append(step_g_loss_u)

        # Train embedder (if applicable)
        step_e_loss_t0 = synth.train_embedder(X_, embedder_opt)
    
    X_ = next(synth.get_batch_data(data, n_windows=len(data)))
    Z_ = next(synth.get_batch_noise())
    
    for _ in range(k_d):
        step_d_loss_real, step_d_loss_fake, step_d_loss_fake_e, correct_list = synth.train_discriminator(X_, Z_, discriminator_opt)
        discriminator_losses_real.append(step_d_loss_real)
        discriminator_losses_fake.append(step_d_loss_fake)
    
    correct_real, correct_fake, correct_fake_e = correct_list[0], correct_list[1], correct_list[2]
    print("Correct Real Predictions:", correct_real)
    print("Correct Fake Predictions:", correct_fake)
    print(f"Current k_g: {k_g}, Current k_d: {k_d}")
    print("Current loss in generator:", step_g_loss_u )
    print("Current loss in discriminator for real:", step_d_loss_real )
    print("Current loss in discriminator for fake:", step_d_loss_fake )
    if step % 3 == 0:
        k_g, k_d = adjust_training_ratio(step_g_loss_u,step_d_loss_real, step_d_loss_fake, k_g, k_d,step)

    while len(generator_losses) < len(discriminator_losses_real):
        generator_losses.append(generator_losses[-1])  # Repeat last generator loss

    while len(discriminator_losses_real) < len(generator_losses):
        discriminator_losses_real.append(discriminator_losses_real[-1])
        discriminator_losses_fake.append(discriminator_losses_fake[-1])

    # Save models and losses at intervals or at the end of training
    save_training_ratios(k_d, k_g)
    completed_steps_joint +=1
    if step % saving_interval == 0 or (completed_steps_embedder == train_steps and 
                                       completed_steps_supervisor == train_steps and 
                                       completed_steps_joint == train_steps):
        synth.save_models()
        with open(progress_file, 'w') as file:
            file.write(f"{completed_steps_embedder}\n{completed_steps_supervisor}\n{completed_steps_joint}\n")
        print(f"Models saved at {step}")
        
        # Save losses
        with open(losses_file, 'wb') as file:
            pickle.dump([generator_losses,discriminator_losses_real, discriminator_losses_fake], file)
if completed_steps_embedder == train_steps and completed_steps_supervisor == train_steps and completed_steps_joint == train_steps:
    with open(losses_file, 'rb') as file:
        losses = pickle.load(file)
        plot_losses(losses)

synth.save_generated_signals()

def read_samples_file(file_path):
    samples = []
    with open(file_path, 'r') as file:
        current_sample = []
        for line in file:
        
            line = line.replace('[', '').replace(']', '').strip()
            
            if line:  
                values = list(map(float, line.split()))
                current_sample.extend(values)
            else:
                if current_sample:
                    samples.append(current_sample)
                    current_sample = []  
                    
    if current_sample:
        samples.append(current_sample)

    return samples

def plot_generated_samples(file_path):
    samples = []
    try:
        samples = read_samples_file(file_path)
        # Plotting the samples
        plt.figure(figsize=(12, 6))
        for i, sample in enumerate(samples[:30]):
            plt.figure(figsize=(12, 6))
            plt.plot(sample, label=f'Sample {i+1}')
            
            plt.title(f'Generated Sample {i+1}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()

            # Save each sample's plot to a separate file
            plot_path = f'/work/zf267656/peltfolder/models/plots/promising_plots/generated_sample_{i+1}_{k}_{model_call}_{train_steps}.png'
            plt.savefig(plot_path)
            plt.close()  # Close the figure to free memory

            print(f'Sample {i+1} saved at {plot_path}')
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except ValueError as e:
        print(f"Error converting string to float: {e}")


plot_generated_samples(f'models/samples/generated_samples_{train_steps*k}_{k}_{model_call}.txt')
real_samples = data[0:1000]
synth_samples = synth.generate_samples(n_samples = 1000)
synth_samples = synth_samples.reshape(synth_samples.shape[0], -1)
real_samples = real_samples.reshape(real_samples.shape[0], real_samples.shape[1])  # Reshapes to (1000, 200)
print(real_samples.shape)
print(synth_samples.shape)
combined_data = np.vstack((real_samples, synth_samples))
for index in range (1, 10):
    print("Real sample:", combined_data[index], "Fake sample:", combined_data[2000-index])
combined_labels = np.hstack((np.ones(len(real_samples)), np.zeros(len(synth_samples))))
#scaled_data = scaler.transform(combined_data)
tsne_results = apply_tsne(combined_data)
plot_tsne(tsne_results, f"plots/tsne/events_gen_events_{model_call}_{starting_gen_epoch}.png",labels=combined_labels)

# Function to decide whether each array is closer to 0 or 1 based on its elements
def decide_output_based_on_closeness(predictions):
    final_outputs = []
    
    # Iterate over each array in the predictions
    for array in predictions:
        # Count how many elements are closer to 1 than to 0
        count_closer_to_1 = np.sum(np.abs(array - 1) < np.abs(array - 0))
        
        # Decide the output based on the majority of elements
        if count_closer_to_1 > len(array) / 2:
            final_outputs.append(1)
        else:
            final_outputs.append(0)
    
    return final_outputs

batch_size = 64
num_real_correct_total = 0
num_fake_correct_total = 0
num_real_incorrect_total = 0
num_fake_incorrect_total = 0
total_real = 0
total_fake = 0

# Get the total number of training steps
num_batches = len(test_data) // batch_size

for i in range(num_batches):
    # Step 1: Fetch real batch data using next
    real_batch = next(synth.get_batch_data(test_data, n_windows=len(test_data)))
    tf.print("Real batch data (first 5 samples):", real_batch[:5])

    # Step 2: Generate noise and pass it directly to the generator to create fake samples
    noise_batch = next(synth.get_batch_noise())  # Generate noise
    fake_latent_representations = synth.generator_aux(noise_batch)  # Directly pass noise to generator_aux
    fake_latent_supervised = synth.supervisor(fake_latent_representations)
    
    # Print the generated fake latent representations
    tf.print("Noise batch (first 5 samples):", noise_batch[:5])
    tf.print("Fake latent representations (first 5 samples):", fake_latent_representations[:5])
    tf.print("Fake latent supervised (first 5 samples):", fake_latent_supervised[:5])

    # Step 3: Get latent representations for real samples
    real_latent_representations = synth.embedder(real_batch)
    
    # Print real latent representations
    tf.print("Real latent representations (first 5 samples):", real_latent_representations[:5])
    
    # Step 4: Get predictions from the discriminator for real and fake samples
    real_predictions = synth.discriminator(real_latent_representations)
    fake_predictions = synth.discriminator(fake_latent_supervised)
    
    # Print real and fake predictions
    tf.print("Real predictions (first 5 samples):", real_predictions[:5])
    tf.print("Fake predictions (first 5 samples):", fake_predictions[:5])
    
    # Step 5: Define real and fake labels (real = 1, fake = 0)
    real_labels = np.ones(len(real_batch))
    fake_labels = np.zeros(len(fake_latent_supervised))
    real_preds = decide_output_based_on_closeness(real_predictions)
    fake_preds = decide_output_based_on_closeness(fake_predictions) 
    
    # Print labels and predictions
    print("Real labels (first 5 samples):", real_labels[:5])
    print("Real preds (first 5 samples):", real_preds[:5])
    print("Fake labels (first 5 samples):", fake_labels[:5])
    print("Fake preds (first 5 samples):", fake_preds[:5])
    
    # Step 7: Calculate correct and incorrect predictions
    num_real_correct = np.sum(real_preds == real_labels)
    num_fake_correct = np.sum(fake_preds == fake_labels)
    num_real_incorrect = np.sum(real_preds != real_labels)
    num_fake_incorrect = np.sum(fake_preds != fake_labels)
    
    # Step 8: Accumulate totals for all batches
    num_real_correct_total += num_real_correct
    num_fake_correct_total += num_fake_correct
    num_real_incorrect_total += num_real_incorrect
    num_fake_incorrect_total += num_fake_incorrect

total_real += num_real_correct_total + num_real_incorrect_total
total_fake += num_fake_correct_total + num_fake_incorrect_total

# Step 9: Output the results
print(f"Number of real samples correctly identified as real: {num_real_correct_total}")
print(f"Number of fake samples correctly identified as fake: {num_fake_correct_total}")
print(f"Total number of real samples: {total_real}")
print(f"Total number of fake samples: {total_fake}")
print(f"Number of real samples misclassified as fake: {num_real_incorrect_total}")
print(f"Number of fake samples misclassified as real: {num_fake_incorrect_total}")