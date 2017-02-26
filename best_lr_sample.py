from utils import *
import ghalton

#Define path
path = "data/sample/"
results_path = path + 'results/'

batch_size = 64
input_shape = (224,224)
gen = image.ImageDataGenerator()
trn_datagen = gen.flow_from_directory(path+'train/', target_size=input_shape, 
                                      batch_size=batch_size)
trn_tuples = zip(*[batches for batches in get_batches(trn_datagen)])
trn_data = np.concatenate(trn_tuples[0])
print(trn_data.shape)
trn_labels = np.concatenate(trn_tuples[1])[:,1:]
print(trn_labels.shape)
val_datagen = gen.flow_from_directory(path+'valid/', target_size=input_shape, 
                                      batch_size=2*batch_size)
val_tuples = zip(*[batches for batches in get_batches(val_datagen)])
val_data = np.concatenate(val_tuples[0])
print(val_data.shape)
val_labels = np.concatenate(val_tuples[1])[:,1:]
print(val_labels.shape)
nb_sample = trn_data.shape[0]
nb_val_sample = val_data.shape[0]

gen_t = image.ImageDataGenerator(rotation_range=12, width_shift_range=0.1, 
                                 height_shift_range=0.025, shear_range=0.05, 
                                 zoom_range=0.1, channel_shift_range=10, 
                                 fill_mode='constant', horizontal_flip=True)
gen = image.ImageDataGenerator()
batches = gen_t.flow(trn_data, trn_labels, batch_size=batch_size)

max_epochs = 60
model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Convolution2D(32,3,3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3,3)),
        Convolution2D(64,3,3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3,3)),
        Flatten(),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
model.compile(optimizer=Adam(), loss='binary_crossentropy', 
              metrics=['accuracy'])
initial_weights = model.get_weights()

def conv1(config):
    model.optimizer.lr = config.initial
    model.optimizer.decay = config.decay
    hist = model.fit_generator(batches, nb_sample, nb_epoch=max_epochs, 
                               validation_data=(val_data, val_labels), 
                               max_q_size=100)
    model.set_weights(initial_weights)
    return hist.history

class LRConfig:
    def __init__(self, initial=1e-3, decay=0.0):
        self.initial = float(initial)
        self.decay = float(decay)

    def __str__(self):
        return 'initial: {}, decay: {}'.format(self.initial, self.decay)

def lrIterator(num_configs=50, initial_range=(-4,-1), decay_range=(0,0.01)):
    sequencer = ghalton.Halton(2)
    all_points = np.asarray(sequencer.get(num_configs))

    initial_start = initial_range[0]
    initial_stop = initial_range[1]
    decay_start = decay_range[0]
    decay_stop = decay_range[1]
    for points in all_points:
        initial = np.power(10, initial_start + 
                          (points[0]*(initial_stop-initial_start)))
        decay = decay_start + (points[1]*(decay_stop-decay_start))
        config = LRConfig(initial=initial, decay=decay)
        yield config

def loss_history(config):
    return conv1(config)

def log_string(config, best_epoch, loss):
    return 'Best config: {}\nBest epoch: {}, Best validation loss: {}\n\n'.format(config, best_epoch, loss)

def best_hyperparams(configIter, get_loss_history):
    import time
    f1 = open('all_lr_configs', 'w+')
    f2 = open('best_lr_configs', 'w+')

    best_loss = float('inf')
    all_configs = configIter()
    for i, config in enumerate(all_configs):
        print(config)
        start_time = time.time()
        history = get_loss_history(config)
        time_elapsed = time.time() - start_time
        print('Epoch: {}, Time Elapsed: {}\n'.format(i, time_elapsed))

        val_losses = np.asarray(history['val_loss'])
        best_epoch = np.argmin(val_losses)
        curr_loss = val_losses[best_epoch]
        log_str = log_string(config, best_epoch, curr_loss)
        print(log_str)
        f1.write(log_str)
        f1.flush()
        if(curr_loss<best_loss):
            f2.write(log_str)
            f2.flush()
            best_loss = curr_loss
    f1.close()
    f2.close()

best_hyperparams(lrIterator, loss_history)
