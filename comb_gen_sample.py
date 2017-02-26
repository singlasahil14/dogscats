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

max_epochs = 50
best_lr = LRConfig(initial=3e-4, decay=6.25e-4)
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
model.compile(optimizer=Adam(lr=best_lr.initial, decay=best_lr.decay),
              loss='binary_crossentropy',
              metrics=['accuracy'])
initial_weights = model.get_weights()

def conv1(config, max_epochs=40):
    gen_t = config.datagen()
    batches = gen_t.flow(trn_data, trn_labels, batch_size=batch_size)
    hist = model.fit_generator(batches, nb_sample, nb_epoch=max_epochs, 
                               validation_data=(val_data, val_labels), 
                               max_q_size=100)
    model.set_weights(initial_weights)
    return hist.history

def log_string(config, best_epoch, loss):
    return 'Config: {}\nBest epoch: {}, Best validation loss: {}\n'.format(
           config, best_epoch, loss)

def get_values(points, range_dict, key):
    points = np.asarray(points)
    if key not in range_dict:
        raise Valueerror('Key: {} not in dictionary'.format(key))
    if(type(range_dict[key]) is tuple):
        start = range_dict[key][0]
        stop = range_dict[key][1]
    else:
        start = 0
        stop = range_dict[key]
    if(key=='channel_shift_range' or key=='rotation_range'):
        values = start + np.int32(np.floor(points*(stop-start+1)))
    else:
        values = start + points*(stop-start)
    return values

def hyperparams_search(range_dict, get_loss_history, 
                       init_config={}, max_epochs=50):
    import time
    f1 = open('comb_all_configs', 'w+')

    configs_losses = []
    print('Config: {}'.format(init_config))
    history = get_loss_history(DatagenConfig(init_config), max_epochs)
    val_losses = np.asarray(history['val_loss'])
    best_epoch = np.argmin(val_losses)
    min_loss = val_losses[best_epoch]
    configs_losses.append((dict(init_config), best_epoch, min_loss))

    log_str = log_string(init_config, best_epoch, min_loss)
    print(log_str)
    f1.write(log_str+'\n')
    f1.flush()
    for key, values in range_dict.iteritems():
        print('\nKey: {}\nValues: {}\n'.format(key, values))
        config = dict(init_config)
        for i, value in enumerate(values):
            config.update({key: value})
            copy_config = dict(config)
            print('Key: {}\nValue: {}'.format(key, value, config))
            start_time = time.time()
            history = get_loss_history(DatagenConfig(config), max_epochs)
            time_elapsed = time.time() - start_time
            print('\nEpoch: {}, Time Elapsed: {}'.format(i, time_elapsed))
 
            val_losses = np.asarray(history['val_loss'])
            loss_plt = sns.tsplot(time="epoch", value="val loss",
                            data=val_losses)

            best_epoch = np.argmin(val_losses)
            min_loss = val_losses[best_epoch]
            configs_losses.append((copy_config, best_epoch, min_loss))
 
            log_str = log_string(config, best_epoch, min_loss)
            print(log_str)
            f1.write(log_str+'\n')
            f1.flush()
        loss_plt.savefig('val_losses')
    f1.close()

    configs_losses = sorted(configs_losses, key=lambda cfg: cfg[2])
    f2 = open('comb_best_configs', 'w+')
    for run_tuple in configs_losses:
        config = run_tuple[0]
        best_epoch = run_tuple[1]
        min_loss = run_tuple[2]
        log_str = log_string(config, best_epoch, min_loss)
        f2.write(log_str)
    f2.close()

#range_dict = {'width_shift_range': [0.05, 0.10, 0.15, 0.20, 0.25],
#              'height_shift_range': [0.05, 0.10, 0.15, 0.20, 0.25],
#              'rotation_range': [9, 18, 27, 36, 45],
#              'shear_range': [0.05, 0.10, 0.15, 0.20, 0.25],
#              'zoom_range': [0.05, 0.10, 0.15, 0.20, 0.25],
#              'channel_shift_range': [8, 16, 24, 32, 40]}
#hyperparams_search(range_dict, conv1)

#range_dict = {'zoom_range': [0.1, 0.2, 0.3, 0.4, 0.5]}
#hyperparams_search(range_dict, conv1, init_config={'width_shift_range': 0.3})

range_dict = {'zoom_range': [0.1, 0.2]}
hyperparams_search(range_dict, conv1, 
                   init_config={'width_shift_range': 0.3},
                   max_epochs=3)
