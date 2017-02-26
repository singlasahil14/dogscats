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
best_lr = LRConfig(initial=0.00254829674798, decay=0.0123456790123)
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

def datagenIterator(range_dict, num_configs=10, best_config={}):
    best_val_loss = np.float('inf')
    sequencer = ghalton.Halton(len(range_dict))
    config = dict(best_config)
    for key, max_value in range_dict.iteritems():
        points = np.sort(np.asarray(sequencer.get(num_configs)))
        if(key=='channel_shift_range' or key=='rotation_range'):
            values = np.int32(np.floor(points*(max_value+1)))
        else:
            values = points*max_value
        for value in values:
            config.update({key:value})
            yield config

def log_string(config, best_epoch, loss):
    return 'Config: {}\nBest epoch: {}, Best validation loss: {}\n\n'.format(
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

def hyperparams_beamsearch(range_dict, get_loss_history, beam_size=1, 
                           num_configs=5, max_epochs=50):
    import time
    f1 = open('all_gen_configs', 'w+')

    sequencer = ghalton.Halton(1)
    prev_configs = [{}]
    prev_losses = [float('inf')]
    configs_losses = []
    
    print('\n\n\n')
    for key, max_value in range_dict.iteritems():
        points = [(i+1)/float(num_configs) for i in range(num_configs)]
        #points = np.sort(np.squeeze(np.asarray(sequencer.get(num_configs))))
        values = get_values(points, range_dict, key)
        print('Prev configs: {}\nPrev losses: {}'.format(
              prev_configs, prev_losses))
        next_losses = []
        next_configs = []
        for prev_config, prev_loss in zip(prev_configs, prev_losses):
            config = dict(prev_config)
            config.update({key:0})

            best_loss = min(prev_losses)
            next_losses.append(prev_loss)
            next_configs.append(dict(config))
            print('\n\n\nKey: {}\nValues: {}\n\n'.format(key, values.tolist()))
            for i, value in enumerate(values):
                config.update({key: value})
                copy_config = dict(config)
                print('\nPrevious Config: {}'.format(prev_config))
                print('Previous validation loss: {}'.format(prev_loss))
                print('Key: {}\nValue: {}\nCurrent config: {}'.format(
                      key, value, config))
                start_time = time.time()
                history = get_loss_history(DatagenConfig(config), max_epochs)
                time_elapsed = time.time() - start_time
                print('\nEpoch: {}, Time Elapsed: {}'.format(i, time_elapsed))

                val_losses = np.asarray(history['val_loss'])
                best_epoch = np.argmin(val_losses)
                min_loss = val_losses[best_epoch]
                next_losses.append(min_loss)
                next_configs.append(copy_config)
                configs_losses.append((copy_config, best_epoch, min_loss))

                log_str = log_string(config, best_epoch, min_loss)
                print('Previous validation loss: {}'.format(prev_loss))
                print(log_str)
                f1.write(log_str)
                f1.flush()

                if(min_loss>prev_loss):
                    break
        print('\n\nNext configs: {}\nNext losses: {}'.format(
              next_configs, next_losses))
        next_losses = np.asarray(next_losses)
        indices_sort = next_losses.argsort()
        next_losses = next_losses.tolist()
        indices_sort = indices_sort.tolist()
        prev_losses = [next_losses[index] 
                       for index in indices_sort[:beam_size]]
        prev_configs = [next_configs[index] 
                        for index in indices_sort[:beam_size]]

    f1.close()

    configs_losses = sorted(configs_losses, key=lambda cfg: cfg[2])
    f2 = open('best_gen_configs', 'w+')
    for run_tuple in configs_losses:
        config = run_tuple[0]
        best_epoch = run_tuple[1]
        min_loss = run_tuple[2]
        log_str = log_string(config, best_epoch, min_loss)
        f2.write(log_str)
    f2.close()

range_dict = {'width_shift_range': 0.5, 'height_shift_range': 0.5, 
              'rotation_range': 45, 'shear_range': 0.5, 
              'zoom_range': 0.5, 'channel_shift_range': 40}
hyperparams_beamsearch(range_dict, conv1)

#range_dict = {'width_shift_range': 0.5, 'height_shift_range': 0.5}
#hyperparams_beamsearch(range_dict, conv1, beam_size=2, 
#                       num_configs=2, max_epochs=2)
