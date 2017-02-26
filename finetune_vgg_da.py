from utils import *

#Define paths
path = 'data/'
#path = 'data/sample/'
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

#Define model
def get_bn_layers(p, num=128):
    return [
        Flatten(input_shape=(512,7,7)),
        Dense(num, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(num, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(1, activation='sigmoid')
        ]

def gen_batches(feat, labels, batch_size=32, epoch_size=None):
    if epoch_size is None:
        epoch_size = len(labels)
    start = 0
    while True:
        epoch_start = start % epoch_size
        curr_size = min(batch_size, epoch_size - epoch_start)
        stop = min(start + curr_size, len(labels))
        yield feat[start:stop], labels[start:stop][:,1:]
        start = stop % (len(labels))

p = 0.5
num_hidden = 4096
batch_size=64
num_epochs = 2

file_path = path+'results/vgg_precomp.h5'
conv_trn_feat = HDF5Matrix(file_path, 'train_features')
trn_labels = HDF5Matrix(file_path, 'train_labels')
conv_val_feat = HDF5Matrix(file_path, 'valid_features').data[:]
val_labels = HDF5Matrix(file_path, 'valid_labels').data[:][:,1:]

num_samples = len(conv_trn_feat)/5
trn_datagen = gen_batches(conv_trn_feat, trn_labels, batch_size, epoch_size=num_samples)
checkpointer = ModelCheckpoint(filepath=model_path+'model.{val_loss:.4f}.hdf5', 
                               verbose=0, save_best_only=True)
bn_model = Sequential(get_bn_layers(p, num=num_hidden))
bn_model.compile(Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
hist = bn_model.fit_generator(trn_datagen, samples_per_epoch=num_samples, nb_epoch=4, 
             validation_data=(conv_val_feat, val_labels), callbacks=[checkpointer], 
             max_q_size=100)
print(min(hist.history['val_loss']))
bn_model.optimizer.lr = 3e-5
bn_model.fit_generator(trn_datagen, samples_per_epoch=num_samples, nb_epoch=4, 
             validation_data=(conv_val_feat, val_labels), callbacks=[checkpointer], 
             max_q_size=100)
bn_model.optimizer.lr = 3e-6
bn_model.fit_generator(trn_datagen, samples_per_epoch=num_samples, nb_epoch=4, 
             validation_data=(conv_val_feat, val_labels), callbacks=[checkpointer], 
             max_q_size=100)
