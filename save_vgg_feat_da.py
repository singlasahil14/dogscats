from utils import *

#Define path
path = "data/"
#path = "data/sample/"
num_epochs = 5
results_path = path + 'results/'

input_shape = (224,224)

#Define conv model
from keras.applications.vgg16 import VGG16
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(3,)+input_shape)
#vgg16 = Model(vgg16.input, vgg16.layers[-2].output)

batch_size = 64
gen_t = image.ImageDataGenerator(rotation_range=12, width_shift_range=0.1, 
                                 height_shift_range=0.025, shear_range=0.05, 
                                 zoom_range=0.1, channel_shift_range=10, 
                                 fill_mode='constant', horizontal_flip=True)
gen = image.ImageDataGenerator()
train_datagen = gen_t.flow_from_directory(path+'train/', target_size=input_shape,
                                          batch_size=batch_size)
valid_datagen = gen.flow_from_directory(path+'valid/', target_size=input_shape, 
                                        batch_size=batch_size)
test_datagen = gen.flow_from_directory(path+'test/', target_size=input_shape, 
                                       batch_size=batch_size, shuffle=False)

s = FeatureSaver(train_datagen=train_datagen, 
                 valid_datagen=valid_datagen,
                 test_datagen=test_datagen)
f = h5py.File(results_path+'vgg_precomp.h5', 'w')
s.save_train(vgg16, f, num_epochs=num_epochs)
s.save_valid(vgg16, f)
s.save_test(vgg16, f)

shutil.copy(results_path+'vgg_precomp.h5', results_path+'vgg_precomp_cp.h5')

print(f['train_features'].shape)
print(f['train_labels'].shape)

print(f['valid_features'].shape)
print(f['valid_labels'].shape)

print(f['test_features'].shape)
print(f['test_names'].shape)
