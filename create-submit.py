from utils import *

#Define paths
path = 'data/'
#path = 'data/sample/'
model_path = path + 'models/'

batch_size=128

feat_path = path+'results/vgg_precomp.h5'
test_names = HDF5Matrix(feat_path, 'test_names').data[:]
test_names = [test_name.split('.')[0] for test_name in test_names]
conv_test_feat = HDF5Matrix(feat_path, 'test_features')

bn_model = load_model(model_path+'model.0.0619.hdf5')

names = np.expand_dims(test_names, axis=1)
preds = do_clip(bn_model.predict(conv_test_feat, batch_size=batch_size), 0.95)
print(preds.shape)
preds = preds[:,1:]
print(preds.shape)

submission = np.hstack((names, preds))
np.savetxt('submit.out', submission, delimiter=',', fmt='%s', header='id,label', comments='')
