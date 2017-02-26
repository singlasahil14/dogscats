from utils import *

data_path ='data/'
orig_path = 'data/original/'
sample_path = 'data/sample/'

# Create dirs
from shutil import rmtree
if os.path.exists(sample_path):
    rmtree(sample_path)
os.makedirs(sample_path)
os.chdir(orig_path+'train/')
for d in glob('*'):
    train_dir = '../../sample/train/'+d
    valid_dir = '../../sample/valid/'+d
    if os.path.exists(train_dir):
        rmtree(train_dir)
    os.makedirs(train_dir)
    if os.path.exists(valid_dir):
        rmtree(valid_dir)
    os.makedirs(valid_dir)
os.chdir('../../../')

# Get filenames and class ids
gen = image.ImageDataGenerator()
datagen = gen.flow_from_directory(orig_path+'train/')
X = np.asarray(datagen.filenames)
y = np.asarray(datagen.classes)

# Get training and validation split
from sklearn.model_selection import train_test_split
train_files, val_files = train_test_split(X, test_size=1000, train_size=2000, stratify=y)

# Copy files
from shutil import copyfile
for file_name in train_files: 
    copyfile(orig_path + 'train/' + file_name, sample_path + 'train/' + file_name)
for file_name in val_files:
    copyfile(orig_path + 'train/' + file_name, sample_path + 'valid/' + file_name)

# Create test filenames
test_prefix = 'test/unknown/'
orig_test_path = orig_path + 'test/'
test_files = np.asarray([f for f in os.listdir(orig_test_path) if os.path.isfile(os.path.join(orig_test_path, f))])
numTest = 1000
test_files = np.random.choice(test_files, numTest, replace=False)

# Create test folder
sample_test_path = sample_path + test_prefix
os.makedirs(sample_test_path)

# Copy test files
for file_name in test_files:
    copyfile(orig_test_path + file_name, sample_test_path + file_name)
