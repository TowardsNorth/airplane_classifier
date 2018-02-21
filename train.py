from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height = 256, 256
train_data_dir = "/Users/chase/Development/DeepLearning/airplane_classifier/images/train"
test_data_dir = "/Users/chase/Development/DeepLearning/airplane_classifier/images/test"
nb_train_samples = 998
nb_test_samples = 246
learning_rate = 0.001
batch_size = 16
epochs = 10

# We want to use the VGG19 model with weights already calculated by imagenet
model = applications.VGG19(weights = "imagenet", include_top = False, input_shape = (img_width, img_height, 3)) 

# Freeze the first 5 layers we dont want to train
for layer in model.layers:
  layer.trainable = False

# Adding custom layers
# Output of the model so far
X = model.output
X = Flatten()(X)
X = Dense(1024, activation = "relu")(X)
X = Dropout(0.5)(X)
X = Dense(1024, activation = 'relu')(X)
predictions = Dense(2, activation = 'softmax')(X)

# Create final model
model_final = Model(input = model.input, output = predictions)

# Compile
model_final.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr = learning_rate), metrics = ['accuracy'])

# Generate our data with automatic data augmentation
train_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range = 0.3,
  rotation_range = 30)

test_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range = 0.3,
  rotation_range = 30)

# Tell Keras where the data is...
train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size = (img_height, img_width),
  batch_size = batch_size,
  class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size = (img_height, img_width),
  class_mode = 'categorical')

# Save the model according to the conditions
checkpoint = ModelCheckpoint('vgg19_1.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)
early = EarlyStopping(monitor = 'val_acc', min_delta = 0, patience = 10, verbose = 1, mode = 'auto')

# Train!
model_final.fit_generator(
  train_generator,
  samples_per_epoch = nb_train_samples,
  epochs = epochs,
  validation_data = test_generator,
  nb_val_samples = nb_test_samples,
  callbacks = [checkpoint, early])