from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint
from sgdr import SGDRScheduler

import os
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K


print('\nLoading Model\n\n')



# load json and create model
json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/checkpoints/best_weights.hdf5")
print("\nLoaded model from disk\n")

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
print('\nCompiled model\n')

print('\nClearing old graphs...')
graph = os.listdir('graph')
for file in graph:
  os.remove('graph/{}'.format(file))


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_dir = "./train"
validation_dir = "./test"

num_classes = len(os.listdir('./train'))
batch_size=32

print('\n\nProcessing training data...')
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size = (28,28),
    batch_size=batch_size,
    class_mode= 'categorical',
    color_mode = 'grayscale')

print('\nProcessing validation data...')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (28,28),
    batch_size=batch_size,
    class_mode = 'categorical',
    color_mode = 'grayscale')


# callbacks
epochs = 30000
samples = 45000

schedule = SGDRScheduler(min_lr=1e-4,
                         max_lr=1e-3,
                         steps_per_epoch=np.ceil(samples/batch_size),
                         lr_decay=0.9,
                         cycle_length=20,
                         mult_factor=1.5)


tensorboard = TensorBoard(log_dir='./graph', histogram_freq=0,
                          write_graph=True, write_images=False)

# best checkpoint
filepath="models/checkpoints/best_weights.hdf5"
best_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# interval checkpoint
filepath="models/checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
interval_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period = 100)



hist = model.fit_generator(
    train_generator,
    steps_per_epoch = samples/batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps=50,
    callbacks = [schedule, tensorboard, best_checkpoint, interval_checkpoint])
