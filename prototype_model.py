
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sgdr import SGDRScheduler

import os
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K

train_datagen = ImageDataGenerator(rescale=1./255

                                    # ,rotation_range = 10,
                                    # width_shift_range=0.05,
                                    # height_shift_range=0.05,
                                    # shear_range=0.05,
                                    # zoom_range=0.05,
                                    # horizontal_flip = True
                                    )

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

schedule = SGDRScheduler(min_lr=1e-5,
                         max_lr=1e-3,
                         steps_per_epoch=np.ceil(samples/batch_size),
                         lr_decay=0.9,
                         cycle_length=10,
                         mult_factor=1.5)

print('\nClearing old graphs...')
graph = os.listdir('graph')
for file in graph:
  os.remove('graph/{}'.format(file))

tensorboard = TensorBoard(log_dir='./graph', histogram_freq=0,
                          write_graph=True, write_images=False)


print('\nClearing old checkpoints...')
checkpoints = os.listdir('models/checkpoints')
for file in checkpoints:
  os.remove('models/checkpoints/{}'.format(file))

# best checkpoint
filepath="models/checkpoints/best_weights.hdf5"
best_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# interval checkpoint
filepath="models/checkpoints/weights-interval-{epoch:02d}-{val_acc:.2f}.hdf5"
interval_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period = 100)


print('\nConstructing Model\n\n')

kernel_size  = (2,2)
pool_size = (2,2)
# construct model
K.clear_session()
model = Sequential()
model.add(Conv2D(32, kernel_size = kernel_size, activation = 'relu', input_shape = (28, 28, 1)))
# model.add(MaxPooling2D(pool_size))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size = kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])



model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
    
hist = model.fit_generator(
    train_generator,
    steps_per_epoch = samples/batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps=50,
    callbacks = [schedule, tensorboard, best_checkpoint, interval_checkpoint])
