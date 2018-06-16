from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


# paths
photo_dir = '/home/student21m15/labs/lab3/optional/'
train_dir = photo_dir + 'train'
val_dir = photo_dir + 'val'

# samples size
samples_size = 1000

# validation image count
validation_samples_size = 300

# image sizes
img_width = 128
img_height = 128
img_depth = 3

# input image shape
input_shape = (img_width, img_height, img_depth)

# images in batch
batch_size = 64

# types of images
nb_classes = 3

# epoches count
nb_epoch = 1000

# convolutional layer core
conv_core = 3
# default dropout
def_dropout = 0.2
# default pool size
def_pool_size = 2


# Sequential model
model = Sequential()

# First cascade
# convolutional layer
model.add(Conv2D(32, (conv_core, conv_core), padding='same',
                      input_shape=(img_width, img_height, img_depth), activation='relu'))
# convolutional layer
model.add(Conv2D(32, (conv_core, conv_core), padding='same',
                      input_shape=(img_width, img_height, img_depth), activation='relu'))
# pooling layer
model.add(MaxPooling2D(pool_size=(def_pool_size, def_pool_size)))
# batch normalization
model.add(BatchNormalization())
# dropout pooling
model.add(Dropout(def_dropout))



# Second cascade
# convolutional layer
model.add(Conv2D(32, (conv_core, conv_core), padding='same',
                      input_shape=(img_width, img_height, img_depth), activation='relu'))
# pooling layer
model.add(MaxPooling2D(pool_size=(def_pool_size, def_pool_size)))
# batch normalization
model.add(BatchNormalization())
# dropout pooling
model.add(Dropout(def_dropout))


# Third cascade
model.add(Conv2D(64, (conv_core, conv_core), padding='same', activation='relu'))
# pooling layer
model.add(MaxPooling2D(pool_size=(def_pool_size, def_pool_size)))
# batch normalization
model.add(BatchNormalization())
# dropout pooling
model.add(Dropout(def_dropout))



# Fourth cascade
model.add(Conv2D(64, (conv_core, conv_core), padding='same', activation='relu'))
# pooling layer
model.add(MaxPooling2D(pool_size=(def_pool_size, def_pool_size)))
# batch normalization
model.add(BatchNormalization())
# dropout pooling
model.add(Dropout(def_dropout))



# flatten the input
model.add(Flatten())
# Fully connected layer
model.add(Dense(1024, activation='relu'))
# ropout pooling
model.add(Dropout(0.5))
# Output layer
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

aug_gen = ImageDataGenerator(rescale=1./255,
                             rotation_range=15,
                             width_shift_range=0.2,
                             height_shift_range=0.15,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')


datagen = ImageDataGenerator(rescale=1./255)

# Train generator
train_generator = aug_gen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Validation generator
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# train model
model.fit_generator(
    train_generator,
    steps_per_epoch=samples_size // batch_size,
    epochs=nb_epoch,
    validation_data=val_generator,
    validation_steps=validation_samples_size // batch_size)

model.save('khmelevsky.h5')
