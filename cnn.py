import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, GlobalAvgPool2D, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt

def get_train_dataset(path, augment=False, split=0.2):
    img_height = 224
    img_width = 224
    image_size = (img_width, img_height)
    batch_size = 32
    # normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    AUTOTUNE = tf.data.AUTOTUNE

    data_dir = pathlib.Path(path)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("Number of image samples: {}".format(image_count))

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    # get class names
    class_names = np.array(sorted([item.name for item in data_dir.glob("*") if item.name != "LICENSE.txt"]))
    valid_size = int(image_count * split)
    train_ds = list_ds.skip(valid_size)
    valid_ds = list_ds.take(valid_size)

    """
    Utility functions grabbed from tf docs
    """
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(tf.cast(one_hot, tf.int64))

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    def configure_for_performance(ds):
        # ds = ds.cache()
        ds = ds.shuffle(1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = train_ds.map(process_path)
    valid_ds = valid_ds.map(process_path)

    # rescale = tf.keras.Sequential([
    #     tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    # ])

    def preprocess(x):
        x = tf.numpy_function(lambda x: tf.keras.preprocessing.image.random_rotation(x, 15, row_axis=0, col_axis=1, channel_axis=2), [x], tf.float32)
        x = tf.numpy_function(lambda x: tf.keras.preprocessing.image.random_zoom(x, (0.8, 1.2), row_axis=0, col_axis=1, channel_axis=2), [x], tf.float32)
        x = tf.numpy_function(lambda x: tf.keras.preprocessing.image.random_shift(x, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2), [x], tf.float32)
        x = tf.numpy_function(lambda x: tf.keras.preprocessing.image.random_shear(x, 0.1, row_axis=0, col_axis=1, channel_axis=2), [x], tf.float32)
        x = tf.image.random_brightness(x, 0.2)
        # x = tf.numpy_function(lambda x: tf.keras.preprocessing.image.random_brightness(x, (0, 0.1)), [x], tf.float32)
        return tf.reshape(x, [img_width, img_height, 3])

    train_ds = train_ds.map(lambda x, y: (x / 255., y))
    valid_ds = valid_ds.map(lambda x, y: (x / 255., y))
    if augment:
        train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
        valid_ds = valid_ds.map(lambda x, y: (preprocess(x), y))

    train_ds = configure_for_performance(train_ds)
    valid_ds = configure_for_performance(valid_ds)

    # if augment:
    #     data_augmentation = tf.keras.Sequential([
    #         # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    #         tf.keras.layers.experimental.preprocessing.RandomRotation(0.05),
    #         tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2)
    #     ])
    #     train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    return train_ds, valid_ds


def get_model(name):
    try:
        print("Loading model...")
        return load_model("models/" + name)
    except:
        print("New model")
        inputs = tf.keras.Input(shape=(224, 224, 3))
        # x = Conv2D(32, 7, strides=2, padding="same", use_bias=False)(inputs)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Conv2D(32, 3, padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Conv2D(64, 3, padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

        x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Conv2D(128, 3, padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

        x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

        x = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAvgPool2D()(x)
        # x = MaxPool2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(300, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(29, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


if __name__ == '__main__':
    name = "cnn64_tfdata"

    # needed this for my setup for some reason, running out of GPU memory maybe -Nut
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_set, valid_set = get_train_dataset('train/', augment=True, split=0.2)
    test_set, _ = get_train_dataset('newtest/', split=0.0)
    model = get_model(name + ".h5")
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=1e-4))

    # for image, label in train_set.take(1):
    #     print("Image shape: ", image.numpy())
    #     print("Label: ", label.numpy())
    # model.summary()
    # for imgs, labels in train_set.take(1):
    #     fig = plt.figure(figsize=(16, 16))
    #     columns = 4
    #     rows = 8
    #     for i, img in enumerate(imgs):
    #         fig.add_subplot(rows, columns, i+1)
    #         plt.imshow(img)
    #     plt.show()

    epochs = 10
    checkpoint_filepath = 'models/cnn64_tfdata5.{epoch:02d}-{val_accuracy:.2f}.h5'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max'
    )
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(train_set, validation_data=valid_set, epochs=epochs, callbacks=[checkpoint_callback, earlystopping_callback])
    perf = model.evaluate(test_set)
    print(perf)
    model.save("models/best_cnn64_tfdata5_{test_accuracy:.2f}.h5".format(test_accuracy = perf[1]))

    # model = tf.keras.models.load_model('models/cnn64_tfdata7.06-0.99.h5')
    # model = tf.keras.models.load_model('models/best_cnn64_tfdata5_0.41.h5')
    # print(model.summary())
    # perf = model.evaluate(valid_set)
    # print(perf)
    # # perf = model.evaluate(test_set)
    # # print(perf)
    # for img, label in test_set:
    #     y_pred = np.argmax(model.predict(img), axis=1)
    #     print(label)
    #     print(y_pred)
    #     break
