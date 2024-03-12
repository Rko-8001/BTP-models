import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Constants
IMAGE_SIZE = 224  # Updated image size
BATCH_SIZE = 64
CHANNELS = 3
EPOCHS = 10

# Load dataset
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "train",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'  # One-hot encoding for multiple classes
)

val_generator = val_datagen.flow_from_directory(
    "test",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Get the number of classes from the training data
n_classes = len(train_generator.classes)

# Load the VGG16 model pre-trained on ImageNet dataset
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

# Freeze the pre-trained layers (optional)
base_model.trainable = False  # Experiment with freezing/unfreezing layers

# Add new layers on top of the pre-trained model
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(2, activation='softmax')(x)  # Update to 2 units
predictions = x


# Create the final model
model = models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Adjust if needed based on dataset size
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator)  # Adjust if needed based on dataset size
)

# Evaluate the model (optional)
scores = model.evaluate(val_generator)
print("Validation Loss:", scores[0])
print("Validation Accuracy:", scores[1])

# Plot training history (optional)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Prediction function (optional, you can adapt this from your previous code)
def predict(model, img):
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  predictions = model.predict(img_array)
  predicted_class = np.argmax(predictions[0])
  return predicted_class

# Save the model (optional)
model.save("models/fruits_transfer_learning.h5")



# import tensorflow as tf
# from tensorflow.keras import models, layers
# import matplotlib.pyplot as plt 
# import numpy as np 

# # Constants
# IMAGE_SIZE = 224  # Updated image size
# BATCH_SIZE = 32
# CHANNELS = 3
# EPOCHS = 30

# # Load dataset
# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     "train",
#     seed=123,
#     shuffle=True,
#     image_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE
# )
# class_names = dataset.class_names

# # Preprocess dataset
# def get_dataset(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
#     assert(train_split + test_split + val_split) == 1
    
#     ds_size = len(ds)
#     print(ds_size)
#     if shuffle:
#         ds = ds.shuffle(shuffle_size, seed=12)
    
#     train_size = int(train_split * ds_size)
#     val_size = int(val_split * ds_size)
    
#     train_ds = ds.take(train_size)
#     val_ds = ds.skip(train_size).take(val_size)
#     test_ds = ds.skip(train_size).skip(val_size)
    
#     return train_ds, val_ds, test_ds

# train_ds, val_ds, test_ds = get_dataset(dataset)

# # Data preprocessing and augmentation
# resize_and_rescale = tf.keras.Sequential([
#   layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
#   layers.experimental.preprocessing.Rescaling(1./255),
# ])
# data_augmentation = tf.keras.Sequential([
#   layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#   layers.experimental.preprocessing.RandomRotation(0.2),
# ])

# # Prepare datasets
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# # Apply data augmentation
# train_ds = train_ds.map(
#     lambda x, y: (data_augmentation(x, training=True), y)
# ).prefetch(buffer_size=tf.data.AUTOTUNE)



# input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
# n_classes = 2  # Since there are two classes: fresh and rotten

# model = models.Sequential([
#     layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(n_classes, activation='softmax'),
# ])


# model.build(input_shape=input_shape)
# model.summary()

# # Compile the model
# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=['accuracy']
# )

# # Train the model
# history = model.fit(
#     train_ds,
#     batch_size=BATCH_SIZE,
#     validation_data=val_ds,
#     verbose=1,
#     epochs=EPOCHS,
# )

# # Evaluate the model
# scores = model.evaluate(test_ds)

# # Plot training history
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(range(EPOCHS), acc, label='Training Accuracy')
# plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(range(EPOCHS), loss, label='Training Loss')
# plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# # Prediction function
# def predict(model, img):
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)

#     predictions = model.predict(img_array)
#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * np.max(predictions[0]), 2)
#     return predicted_class, confidence

# # Predict on test dataset
# plt.figure(figsize=(15, 15))
# for images, labels in test_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
        
#         predicted_class, confidence = predict(model, images[i])
#         actual_class = class_names[labels[i]] 
        
#         plt.title(f"Actual: {actual_class}, Predicted: {predicted_class}. Confidence: {confidence}%")
        
#         plt.axis("off")

# # Save the model
# import os
# # model_version = max([int(i) for i in os.listdir("models") + [0]]) + 1
# # model.save(f"models/{model_version}")
# model.save("models/upgraded_DS.h5")
# # model.save("models/fruits.h5")
