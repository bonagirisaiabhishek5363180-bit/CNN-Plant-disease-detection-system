import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
train_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2,horizontal_flip=True,zoom_range=0.2,rotation_range=20)
val_datagen=ImageDataGenerator(rescale=1./255,
                               validation_split=0.2)

train_data=train_datagen.flow_from_directory("dataset",
                                             target_size=(224,224),
                                             batch_size=32,
                                             class_mode='categorical',
                                             subset='training')

validation_data=val_datagen.flow_from_directory("dataset",
                                                  target_size=(224,224),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  subset='validation')
base_model=MobileNetV2(weights='imagenet',include_top=False,input_shape=(224,224,3))
base_model.trainable=True
for layer in base_model.layers[:-30]:
  layer.trainable=False
model=Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128,activation='relu'),
    Dense(train_data.num_classes,activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_data,validation_data=validation_data,epochs=15)
model.save("finetuned_plant_disease_detection_model.keras")

model.save("plant_disease_detection_model.keras")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig("accuracy_plot.png")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig("loss.png")
results = model.evaluate(validation_data)
print("Loss:", results[0])
print("Accuracy:", results[1])
