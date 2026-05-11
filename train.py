import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 validation_split=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 rotation_range=20)
train_data=train_datagen.flow_from_directory("dataset",target_size=(224,224),batch_size=8,class_mode='categorical',subset='training')
validation_data=train_datagen.flow_from_directory("dataset",target_size=(224,224),batch_size=8,class_mode='categorical',subset='validation')
base_model=MobileNetV2(weights='imagenet',include_top=False,input_shape=(224,224,3))
base_model.trainable=False
model=Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128,activation='relu'),
    Dense(train_data.num_classes,activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data,validation_data=validation_data,epochs=10)
model.save("plant_disease_detection_model.h5")


