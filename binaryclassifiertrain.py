from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2   
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)
train_data=train_datagen.flow_from_directory(
    "plant_vs_nonplant_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_data=val_datagen.flow_from_directory(
    "plant_vs_nonplant_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')
base_model=MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3))
model=Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])
base_model.trainable=True
for layer in base_model.layers[:-30]:
    layer.trainable=False
model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history=model.fit(train_data, epochs=5, validation_data=validation_data)
model.save("backend/plant_nonplant_classifier.keras")