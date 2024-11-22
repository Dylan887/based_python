import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 定义 VGG16 模型
class VGG16(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        self.features = models.Sequential([
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),

            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),

            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),

            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),

            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),
        ])
        self.classifier = models.Sequential([
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ])

    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 使用 ImageDataGenerator 加载并预处理数据集
data_dir = 'data'
input_shape = (224, 224)
batch_size = 4

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    directory=f'{data_dir}/train',
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    directory=f'{data_dir}/validation',
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='binary'
)

# 初始化模型、优化器和损失函数
model = VGG16(num_classes=2)
# 构建模型结构（明确指定输入形状）
model.build(input_shape=(None, 224, 224, 3))  # None 表示动态批次大小

# 查看模型结构
model.summary()
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练循环
epochs = 20
steps_per_epoch = train_gen.samples // batch_size
validation_steps = val_gen.samples // batch_size

for epoch in range(epochs):
    print(f"=========== Epoch {epoch + 1} ==============")
    history = model.fit(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_gen,
                        validation_steps=validation_steps,
                        epochs=1)
    train_loss = history.history['loss'][0]
    val_loss = history.history['val_loss'][0]
    val_accuracy = history.history['val_accuracy'][0]
    print(f"训练集上的损失：{train_loss}")
    print(f"验证集上的损失：{val_loss}")
    print(f"验证集上的精度：{val_accuracy:.1%}")
    
    # 保存模型
    model.save_weights(f"Adogandcat_epoch_{epoch + 1}.h5")
    print("模型已保存。")


