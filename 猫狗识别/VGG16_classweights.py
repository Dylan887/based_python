import tensorflow as tf
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

# 加载预训练权重
model = VGG16(num_classes=2)
model.build(input_shape=(None, 224, 224, 3))
model.load_weights('Adogandcat_epoch_20.h5')  # 假设之前训练好的权重
# 冻结部分卷积层，仅微调分类器
for layer in model.features.layers[:-4]:
    layer.trainable = False

# 数据增强（训练集 + 验证集划分）
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2  # 20% 数据作为验证集
)

# 训练数据生成器
train_gen = train_datagen.flow_from_directory(
    directory='data2/猫狗大战/train',  # 包含 cat 和 dog 两类
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'  # 加载训练子集
)

# 验证数据生成器
val_gen = train_datagen.flow_from_directory(
    directory='data2/猫狗大战/train',  # 包含 cat 和 dog 两类
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # 加载验证子集
)

# 类别权重补偿（假设猫为 0，狗为 1）
class_weights = {
    0: 4.5,  # 猫的权重，假设在训练集中较少
    1: 1.0   # 狗的权重，假设在训练集中较多
}

# 编译模型
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size,
    epochs=20,
    class_weight=class_weights  # 应用类别权重
)

model.save_weights('vgg16_dog_cat_classifier_with_split')
