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
batch_size = 32

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

# 加载预训练的 VGG16 模型权重
vgg16_base = VGG16(num_classes=2)
vgg16_base(np.zeros((1, 224, 224, 3)))  # 先调用模型创建变量
vgg16_base.load_weights('Adogandcat_epoch_20.h5')  # 假设已经训练并保存了第 20 轮的模型权重

# 冻结 VGG16 的卷积层，只训练分类器部分
for layer in vgg16_base.features.layers:
    layer.trainable = False

# 定义新的 CNN 模型，并使用 VGG16 作为特征提取器
class CustomCNN(tf.keras.Model):
    def __init__(self, vgg16_base, num_classes=2):
        super(CustomCNN, self).__init__()
        self.vgg16_base = vgg16_base.features  # 使用 VGG16 的特征提取部分
        self.new_classifier = models.Sequential([
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ])

    def call(self, x):
        x = self.vgg16_base(x)
        x = self.new_classifier(x)
        return x

# 初始化新的 CNN 模型、优化器和损失函数
model = CustomCNN(vgg16_base=vgg16_base, num_classes=2)

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练循环
epochs = 10
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
    model.save_weights(f"CustomCNN_epoch_{epoch + 1}.h5")
    print("模型已保存。")




#预测代码
# 加载训练好的 CustomCNN 模型
class CustomCNN(tf.keras.Model):
    def __init__(self, vgg16_base, num_classes=2):
        super(CustomCNN, self).__init__()
        self.vgg16_base = vgg16_base.features  # 使用 VGG16 的特征提取部分
        self.new_classifier = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])

    def call(self, x):
        x = self.vgg16_base(x)
        x = self.new_classifier(x)
        return x

# 定义和加载 CustomCNN
vgg16_base = VGG16(num_classes=2)  # 使用之前定义的 VGG16 类
vgg16_base.build(input_shape=(None, 224, 224, 3))  # 构建 VGG16 基础
vgg16_base.load_weights('Adogandcat_epoch_20.h5')  # 加载预训练的 VGG16 权重

custom_cnn = CustomCNN(vgg16_base=vgg16_base, num_classes=2)
custom_cnn.build(input_shape=(None, 224, 224, 3))
custom_cnn.load_weights('CustomCNN_epoch_10.h5')  # 替换为训练好的 CustomCNN 权重路径


# 加载和预处理图像
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # 加载图像并调整大小
    img_array = img_to_array(img)  # 转换为 NumPy 数组
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
    img_array = preprocess_input(img_array)  # VGG16 所需的标准化
    return img, img_array

# 预测和显示图像
def predict_and_display(image_path, model, model_name):
    # 加载图像
    original_img, processed_img = load_and_preprocess_image(image_path)

    # 预测类别
    predictions = model(processed_img, training=False)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    # 显示结果
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.axis('off')
    plt.title(f"Model: {model_name}\nPredicted Class: {predicted_class}\nConfidence: {confidence:.2f}")
    plt.show()

# 测试图像路径
image_path = 'data/test/1.jpg'  # 替换为实际图像路径

# 使用 CustomCNN 预测
predict_and_display(image_path, custom_cnn, "Custom CNN")

