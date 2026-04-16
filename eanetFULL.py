"""
EANet (External Attention Network) для классификации цветов tf_flowers.

Данный скрипт объединяет два гайда:
1. Архитектура EANet из eanet.py
2. Датасет tf_flowers из train_eanet_flowers.py

Основные изменения для совместимости:
    - Изменен input_shape с (32, 32, 3) на (224, 224, 3) для цветов
    - Увеличен patch_size до 16 (224/16 = 14x14 патчей) для сохранения структуры
    - Добавлен слой Rescaling(1./255) для нормализации изображений
    - Изменен loss на SparseCategoricalCrossentropy (метки в tf_flowers не one-hot)
    - Упрощен пайплайн данных (убраны сложные повторения из BiT гайда)
    - Удалена BiT модель, оставлена только архитектура EANet

Автоматически загружает датасет tf_flowers при первом запуске.
"""

import os

# Устанавливаем бэкенд TensorFlow для Keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
from keras import ops
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Конфигурация
tfds.disable_progress_bar()
SEEDS = 42
keras.utils.set_random_seed(SEEDS)

# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================
"""
Загружаем датасет tf_flowers:
    - 5 классов цветов (daisy, dandelion, roses, sunflowers, tulips)
    - Разделение: 85% train, 15% validation
    - Автоматическая загрузка при первом запуске
"""
train_ds, validation_ds = tfds.load(
    "tf_flowers",
    split=["train[:85%]", "train[85%:]"],
    as_supervised=True,
)

# Визуализация датасета
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")
plt.show()

# =============================================================================
# ГИПЕРПАРАМЕТРЫ
# =============================================================================
# Параметры датасета
NUM_CLASSES = 5
# ИЗМЕНЕНО: input_shape с (32, 32, 3) на (224, 224, 3) для цветов
input_shape = (224, 224, 3)
RESIZE_TO = 384
CROP_TO = 224
BATCH_SIZE = 64
STEPS_PER_EPOCH = 10
AUTO = tf.data.AUTOTUNE

# Параметры EANet
weight_decay = 0.0001
learning_rate = 0.001
label_smoothing = 0.1
num_epochs = 50
# ИЗМЕНЕНО: patch_size увеличен до 16 (224/16 = 14x14 патчей)
patch_size = 16
num_patches = (input_shape[0] // patch_size) ** 2
embedding_dim = 64
mlp_dim = 64
dim_coefficient = 4
num_heads = 4
attention_dropout = 0.2
projection_dropout = 0.2
num_transformer_blocks = 8

print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2}")
print(f"Patches per image: {num_patches}")

# =============================================================================
# АУГМЕНТАЦИЯ ДАННЫХ
# =============================================================================
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.1),
        layers.RandomContrast(factor=0.1),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)


# =============================================================================
# АРХИТЕКТУРА EANET
# =============================================================================
class PatchExtract(layers.Layer):
    """
    Извлекает патчи из входного изображения.

    Args:
        patch_size: размер квадратного патча
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        B, H, W, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[-1]
        x = ops.image.extract_patches(x, self.patch_size)
        x = ops.reshape(x, (B, -1, self.patch_size * self.patch_size * C))
        return x


class PatchEmbedding(layers.Layer):
    """
    Создает эмбеддинги для патчей с позиционным кодированием.

    Args:
        num_patch: количество патчей
        embed_dim: размерность эмбеддинга
    """

    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = ops.arange(start=0, stop=self.num_patch, step=1)
        return self.proj(patch) + self.pos_embed(pos)


def external_attention(x, dim, num_heads, dim_coefficient=4,
                       attention_dropout=0, projection_dropout=0):
    """
    Блок внешнего внимания (External Attention).

    Args:
        x: входной тензор
        dim: размерность
        num_heads: количество голов внимания
        dim_coefficient: коэффициент размерности
        attention_dropout: dropout для внимания
        projection_dropout: dropout для проекции
    """
    _, num_patch, channel = x.shape
    assert dim % num_heads == 0
    num_heads = num_heads * dim_coefficient

    x = layers.Dense(dim * dim_coefficient)(x)
    x = ops.reshape(x, (-1, num_patch, num_heads, dim * dim_coefficient // num_heads))
    x = ops.transpose(x, axes=[0, 2, 1, 3])
    attn = layers.Dense(dim // dim_coefficient)(x)
    attn = layers.Softmax(axis=2)(attn)
    attn = layers.Lambda(
        lambda attn: ops.divide(
            attn,
            ops.convert_to_tensor(1e-9) + ops.sum(attn, axis=-1, keepdims=True),
        )
    )(attn)
    attn = layers.Dropout(attention_dropout)(attn)
    x = layers.Dense(dim * dim_coefficient // num_heads)(attn)
    x = ops.transpose(x, axes=[0, 2, 1, 3])
    x = ops.reshape(x, [-1, num_patch, dim * dim_coefficient])
    x = layers.Dense(dim)(x)
    x = layers.Dropout(projection_dropout)(x)
    return x


def mlp(x, embedding_dim, mlp_dim, drop_rate=0.2):
    """
    Многослойный перцептрон (MLP) для трансформера.

    Args:
        x: входной тензор
        embedding_dim: размерность эмбеддинга
        mlp_dim: размерность скрытого слоя
        drop_rate: вероятность dropout
    """
    x = layers.Dense(mlp_dim, activation=ops.gelu)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Dropout(drop_rate)(x)
    return x


def transformer_encoder(x, embedding_dim, mlp_dim, num_heads, dim_coefficient,
                        attention_dropout, projection_dropout,
                        attention_type="external_attention"):
    """
    Трансформер энкодер с внешним вниманием.

    Args:
        x: входной тензор
        embedding_dim: размерность эмбеддинга
        mlp_dim: размерность MLP
        num_heads: количество голов внимания
        dim_coefficient: коэффициент размерности
        attention_dropout: dropout для внимания
        projection_dropout: dropout для проекции
        attention_type: тип внимания
    """
    residual_1 = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    if attention_type == "external_attention":
        x = external_attention(
            x, embedding_dim, num_heads, dim_coefficient,
            attention_dropout, projection_dropout,
        )
    x = layers.add([x, residual_1])
    residual_2 = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = mlp(x, embedding_dim, mlp_dim)
    x = layers.add([x, residual_2])
    return x


def get_model():
    """
    Создает модель EANet для классификации цветов.

    Returns:
        keras.Model: скомпилированная модель
    """
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    # ИЗМЕНЕНО: добавлен слой Rescaling для нормализации
    x = layers.Rescaling(1. / 255)(x)
    x = PatchExtract(patch_size)(x)
    x = PatchEmbedding(num_patches, embedding_dim)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x, embedding_dim, mlp_dim, num_heads, dim_coefficient,
            attention_dropout, projection_dropout, "external_attention"
        )

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# =============================================================================
# ПОДГОТОВКА ДАННЫХ
# =============================================================================
# ИЗМЕНЕНО: упрощенный пайплайн данных
def preprocess_train(image, label):
    """Предобработка для тренировочных данных."""
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = tf.image.random_crop(image, size=[CROP_TO, CROP_TO, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label


def preprocess_test(image, label):
    """Предобработка для валидационных данных."""
    image = tf.image.resize(image, (CROP_TO, CROP_TO))
    return image, label


# Создание конвейеров данных
pipeline_train = (train_ds
                  .shuffle(1000)
                  .map(preprocess_train, num_parallel_calls=AUTO)
                  .batch(BATCH_SIZE)
                  .prefetch(AUTO))

pipeline_validation = (validation_ds
                       .map(preprocess_test, num_parallel_calls=AUTO)
                       .batch(BATCH_SIZE)
                       .prefetch(AUTO))

# Визуализация батча
image_batch, label_batch = next(iter(pipeline_train))
plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(image_batch[n].numpy().astype("uint8"))
    plt.title(label_batch[n].numpy())
    plt.axis("off")
plt.show()

# =============================================================================
# СОЗДАНИЕ И КОМПИЛЯЦИЯ МОДЕЛИ
# =============================================================================
model = get_model()
# ИЗМЕНЕНО: loss с CategoricalCrossentropy на SparseCategoricalCrossentropy
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.AdamW(learning_rate=learning_rate,
                                     weight_decay=weight_decay),
    metrics=["accuracy"]
)

model.summary()

# =============================================================================
# ОБУЧЕНИЕ
# =============================================================================
history = model.fit(
    pipeline_train,
    validation_data=pipeline_validation,
    epochs=num_epochs,
)

# =============================================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================
# График потерь
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs")
plt.legend()
plt.grid()
plt.show()

# График точности
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.show()

# =============================================================================
# ОЦЕНКА МОДЕЛИ
# =============================================================================
loss, accuracy = model.evaluate(pipeline_validation)
print(f"Test loss: {round(loss, 2)}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")