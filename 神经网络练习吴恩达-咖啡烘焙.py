import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
X = np.array([
    [200, 17],
    [120, 5],
    [250, 20],
    [180, 15],
    [100, 3],
    [220, 18],
    [150, 10],
    [260, 25]
], dtype=float)


Y = np.array([
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
    [0],
    [1]
], dtype=float)
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X)
Xn = normalizer(X)
model = tf.keras.Sequential(
    [   normalizer,
        tf.keras.layers.Dense(3,activation='sigmoid'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ]

)
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)#优化器更新w和b
)#是给神经网络配置训练规则。
history = model.fit(
    X,
    Y,
    epochs = 100
)#开始训练
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
x_test = np.array([200,16])
prediction = model.predict(x_test)#让模型进行预测的函数
print(prediction)
if prediction>=0.5:
    print('good coffee')
else:
    print('bad coffee')

























