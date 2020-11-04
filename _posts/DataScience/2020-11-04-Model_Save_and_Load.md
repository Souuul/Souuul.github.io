---
title:  "[Data Science] Model Save & Load"
header:
  teaser: "/assets/image/1200px-TensorFlowLogo.svg.png"
categories: 
  - Data Science
tags:
  - Model
  - Save and Load
---
## Model Save & Load

이번시간에는 학습한 모델을 저장하고 불러와서 이용해보도록 하겠습니다. 지난시간에는 전의학습을 통해서 모델을 사용하는방법을 배웠다면 이번시간에는 중간에 종료된 모델을 다시 이어서 사용해보도록 하겠습니다. 

### Model Save

모델을 저장하는 방법에는 여러가지가 있지만 이번에는 keras의 model.save를 통해서 저장을 해보도록 하겠습니다.

``` python
model.save('경로/파일명.h5')    
```

먼저 학습을 진행하고 상기 명령어를 통하여 모델을 저장하면 h5 확장자의 파일이 지정된 경로에 저장됩니다.

### Model Load

그렇다면 저장된 모델을 불러와 보도록 하겠습니다. 사용할 모델을 그대로 사용하시려면 불러온 모델에 변수를 할당하고 compile과 fit을 통하여 학습을 진행하시면 됩니다.

```python
# Model Load
model = tf.keras.models.load_model('./Contest_model/contest_vgg.h5')

# Training
model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

epochs = 30
history = model.fit_generator(
    trainGen, 
    epochs=epochs,
    steps_per_epoch=100, 
    validation_data=validationGen,
    validation_steps=100,
    verbose=1
```
