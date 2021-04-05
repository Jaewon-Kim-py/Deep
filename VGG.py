#논문의 경우 imagenet(224*224) 데이터를 사용하였지만, 작성자가 사용한 데이터는 CIFAR10. input shape에 차이가 있어 논문처럼 16,19 layers 쌓지 않고 해당 사이즈를 축소
class VGG(Model):
  def __init__(self):
    super(VGG16, self).__init__()
    self.conv1_1 = Conv2D(64, 3, padding = 'same', activation = 'relu')
    self.conv1_2 = Conv2D(64, 3, padding = 'same', activation = 'relu')
    self.pool1 = MaxPool2D(2)
    self.conv2_1 = Conv2D(128, 3, padding = 'same', activation = 'relu')
    self.conv2_2 = Conv2D(128, 3, padding = 'same', activation = 'relu')
    self.pool2 = MaxPool2D(2)
    self.conv3_1 = Conv2D(128, 3, padding = 'same', activation = 'relu')
    self.conv3_2 = Conv2D(128, 3, padding = 'same', activation = 'relu')
    self.flatten = Flatten()
    self.dense1 = Dense(256, activation = 'relu')
    self.dense2 = Dense(256, activation = 'relu') 
    self.dense3 = Dense(10, activation = 'softmax')

  def call(self, x):
    x = self.conv1_1(x)
    x = self.conv1_2(x)
    x = self.pool1(x)
    x = self.conv2_1(x)
    x = self.conv2_2(x)
    x = self.pool2(x)
    x = self.conv3_1(x)
    x = self.conv3_2(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)

    return x
