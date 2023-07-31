import sys
import sys
sys.path.insert(0,'duh/tfts')

import matplotlib.pyplot as plt
import tfts
from tfts import AutoModel, AutoConfig, KerasTrainer


train_length = 24
predict_length = 8
(x_train, y_train), (x_valid, y_valid) = tfts.get_data("sine", train_length, predict_length, test_size=0.2)

model = AutoModel("transformer", predict_length=predict_length)
trainer = KerasTrainer(model)
trainer.train((x_train, y_train), (x_valid, y_valid), n_epochs=5)

pred = trainer.predict(x_valid)
trainer.plot(history=x_valid, true=y_valid, pred=pred)
plt.show()