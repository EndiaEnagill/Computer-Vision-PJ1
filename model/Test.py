import numpy as np

class tester:
    def __init__(self, model, loss, dataloader):
        self.model = model
        self.loss = loss
        self.dataloader = dataloader

    def evaluate(self):
        total_loss = 0
        total_acc = 0
        for img, label in self.dataloader.get_test_batches():
            y_score = self.model.forward(img)
            y_pred = np.argmax(y_score, axis = 1)
            loss = self.loss.forward(y_score, label)

            total_loss += loss * len(img)
            total_acc += np.sum(y_pred == label)
        valid_loss = total_loss / len(self.dataloader.test_labels)
        valid_acc = total_acc / len(self.dataloader.test_labels)

        return valid_loss, valid_acc
