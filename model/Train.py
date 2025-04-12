import numpy as np
import time
import os
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, optimizer, loss, dataloader, num_epochs, eval_step):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.eval_step = eval_step

        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []

        self.model_cache = {}
        self.log = []
        self.time = []

    def train(self, save_model_cache = 0, versible = False):
        """
        save_model_cache: 是否保存中间模型, 默认为0, 否则设定为模型保存编号(int)
        """
        # 是否保存中间模型
        if save_model_cache:
            folder_path = f'./model_data/{save_model_cache}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        for epoch in range(1, self.num_epochs + 1):
            print(f'training for epoch {epoch}......')
            start = time.time()
            total_loss = 0
            total_acc = 0
            for img, label in self.dataloader.get_train_batches():
                y_score = self.model.forward(img)
                y_pred = np.argmax(y_score, axis = 1)
                loss = self.loss.forward(y_score, label)

                # 保存训练误差和准确率
                total_loss += loss * len(img)
                total_acc += np.sum(y_pred == label)

                # 梯度下降
                grad = self.loss.backward()
                self.model.backward(grad)
                self.optimizer.step(self.model)
            end = time.time() # 记录训练一个epoch需要的时间
            self.time.append(end - start)

            # 记录训练误差和准确率
            train_loss = total_loss / len(self.dataloader.train_labels)
            train_acc = total_acc / len(self.dataloader.train_labels)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            # 每eval_step验证一次模型效果
            if epoch % self.eval_step == 0:
                valid_loss, valid_acc = self.evaluate()
                self.valid_loss.append(valid_loss)
                self.valid_acc.append(valid_acc)

                #保存中间结果
                self.model_cache[epoch] = self.model.copy()

                # 保存训练日志
                log = f"""
                        Epoch: {epoch}
                        Learning Rate: {self.optimizer.learning_rate:.4f}
                        Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}
                        Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc:.4f}
                        Train average Time: {(sum(self.time)/len(self.time)):.4f}
                    """
                self.log.append(log)
            # 是否显示训练过程
            if versible:
                if epoch % 5 == 0:
                    self.train_path(dir = None, epoch = epoch)
        # 保存最后一次的训练结果
        self.model_cache[epoch] = self.model.copy()
    
    def evaluate(self):
        total_loss = 0
        total_acc = 0
        for img, label in self.dataloader.get_valid_batches():
            y_score = self.model.forward(img)
            y_pred = np.argmax(y_score, axis = 1)
            loss = self.loss.forward(y_score, label)

            # 保存训练误差和准确率
            total_loss += loss * len(img)
            total_acc += np.sum(y_pred == label)
        valid_loss = total_loss / len(self.dataloader.valid_labels)
        valid_acc = total_acc / len(self.dataloader.valid_labels)

        return valid_loss, valid_acc
    
    def save_log(self, dir):
        with open(os.path.join(dir, 'train_log.txt'), 'w') as f:
            for log in self.log:
                f.write(log + '\n')
    
    def clear_cache(self):
        self.model_cache = {}
        self.log = []

    def save_best_model(self, dir):
        best_idx = (np.argmin(self.valid_loss) + 1) * self.eval_step
        m = self.model_cache[best_idx]
        m.save_model(os.path.join(dir, f'best_model.pkl'))
        return
    
    def valid_path(self, dir = None, epoch = 1):
        plt.subplot(1, 2, 1)
        plt.plot(range(0, epoch + 1, self.eval_step)[1:], self.valid_acc, label = 'valid accuracy', color = 'blue')
        plt.xlabel('epoch')
        plt.title('Valid Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(range(0, epoch + 1, self.eval_step)[1:], self.valid_loss, label = 'valid loss', color = 'red')
        plt.xlabel('epoch')
        plt.title('Valid Loss')

        plt.suptitle('Valid Path')

        if dir:
            plt.savefig(os.path.join(dir, 'valid_path.png'))
        plt.show()
    
    def train_path(self, dir = None, epoch = 1):
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch + 1), self.train_acc, label = 'train accuracy', color = 'blue')
        plt.xlabel('epoch')
        plt.title('Train Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch + 1), self.train_loss, label = 'train loss', color = 'red')
        plt.xlabel('epoch')
        plt.title('Train Loss')
        
        plt.suptitle('Train Path')

        if dir:
            plt.savefig(os.path.join(dir, 'train_path.png'))
        plt.show()

    def loss_plot(self, dir = None, epoch = 1):
        plt.plot(range(1, epoch + 1), self.train_loss, label = 'train loss', color = 'blue')
        plt.plot(range(0, epoch + 1, self.eval_step)[1:], self.valid_loss, label = 'valid loss', color = 'red')

        plt.title('Loss')
        plt.xlabel('epoch')
        plt.legend()

        if dir:
            plt.savefig(os.path.join(dir, 'Loss.png'))
        plt.show()

    def acc_plot(self, dir = None, epoch = 1):
        plt.plot(range(1, epoch + 1), self.train_acc, label = 'train acc', color = 'blue')
        plt.plot(range(0, epoch + 1, self.eval_step)[1:], self.valid_acc, label = 'valid acc', color = 'red')

        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.legend()

        if dir:
            plt.savefig(os.path.join(dir, 'Acc.png'))
        plt.show()