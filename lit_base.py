import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics



LOSS = nn.CrossEntropyLoss()
class Base(pl.LightningModule):
  def __init__(self, model, LOSS = LOSS, lr = 1e-7, batch_size = 4, num_classes = 266, lr_step_size=3, lr_gamma=0.3):
    super(Base, self).__init__()
    self.num_classes = num_classes
    self.lr_step_size = lr_step_size
    self.lr_gamma = lr_gamma
    self.model = model
    self.loss_fn = LOSS
    self.lr = lr
    self.batch_size = batch_size

    #top 1 acc
    self.train_acc_top1 = torchmetrics.Accuracy()
    self.val_acc_top1 = torchmetrics.Accuracy()
    self.test_acc_top1 = torchmetrics.Accuracy()

    #top 5 acc
    self.train_acc_top5 = torchmetrics.Accuracy(top_k= 5)
    self.val_acc_top5 = torchmetrics.Accuracy(top_k= 5)
    self.test_acc_top5 = torchmetrics.Accuracy(top_k= 5)

  def forward(self, x):
    logits = self.model(x)
    return logits

  def training_step(self, batch, batch_index):
      x, y = batch
      y = y.squeeze()
      logits = self(*x)
      loss = self.loss_fn(logits, y)
      self.log("train_loss", loss)
      self.train_acc_top1(logits.softmax(dim=-1), y)
      self.train_acc_top5(logits.softmax(dim=-1), y)
      self.log("train_acc_top1", self.train_acc_top1, on_step= False, on_epoch=True, prog_bar= True)
      self.log("train_acc_top5", self.train_acc_top5, on_step= False, on_epoch=True, prog_bar= True)
      return loss

  def validation_step(self, batch, batch_index):
      *x, y = batch
      y = y.squeeze()
      logits = self(*x)
      loss = self.loss_fn(logits, y)
      self.log("val_loss", loss, prog_bar=True)
      self.val_acc_top1(logits.softmax(dim=-1), y)
      self.log('val_acc_top1', self.val_acc_top1, on_step= False, on_epoch = True, prog_bar= True)
      self.val_acc_top5(logits.softmax(dim=-1), y)
      self.log('val_acc_top5', self.val_acc_top5, on_step= False, on_epoch = True, prog_bar= True)


  def test_step(self, batch, batch_index):
    *x, y = batch
    y = y.squeeze()
    logits = self(*x)
    loss = self.loss_fn(logits, y)
    self.log("test_loss", loss, prog_bar=True)
    self.test_acc_top1(logits.softmax(dim=-1), y)
    self.log('test_acc_top1', self.val_acc_top1, on_step= False, on_epoch = True, prog_bar= True)
    self.test_acc_top5(logits.softmax(dim=-1), y)
    self.log('test_acc_top5', self.val_acc_top5, on_step= False, on_epoch = True, prog_bar= True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.lr_step_size, gamma=self.lr_gamma, last_epoch=-1, verbose= True)
    return {'optimizer' : optimizer, 'lr_scheduler' : scheduler}
