import torch
import wandb

from neuralop import Trainer, LpLoss
from neuralop.models import FNO
from torch.optim import AdamW
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_wandb = True
torch.manual_seed(0)

def load_data(train_path, test_path):

    train_dataset = torch.load(f'./no_data/{train_path}', weights_only=False)
    test_dataset = torch.load(f'./no_data/{test_path}', weights_only=False)

    train_loader = DataLoader(train_dataset, 
                          batch_size=32, 
                          num_workers=0, 
                          pin_memory=True, 
                          persistent_workers=False,)
    test_loader = DataLoader(test_dataset, 
                          batch_size=32, 
                          num_workers=0, 
                          pin_memory=True, 
                          persistent_workers=False, 
                          shuffle=False)
    
    return train_loader, test_loader

if __name__ == '__main__':

    train_loader, test_loader = load_data('train_sine.pt', 'test_sine.pt')
    
    model = FNO(n_modes=(16,),
             in_channels=1,
             out_channels=1,
             hidden_channels=32,
             projection_channel_ratio=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(),
                                    lr=8e-4,
                                    weight_decay=0.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    l2loss = LpLoss(d=1, p=2)

    train_loss = l2loss
    eval_losses={'l2': l2loss}

    if use_wandb:
        wandb.init(
            project='NeuralOperator',
        )

    trainer = Trainer(model=model, n_epochs=54,
                    device=device,
                    wandb_log=use_wandb,
                    eval_interval=3,
                    use_distributed=False,
                    verbose=True)
    
    trainer.train(train_loader=train_loader,
              test_loaders={'128': test_loader},
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses, 
              save_every=20, 
              save_dir='./models/sine')
