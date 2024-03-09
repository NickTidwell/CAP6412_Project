import torch, argparse
from torch.utils.data import DataLoader
from src.dataset.SampleDataset import MyDataset
from src.models.SampleModel import SampleModel
from tqdm import tqdm  # Import tqdm for progress visualization

def load_model(args):
    type = args.model_type
    if type == "sample":
        return SampleModel(args)
    else:
        raise ValueError(f"Invalid model type: {type}")
    
def load_dataset(args):
    if args.dataset_type == "sample":
        rnd_data = torch.rand(args.batch_size, 3, 25, 25,10) # Batch, channel, X, Y
        return MyDataset(rnd_data)  
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    
class Trainer:
    def __init__(self, args, train_dataset):
        self.args = args
        self.train_dataset = train_dataset

    def train(self):

        model = load_model(self.args) 

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch 0/{self.args.epochs}")

        # Training loop
        for epoch in range(self.args.epochs):
            for batch in train_loader:
                inputs, targets = batch

                outputs = model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update tqdm progress bar
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Batch Loss: {loss.item():.4f}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #TDOO: Update these as needed
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--test_list', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument("--model-type", choices=["sample"], default="sample",
                    help="Specify the model architecture (sample, ...  , ...)")
    parser.add_argument("--dataset-type", choices=["sample"], default="sample",
                        help="Specify the dataset type (sample, etc.)")
    args = parser.parse_args()

    dataset = load_dataset(args)
    trainer = Trainer(args, train_dataset=dataset)
    trainer.train()

  
