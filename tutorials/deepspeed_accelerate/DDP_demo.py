from accelerate import Accelerator,DeepSpeedPlugin
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim) 
    
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    input_dim = 10
    hidden_dim = 256
    output_dim = 2
    batch_size = 64
    data_size = 1000

    input_data = torch.rand(data_size,input_dim,dtype=torch.float16)
    labels = torch.rand(data_size,output_dim,dtype=torch.float16)

    dataset = TensorDataset(input_data,labels)
    dataloader = DataLoader(dataset,batch_size=batch_size)

    model = SimpleNet(input_dim,hidden_dim,output_dim).to(dtype=torch.float16)
    # deepspeed = DeepSpeedPlugin(zero_stage=2,gradient_clipping=1.0)
    # accelerator = Accelerator(deepspeed_plugins=deepspeed)

    accelerator = Accelerator()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.MSELoss()
   
    model,optimizer,dataloader = accelerator.prepare(model,optimizer,dataloader)

    for epoch in range(10):
        model.train()
        for batch in dataloader:
            inputs,labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            accelerator.backward(loss)
            optimizer.step()
        print(f'Epoch {epoch} loss {loss.item()}')
    
    accelerator.save(model.state_dict(),'model.pth')