import torch
import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, BatchNorm
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score
import numpy as np

from torch_geometric.datasets import MoleculeNet

dataset = MoleculeNet(root='data/MoleculeNet', name='HIV')

#This is the value for the amount of classes
num_classes = dataset.num_classes
clean_dataset = []
for data in dataset:
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()
    if not hasattr(data, 'edge_weight') or data.edge_weight is None:
        data.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float)
    if not torch.isnan(data.y).any():
        data.y = data.y.float()      
        clean_dataset.append(data)

    

dataset = clean_dataset

#These three lines split the full dataset into smaller datasets, so as an example, train_dataset will be comprised of the first 60 percent of the graphs
train_dataset = dataset[:int(len(dataset) * 0.6)]
val_dataset = dataset[int(len(dataset) * 0.6): int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

#This takes the graphs in each dataset and split them into batches of graphs in the size of 16 graphs per batch
#The benefit of a batches is that it takes less memory by splitting the graph into batches instead of taking it all at once, thereby making it faster and requiring less memory
#Shuffle is set to true, this randomizes the order of graphs each epoch, insuring a model that is more generalized
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

#This initializes the amount of in channels used in the model
#It takes one of the graphs in the list as they all have the same amount of node features for theire nodes
#Then is takes the amount of node features using x.shape[1], where graph.x.shape would give values in a tuple for (number of nodes, number of node features)
in_channel = dataset[0].x.shape[1]

#This initializes the amount of hidden channels used in the model
#We use hidden_channels to create a more in depth learning model and be able to use Relu and other non-linear activation functions
#This is an example of how its done: Lets say all nodes are people and we only have one node feature that is age, then it will create a new vector of 32 values where they could be (age, age relative to friend_count, average age of neighbors, ...)
hidden_channel = 64

#This class is the algorithm/model itself, where it inherits torch.nn.Module, that is the base for all neural networks in PyTorch
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout = 0.5):
        #Super calls the constructor of torch.nn.Module to use the base model, that can track hiden layers and parameters
        super(GCN, self).__init__()

        #These lines are used to initiate the three GCN layers with their input_features and output_features example (self.conv2 = GCNConv(in_channels (input_feature), hidden_channels (output_feature)))
        #The in_channels are the amount of features on the nodes in the graph and the hidden_channels are the amount of node embeddings
        #TILFØJ - batchNorm og dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin = torch.nn.Linear(hidden_channels, 1)
        
        self.dropout = dropout
    def forward(self, x, edge_index, batch):
        #These line is used to aggragate the feature vector of all neighboring nodes + the node itself into a new updated value. It uses the spezified aggrecation function "Mean"
        #Its uses .relu to introduce non-linearity into the model, making it able to learn more complex patterns, when data is not linear. Its very common in real life that graph data is non-linear
        #TILFØJ - batchNorm og dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #global_mean_pool is used to take the mean value of all node embeddings in a graph and thereby assigning a vector of the mean node embedding values to a graph
        #Node embeddings are the finalized node features, meaning that its the node features, when the algorithm has agrigatted values from the neighbors  
        x = global_mean_pool(x, batch)
        
        #This line converts the global_mean_pool vector into a vector of logits using self.lin(x)
        #Afterwards it uses f.log_softmax() to convert the logits vector into log probability, that is used to calculate the loss
        #ÆNDRE TEKST!
        return self.lin(x)
    
#These three lines are for moving the calculations onto the GPU to make it faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_channel, hidden_channel).to(device)
#The optimizer is a function used to update the weights in each node of the graph
#lr is learning rate and this is how large the steps are towards the optimal value (critical point), make sure the steps are not too large as it will keep overshooting the critical point
#weight_decay is a way to make the weights of the model smaller, as we dont want large weight. This is because we dont want weight that are very sensitive to change as large weights are, this makes the model overfit.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)


#This is the function for training the model
#FIND UD AF!
def train():
    model.train()
    #Here it just initilizes the total_loss variable
    total_loss = 0
    #Here the model goes through all batches in the train_loader
    for batch in train_loader:
        #This moves the calculations to Cuda if available else CPU
        batch = batch.to(device)
        #Here it removes the old gradient from memory, as pytorch will save the previous calculated gradient and if its not removed then the program can mix up the gradients and not choose the one calculated in this instance of the for loop
        optimizer.zero_grad()
        #Here we insert the batch of graphs into the model and get the log probability out, this is a tensor, where each row is a graph in the batch and each columns is a log probability of a class, so three columns in this case, one for each motif
        out = model(batch.x, batch.edge_index, batch.batch)
        #This calculates the loss, where it takes the log probability of the actual class of all graphs and then averages it out
        #By the actual class, i mean that even tho the model predicted class 1, then it still takes the log probability for class 0, if that was the true class
        #ÆNDRE TEKST
        loss = F.binary_cross_entropy_with_logits(out.view(-1), batch.y.view(-1).float())
        #This initiates backpropagation, where it calculates the gradient of the loss respect to the trainable parameters (The weights assosiated to the node features)
        loss.backward()
        #This tells the optimizer to change the model parameters
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

#This is the function for testing the model with the argument "Loader", this is the train_loader, val_loader and test_loader
#This insures that pytorch doesnt save tensors for gradient calculations and therefore saves memory (But this is only for the test function)
@torch.no_grad()
def test(loader):
    model.eval()

    #This is a list of the true label of all graphs, so if the true label is the binary value false, then there will be a zero, if on the other hand its the binary value true, then its a one
    y_true = []
    #This is a list of probabilities of a graph having the binary value true. This list contains a probability for each graph in the dataset. 
    #The graph on each index of the y_prob list is equal to the index for the same graph in the y_true list. Meaning that a graph on index 0 of the y_prob has the index 0 in y_true
    y_prob = []
    
    #This goes through each batch of a loader (train_loader, val_loader or test_loader)
    for batch in loader:
        #This moves the batch calculations to Cuda if available else CPU
        batch = batch.to(device)
        #This runs the model on a batch, where batch.x is all node features for all nodes in all the graphs of the batch.
        #Batch.edge_index is a matrix of all edges in the batch of graphs, where row 1 is the source node for the edge and row 2 is the target node for the edge, so each column is an edge and there is only two rows
        #Batch.batch is a tensor of whitch nodes belongs to which graph in the batch. Its a 1D tensor, that is the length of the total amount of nodes in all the graphs of the batch 
        #Lets say we have 4 graphs in a batch, then the values in the tensor can go from 0 to 3, where the value will indicate what graph the node belongs to
        out = model(batch.x, batch.edge_index, batch.batch)
        #This takes picks the class with the highest log probability and thereby choosing the class that the model is most confident to give the graph 
        #ÆNDRE TEKST
        prob = torch.sigmoid(out.view(-1)).cpu().numpy()
        
        #Then the probability is extended to the list y_prob. The reason for using extend is that prob is a tensor of probabilities and roc_auc_score needs a clean list of values and not a list of list, this is where extend extracts the values of the tensor 
        # and add the values in the tensor to the end of the list
        y_prob.extend(prob)
        #Here the true binary value for a graph is extracted and put into the y_true list
        y_true.extend(batch.y.view(-1).cpu().numpy())
    
    #This checks if the list y_true har more than one unique value, as roc_auc_score needs to compare two unique values, meaning that it needs to compare one true binary false value and one true binary true value
    #If there isnt two unique values, then the program would crash, therefore this security measure is set in
    if len(np.unique(y_true)) < 2:
        return float('nan')
    else:
        #Here the ROC_AUC_SCORE is calculated, where it compares two graphs, one being true "false" and one being true "true"
        return roc_auc_score(y_true, y_prob)

#These lines are just initializing two variables and setting them to zero
best_val_acc = 0
test_acc = 0

#This for loop is running for some amount of epochs
for epoch in range(1,100):
    #loss is getting a loss value from the function train(), where its the model loss for this training loop/epoch
    loss = train()


    val_auc = test(val_loader)
    test_auc = test(test_loader)

    #This if-statement is used to print the loss value, validation accuracy and the test accuracy for each tenth epoch
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
              f"AUC-Val: {val_auc:.4f}, "
              f"AUC-Test: {test_auc:.4f}")
