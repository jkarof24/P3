import torch
import torch.nn.functional as F

from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets.motif_generator import CycleMotif

from sklearn.utils import shuffle

#This is a list that will contain all graphs that will be created
graphs = []
#This is a list of class index  
labels = []

#This is the number of graphs
num_graphs_per_class = 200
#This is the number of nodes per graph
num_nodes_per_graph = 50
#This is the number of edges, that each node have on average, as when creating the graph its doing it stochastically, meaning that its a probability around 5 in this case, so some nodes might have 6 or 7 nodes while others might have 3 or 4
num_edges_per_node = 3


#This is the different motifs that can be added to the graphs to create different classes
motifs = ["HouseMotif", "CycleMotif", "GridMotif"]

for class_index, motif in enumerate(motifs):         
    if motif == "CycleMotif":
        motif = CycleMotif(7)         
        
    for i in range(num_graphs_per_class):
        #This creates a synthetic graph
        #Graph_generator creates the base graph with nodes and edges, where motif adds a subgraph in the shape of example a house onto the graph. This subgraph is comprised of 5 nodes and makes the full graph 305 nodes in this case
        dataset = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=num_nodes_per_graph, num_edges=num_edges_per_node),
            motif_generator=motif,
            num_motifs=3
        )
        #This appends each created graph into the "graphs" list
        graphs.append(dataset[0])
        #This appends the class index (Motif number (0 = "HouseMotif", 1 = "CycleMotif", 2 = "GridMotif")) to a list
        labels.append(class_index)

#This is to shuffle the lists (graphs and labels), so that all HouseMotif graphs arent in a row, the same goes for the other graphs/motifs (CycleMotif graphs and GridMotif graphs)
#The problem occures when they are in a row, when splitting the data we would get all HouseMotif graphs and CycleMotif graphs and some GridMotif graphs in the train_dataset, but only get GridMotif graphs in val_dataset and test_dataset
#This means that HouseMotif and CycleMotif is not represented in the validation and test dataset
graphs, labels = shuffle(graphs, labels, random_state=42)

#Here we go over each graph in the graph list, wher i will hold the graph index and data will hold the graph object itself
for i, data in enumerate(graphs):
    #Then we set the class of the graph to the value on index i in the label list. This is so the graph stores its class that it has been "assigned" when created
    data.y = torch.tensor([labels[i]])

#These three lines split the graphs into datasets, so as an example, train_dataset will be comprised of the first 80 percent of the graphs
train_dataset = graphs[:int(len(graphs) * 0.6)]
val_dataset = graphs[int(len(graphs) * 0.6): int(len(graphs) * 0.8)]
test_dataset = graphs[int(len(graphs) * 0.8):]

#This takes the graphs in each dataset and split them into batches of graphs in the size of 16 graphs per batch
#The benefit of a batches is that it takes less memory by splitting the graph into batches instead of taking it all at once, thereby making it faster and requiring less memory
#Shuffle is set to true, this randomizes the order of graphs each epoch, insuring a model that is more generalized
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

#There are no x values to predict y, which is the target value, so we need to create some synthetic x values
#data.num_nodes is the number of y values, for which we create an equivalent number of x values
for data in graphs:
    degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float().unsqueeze(1)

    noise = torch.randn(data.num_nodes, 1) * 0.01

    data.x = torch.cat([degree ,noise], dim=1)

#This is the value for the amount of classes
num_classes = dataset.num_classes

#This initializes the amount of in channels used in the model
#It takes one of the graphs in the list as they all have the same amount of node features for theire nodes
#Then is takes the amount of node features using x.shape[1], where graph.x.shape would give values in a tuple for (number of nodes, number of node features)
in_channel = graphs[0].x.shape[1]

#This initializes the amount of hidden channels used in the model
#We use hidden_channels to create a more in depth learning model and be able to use Relu and other non-linear activation functions
#This is an example of how its done: Lets say all nodes are people and we only have one node feature that is age, then it will create a new vector of 32 values where they could be (age, age relative to friend_count, average age of neighbors, ...)
hidden_channel = 32

#This class is the algorithm/model itself, where it inherits torch.nn.Module, that is the base for all neural networks in PyTorch
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        #Super calls the constructor of torch.nn.Module to use the base model, that can track hiden layers and parameters
        super(GCN, self).__init__()

        #These lines are used to initiate the three GCN layers with their input_features and output_features example (self.conv2 = GCNConv(in_channels (input_feature), hidden_channels (output_feature)))
        #The in_channels are the amount of features on the nodes in the graph and the hidden_channels are the 
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        #These line is used to aggragate the feature vector of all neighboring nodes + the node itself into a new updated value. It uses the spezified aggrecation function "Mean"
        #Its uses .relu to introduce non-linearity into the model, making it able to learn more complex patterns, when data is not linear. Its very common in real life that graph data is non-linear
        x = F.relu(self.conv1(x , edge_index))
        x = F.relu(self.conv2(x, edge_index))
        #global_mean_pool is used to take the mean value of all node embeddings in a graph and thereby assigning a vector of the mean node embedding values to a graph
        #Node embeddings are the finalized node features, meaning that its the node features, when the algorithm has agrigatted values from the neighbors  
        x = global_mean_pool(x, batch)
        #This line converts the global_mean_pool vector into a vector of logits using self.lin(x)
        #Afterwards it uses f.log_softmax() to convert the logits vector into log probability, that is used to calculate the loss
        return F.log_softmax(self.lin(x), dim=1)
    
#These three lines are for moving the calculations onto the GPU to make it faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_channel, hidden_channel, num_classes).to(device)
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
        loss = F.nll_loss(out, batch.y)
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
    #Initializes the count of correct classified graphs
    correct = 0
    #Initializes the total amount of graphs processed
    total = 0

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
        pred = out.argmax(dim=1)
        #This looks if the pred is the correct classification, if so then it adds +1 to the counter
        correct += (pred == batch.y).sum().item()
        #Here it just adds +1 to the total counter
        total += batch.y.size(0)
    #Here it returns the percentage of correct classified graphs
    return correct / total

#These lines are just initializing two variables and setting them to zero
best_val_acc = 0
test_acc = 0

#This for loop is running for some amount of epochs
for epoch in range(1,100):
    #loss is getting a loss value from the function train(), where its the model loss for this training loop/epoch
    loss = train()
    
    #Here we caputure the train, test and tmp_test accuracy
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    tmp_test_acc = test(test_loader)

    #This is to save the best validation accuracy and the accompanied test accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

    #This if-statement is used to print the loss value, validation accuracy and the test accuracy for each tenth epoch
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
              f"Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}")

#Here we print the best validation accuracy and the accompanied test accuracy
print(f"\nBest Val Acc: {best_val_acc:.4f}, Final Test Acc: {test_acc:.4f}")