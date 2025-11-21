import torch
import torch.nn.functional as F

from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.nn import GCNConv


#This code is used to construc/fetch the data
dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=30000, num_edges=5),
    motif_generator='house',
    num_motifs=80,
)
#When a dataset is constructed/fetched as shown earlier, a dataset object is created which can contain multiple graphs (Example: (dataset = [graph0, graph1, graph2,...]))
#Therefore, we need to access a specific graph before we can work with the data, even if the dataset object only contains one graph
data = dataset[0]

#There are no x values to predict y, which is the target value, so we need to create some synthetic x values, which is done in the two lines below
#num_nodes is the number of y values, for which we create an equivalent number of x values
num_nodes = data.y.shape[0]
data.x = torch.bincount(data.edge_index[0], minlength=num_nodes).float().unsqueeze(1)

num_classes = int(data.y.max().item()) + 1 


#This class is the algorithm/model itself, where it inherits torch.nn.Module, that is the base for all neural networks in PyTorch
class GCN(torch.nn.Module):
    def __init__(self):
        #Super calls the constructor of torch.nn.Module to use the base model, that can track hiden layers and parameters
        super(GCN, self).__init__()

        #These lines are used to initiate the three GCN layers with their input_features and output_features example (self.conv2 = GCNConv(32 (input_feature), 16 (output_feature)))
        self.conv1 = GCNConv(data.x.shape[1], 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        #These line is used to aggragate the feature vector of all neighboring nodes + the node itself into a new updated value. It uses the spezified aggrecation function "Mean"
        #Its uses .relu to introduce non-linearity into the model, making it able to learn more complex patterns, when data is not linear. Its very common in real life that graph data is non-linear
        x = F.relu(self.conv1(x , edge_index))
        x = F.relu(self.conv2(x, edge_index))
        #This layer is used for classification and not feature extraction and therefore we do not use relu, because relu would destort the negative logits and make them zero, thereby introducing bias into the logits/classification
        x = self.conv3(x, edge_index)
        #This line converts the logits into log probabilities to be used in train and test
        return F.log_softmax(x, dim= 1)
    
#These two lines are for moving the calculations onto the GPU to make it faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, data = GCN().to(device), data.to(device)
#The optimizer is a function used to update the weights in each node of the graph
#lr is learning rate and this is how large the steps are towards the optimal value (critical point), make sure the steps are not too large as it will keep overshooting the critical point
#weight_decay is a way to make the weights of the model smaller, as we dont want large weight. This is because we dont want weight that are very sensitive to change as large weights are, this makes the model overfit.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

#This is for setting up the training data, as the graph data didnt include this from the beginning
perm = torch.randperm(num_nodes)

#This defines the sizes of the different parts of training
train_size = int(0.8 * num_nodes)
val_size = int(0.1 * num_nodes)
test_size = num_nodes - train_size - val_size

#Here we are creatiing tensors/matrix with false value in them
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

#Here we are overwritting some of the false with true using perm, this happens by perm having a matrix with random 
data.train_mask[perm[:train_size]] = True
data.val_mask[perm[train_size:train_size + val_size]] = True
data.test_mask[perm[train_size + val_size:]] = True

#This is the function for training the model
def train():
    model.train()
    #This line is used to wipe the old gradient, as pytorch saves gradients and can begin to mix them up if they stack up
    optimizer.zero_grad()
    #This calculates the loss in training, where it compares the estimated values (model(data.x, data.edge_index)[data.train_mask]), with the actual values (data.y[data.train_mask])
    loss = F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
    #This is initializing the backpropagation, where it calculates the gradient for all parameters and storing them
    loss.backward()
    #This takes the gradients and uses them to change the parameters so it reduces the loss
    optimizer.step()
    #This returns the loss value, from the loss tensor. F.nll_loss would return (tensor(some loss-value, the gradient function used to create this tensor)), so loss.item is used to extract the loss-value
    return loss.item()

#This is the function for testing the model
def test():
    model.eval()
    #logits contain the log probabilities of the model that is returned via (return F.log_softmax(x, dim=1)) and accs is an empty list that will contain the accuracies of the model
    logits, accs = model(data.x, data.edge_index), []
    #Here we loop through the three masks (train, val, test), where each mask is a 1D-boolean tensor of the length num_nodes. This means that the tensor is contains true or false for each node
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        #This takes log probability from each class given a node in logits[mask], then it picks the class with the highest log probability and assigns that class to the node, as class with the highest log probability is what the model is most confident in
        #logits[mask] is a tensor of all three log probabilities for each nodes 
        pred = logits[mask].max(1)[1]
        #This line we want to compare our predicted class for each node (pred) with the actual class (data.y[mask]) of the node using pred.eq(data.y[mask])
        #Then it counts the amount of correctly predicted classes with .sum() and extracting that value using .item()
        #Then we devide by the total amount of nodes in that mask using mask.sum().item() to get a percentage of true predicted classes
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        #Then we append the accuracy to the accs list
        accs.append(acc)
    #This function then returns the accuracies
    return accs

#These lines are just initializing two variables and setting them to zero
best_val_acc = 0
test_acc = 0
#This for loop is running for some amount of epochs
for epoch in range(1,100):
    #loss is getting a loss value from the function train(), where its the model loss for this training loop/epoch
    loss = train()
    #This line captures the accuracy for val and test, where train is left out using "_"
    _, val_acc, tmp_test_acc = test()
    #This if-statement is used to print the loss value, validation accuracy and the test accuracy for each tenth epoch
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}")


