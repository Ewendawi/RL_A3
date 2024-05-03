
import torch
import torch.nn.functional as F


# outpput action value
# use tanh to bound the output in [-1, 1] and scale it to the output_bound
class DeterministicContinuousPolicyNetwork(torch.nn.Module):
    def __init__(self, input_size, layers_dim, output_size, ouput_bound=1.0):
        super(DeterministicContinuousPolicyNetwork, self).__init__()
        self.fcs = torch.nn.ModuleList() 
        for i, dim in enumerate(layers_dim):
            if i == 0:
                self.fcs.append(torch.nn.Linear(input_size, dim))
            else:
                self.fcs.append(torch.nn.Linear(layers_dim[i-1], dim))
        self.fc_a = torch.nn.Linear(layers_dim[-1], output_size)
        self.output_bound = ouput_bound

    def forward(self, x): 
        for fc in self.fcs:
            x = F.relu(fc(x)) 

        x = torch.tanh(self.fc_a(x)) * self.output_bound
        return x

# output mu and std of action
# std is independent of the input state
class ContinuousPolicyNetwork2(torch.nn.Module):
    def __init__(self, input_size, layers_dim, output_size, output_bound):
        super(ContinuousPolicyNetwork2, self).__init__()
        self.fcs = torch.nn.ModuleList() #fully connected layers
        for i, dim in enumerate(layers_dim):
            if i == 0:
                self.fcs.append(torch.nn.Linear(input_size, dim))
            else:
                self.fcs.append(torch.nn.Linear(layers_dim[i-1], dim))
        self.fc_mu = torch.nn.Linear(layers_dim[-1], output_size)

        self.log_std = torch.nn.Parameter(torch.zeros(output_size))
        self.output_bound = torch.tensor(output_bound)

    def forward(self, x):
        for fc in self.fcs:
            x = F.relu(fc(x)) 

        mu = self.output_bound * torch.tanh(self.fc_mu(x))
        log_std = self.log_std.expand_as(mu)
        std = torch.exp(log_std)
        return mu, std

# output mu and std of action
# std is dependent on the input state
class ContinuousPolicyNetwork(torch.nn.Module):
    def __init__(self, input_size, layers_dim, output_size, output_bound):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fcs = torch.nn.ModuleList() 
        for i, dim in enumerate(layers_dim):
            if i == 0:
                self.fcs.append(torch.nn.Linear(input_size, dim))
            else:
                self.fcs.append(torch.nn.Linear(layers_dim[i-1], dim))
        self.fc_mu = torch.nn.Linear(layers_dim[-1], output_size)
        self.fc_std = torch.nn.Linear(layers_dim[-1], output_size)
        self.output_bound = output_bound

    def forward(self, x):
        for fc in self.fcs:
            x = F.relu(fc(x)) 

        mu = self.output_bound * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class DiscretePolicyNetwork(torch.nn.Module):
    def __init__(self, input_size, layers_dim, output_size):
        super(DiscretePolicyNetwork, self).__init__()
        self.fcs = torch.nn.ModuleList() 
        for i, dim in enumerate(layers_dim):
            if i == 0:
                self.fcs.append(torch.nn.Linear(input_size, dim))
            else:
                self.fcs.append(torch.nn.Linear(layers_dim[i-1], dim))
        self.fc_A = torch.nn.Linear(layers_dim[-1], output_size) 

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0)
            
            # module.weight.data.normal_(mean=0.0, std=1.0)
            # if module.bias is not None:
            #     module.bias.data.zero_()


    def forward(self, x): 
        for fc in self.fcs:
            x = F.relu(fc(x)) 

        x = F.softmax(self.fc_A(x), dim=1)
        return x


class ValueNetwork(torch.nn.Module):
    def __init__(self, input_size, layers_dim):
        super(ValueNetwork, self).__init__()
        self.fcs = torch.nn.ModuleList() 
        for i, dim in enumerate(layers_dim):
            if i == 0:
                self.fcs.append(torch.nn.Linear(input_size, dim))
            else:
                self.fcs.append(torch.nn.Linear(layers_dim[i-1], dim))
        self.fc_V = torch.nn.Linear(layers_dim[-1], 1)

    def forward(self, x): 
        for fc in self.fcs:
            x = F.relu(fc(x)) 

        x = self.fc_V(x)
        return x

# output Q value of discrete actions
class QNetwork(torch.nn.Module):
    def __init__(self, input_size, layers_dim, output_size):
        super(QNetwork, self).__init__()
        self.fcs = torch.nn.ModuleList() 
        for i, dim in enumerate(layers_dim):
            if i == 0:
                self.fcs.append(torch.nn.Linear(input_size, dim))
            else:
                self.fcs.append(torch.nn.Linear(layers_dim[i-1], dim))
        self.fc_Q = torch.nn.Linear(layers_dim[-1], output_size) 

    def forward(self, x): 
        for fc in self.fcs:
            x = F.relu(fc(x)) 

        x = self.fc_Q(x)
        return x

# output Q value of continous actions
# 1. embed state and action into two latent spaces
# 2. concatenate the two latent spaces as the input of the Q network
class QNetworkContinnous(torch.nn.Module):
    def __init__(self, state_size, state_dim, action_size, action_dim, layers_dim):
        super(QNetworkContinnous, self).__init__()
        self.state_fcs = torch.nn.ModuleList() #fully connected layers
        for i, dim in enumerate(state_dim):
            if i == 0:
                self.state_fcs.append(torch.nn.Linear(state_size, dim))
            else:
                self.state_fcs.append(torch.nn.Linear(state_dim[i-1], dim))
        last_state_dim = state_size
        if len(state_dim) > 0:
            last_state_dim = state_dim[-1]
        
        self.action_fcs = torch.nn.ModuleList()
        for i, dim in enumerate(action_dim):
            if i == 0:
                self.action_fcs.append(torch.nn.Linear(action_size, dim))
            else:
                self.action_fcs.append(torch.nn.Linear(action_dim[i-1], dim))
        last_action_dim = action_size
        if len(action_dim) > 0:
            last_action_dim = action_dim[-1]

        self.fcs = torch.nn.ModuleList()
        for i, dim in enumerate(layers_dim):
            if i == 0:
                self.fcs.append(torch.nn.Linear(last_state_dim + last_action_dim, dim))
            else:
                self.fcs.append(torch.nn.Linear(layers_dim[i-1], dim))
        self.fc_Q = torch.nn.Linear(layers_dim[-1], 1) 

    def forward(self, state, action): 
        # state_x = torch.nn.functional.normalize(state, dim=0)
        state_x = state
        for fc in self.state_fcs:
            state_x = F.relu(fc(state_x))
        # action_x = torch.nn.functional.normalize(action, dim=0)
        action_x = action
        for fc in self.action_fcs:
            action_x = F.relu(fc(action_x))
        x = torch.cat([state_x, action_x], 1)
        for fc in self.fcs:
            x = F.relu(fc(x)) 
        x = self.fc_Q(x)
        return x