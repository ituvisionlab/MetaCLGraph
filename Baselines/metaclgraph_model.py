import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from .ergnn_utils import *
import dgl
import numpy as np

samplers = {'CM': CM_sampler(plus=False), 'CM_plus':CM_sampler(plus=True), 'MF':MF_sampler(plus=False), 'MF_plus':MF_sampler(plus=True),'random':random_sampler(plus=False)}


def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs

def kaiming_normal_init(m):
	if isinstance(m, torch.nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, torch.nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class NET(torch.nn.Module):
    """
        A template for implementing new methods for NCGL tasks. The major part for users to care about is the implementation of the function ``observe()``, which is how the implemented NCGL method learns each new task.

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

        """
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager
        self.args = args
        self.activation = F.elu
        self.sampler = samplers[args.lamamlgnnadam_args['sampler'][0]]
        
        # setup network
        self.net = model
        # self.net.apply(kaiming_normal_init)

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        self.current_task = 0
        self.n_classes = 5
        self.n_known = 0
        self.prev_model = None
        
        # replay buffer
        
        self.current_task = -1
        self.buffer_node_ids = {}
        self.budget = int(args.lamamlgnnadam_args['budget'][0])
        self.d_CM = args.lamamlgnnadam_args['d'][0] # d for CM sampler of ERGNN
        self.aux_g = None
        
        #lamaml
        
        self.alpha_lr = nn.ParameterList([])
        for p in self.net.parameters():
            self.alpha_lr.append(nn.Parameter(5e-3 * torch.ones(p.shape, requires_grad=True).to(device='cuda:{}'.format(args.gpu))))
        self.opt_lr = torch.optim.Adam(self.alpha_lr, lr=args.lr, weight_decay=args.weight_decay)
        self.ep = 0
    
    def forward(self, features):
        output = self.net(features)
        return output
        # h = features
        # h = self.feature_extractor(g, h)[0]
        # if len(h.shape)==3:
        #     h = h.flatten(1)
        # h = self.activation(h)
        # h = self.gat(g, h)[0]
        # if len(h.shape)==3:
        #     h = h.mean(1)
        # return h
    
    def inner_update(self, g, features, train_ids, fast_weights, labels, t):
        if fast_weights is None:
            fast_weights = list(map(lambda p: p, self.net.parameters()))
        o1, o2 = self.task_manager.get_label_offset(t)
        if self.args.cls_balance:
            n_per_cls = [(labels[train_ids] == j).sum() for j in range(self.args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(self.args.gpu))
        else:
            loss_w_ = [1. for i in range(self.args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(self.args.gpu))
            
        logits, _ = self.net.maml(g, features, fast_weights)
        if self.args.classifier_increase:
            loss = self.ce(logits[train_ids,o1:o2], labels[train_ids], weight=loss_w_[o1: o2])
        else:
            loss = self.ce(logits[train_ids], labels[train_ids], weight=loss_w_)
        grads = list(torch.autograd.grad(loss, fast_weights))
        
        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], -0.15, 0.15)
        # fast_weights = list(map(lambda p: p[1][0] - nn.functional.leaky_relu(p[1][1], negative_slope=0.2) * p[0], zip(grads, zip(fast_weights, self.alpha_lr))))
        if self.ep==1:
            self.first_moment = list(map(lambda p: torch.zeros_like(p), grads))
            self.second_moment = list(map(lambda p: torch.zeros_like(p), grads))
        self.first_moment = list(map(lambda p: 0.9*p[0] + (1-0.9)*p[1], zip(self.first_moment, grads)))
        self.second_moment = list(map(lambda p: 0.99*p[0] + (1-0.99)*(p[1]**2), zip(self.second_moment, grads)))
        self.first_moment = [p/(1-0.9**(self.ep)) for p in self.first_moment]
        self.second_moment = [p/(1-0.99**(self.ep)) for p in self.second_moment]
        fast_weights = list(map(lambda p: p[0] - nn.functional.leaky_relu(p[1][2], negative_slope=0.2) * (p[1][0]/(torch.sqrt(p[1][1]+1e-8))), zip(fast_weights, zip(self.first_moment, self.second_moment, self.alpha_lr)))) # adam
        return fast_weights
    
    def inner_update_batch(self, blocks, features, train_ids, fast_weights, labels, t):
        if fast_weights is None:
            fast_weights = list(map(lambda p: p, self.net.parameters()))
        o1, o2 = self.task_manager.get_label_offset(t)
        if self.args.cls_balance:
            n_per_cls = [(labels == j).sum() for j in range(self.args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(self.args.gpu))
        else:
            loss_w_ = [1. for i in range(self.args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(self.args.gpu))
            
        logits, _ = self.net.maml_batch(blocks, features, fast_weights)
        loss = self.ce(logits[:, o1:o2], labels.squeeze(-1), weight=loss_w_[o1: o2])
        grads = list(torch.autograd.grad(loss, fast_weights))
        
        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], -0.15, 0.15)
        # fast_weights = list(map(lambda p: p[1][0] - nn.functional.leaky_relu(p[1][1], negative_slope=0.2) * p[0], zip(grads, zip(fast_weights, self.alpha_lr))))
        if self.ep==1:
            self.first_moment = list(map(lambda p: torch.zeros_like(p), grads))
            self.second_moment = list(map(lambda p: torch.zeros_like(p), grads))
        self.first_moment = list(map(lambda p: 0.9*p[0] + (1-0.9)*p[1], zip(self.first_moment, grads)))
        self.second_moment = list(map(lambda p: 0.99*p[0] + (1-0.99)*(p[1]**2), zip(self.second_moment, grads)))
        self.first_moment = [p/(1-0.9**(self.ep)) for p in self.first_moment]
        self.second_moment = [p/(1-0.99**(self.ep)) for p in self.second_moment]
        fast_weights = list(map(lambda p: p[0] - nn.functional.leaky_relu(p[1][2], negative_slope=0.2) * (p[1][0]/(torch.sqrt(p[1][1]+1e-8))), zip(fast_weights, zip(self.first_moment, self.second_moment, self.alpha_lr)))) # adam
        return fast_weights
    
    def meta_loss(self, g, features, train_ids, fast_weights, labels, t):
        logits, _ = self.net.maml(g, features, fast_weights) # implement maml
        o1, o2 = self.task_manager.get_label_offset(t)
        if self.args.cls_balance:
            n_per_cls = [(labels[train_ids] == j).sum() for j in range(self.args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(self.args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(self.args.gpu))
        if self.args.classifier_increase:
            loss = self.ce(logits[train_ids,o1:o2], labels[train_ids], weight=loss_w_[o1:o2])
        else:
            loss = self.ce(logits[train_ids], labels[train_ids], weight=loss_w_)
        return loss, logits
    def meta_loss_batch(self, blocks, features, train_ids, fast_weights, labels, t):
        logits, _ = self.net.maml_batch(blocks, features, fast_weights) # implement maml
        o1, o2 = self.task_manager.get_label_offset(t)
        if self.args.cls_balance:
            n_per_cls = [(labels == j).sum() for j in range(self.args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(self.args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(self.args.gpu))
        loss = self.ce(logits[:,o1:o2], labels.squeeze(-1), weight=loss_w_[o1: o2])
        return loss, logits
    
    def push_to_buffer(self, ids_per_cls_train, g, features, dataset, labels, t):
        sampled_ids = self.sampler(ids_per_cls_train, self.budget, features, self.net.second_last_h, self.d_CM)
        old_ids = g.ndata['_ID'].cpu() # '_ID' are the original ids in the original graph before splitting
        self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
    
    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                The method for learning the given tasks. Each time a new task is presented, this function will be called to learn the task. Therefore, how the model adapts to new tasks and prevent forgetting on old tasks are all implemented in this function.
                More detailed comments accompanying the code can be found in the source code of this template in our GitHub repository.

                :param args: Same as the args in __init__().
                :param g: The graph of the current task.
                :param features: Node features of the current task.
                :param labels: Labels of the nodes in the current task.
                :param t: Index of the current task.
                :param prev_model: The model obtained after learning the previous task.
                :param train_ids: The indices of the nodes participating in the training.
                :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                :param dataset: The entire dataset (not in use in the current baseline).

                """

        # Always set the model in the training mode, since this function will not be called during testing
        self.net.train()
        # if the given task is a new task, set self.current_task to denote the current task index. This is mainly designed for the mini-batch training scenario, in which the data of a task may not come in simultaneously, and each batch of data may either belong to an existing task or a new task..
        if t != self.current_task:            
            self.current_task = t        
        self.ep += 1
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t) # Since the output dimensions of a model may correspond to multiple tasks, offset1 and offset2 denote the starting end ending dimension of the output for task t.
        fast_weights = None
        
        if t==0:
            fast_weights = self.inner_update(g, features, train_ids, fast_weights, labels, t)
            meta_loss, logits = self.meta_loss(g, features, train_ids, fast_weights, labels, t)
            self.net.zero_grad()
            self.alpha_lr.zero_grad()
            meta_loss.backward()
            
        else:
            nodes = [self.buffer_node_ids[i] for i in range(t)]
            if self.ep ==1:
                self.xg, self.ids_per_cls, [self.train_ids, _, _] = dataset.get_graph(tasks_to_retain = [t*2, (2*t)+1],node_ids=nodes, remove_edges=False)
                self.xg = self.xg.to(device='cuda:{}'.format(self.args.gpu))
                aux_features, aux_labels = self.xg.srcdata['feat'], self.xg.dstdata['label'].squeeze()
                self.train_ids = []
                if self.args.dataset == 'corafull':
                    for i in range(features[train_ids].shape[0]):
                        for j in range(aux_features.shape[0]):
                            if torch.equal(features[i], aux_features[j]):
                                self.train_ids.append(j)
                                break
                else:
                    x = [i for i in range(int(0.6*len(self.ids_per_cls[0])))]
                    y = [i+self.ids_per_cls[1][0] for i in range(int(0.6*len(self.ids_per_cls[1])))]
                    self.train_ids = x + y
                x = features.shape[0]

                for i in range(t):
                    buffer_length = []
                    buffer_length = [s+x for s in range(len(nodes[i]))]
                    self.train_ids.extend(buffer_length)
                    x = buffer_length[-1] + 1
            fast_weights = self.inner_update(g, features, train_ids, fast_weights, labels, t)
            aux_features, aux_labels = self.xg.srcdata['feat'], self.xg.dstdata['label'].squeeze()

            meta_loss, logits = self.meta_loss(self.xg, aux_features, self.train_ids, fast_weights, aux_labels, t)
            self.net.zero_grad()
            self.alpha_lr.zero_grad()
            meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2)
        torch.nn.utils.clip_grad_norm_(self.alpha_lr.parameters(), 2)
        self.opt.step()
        self.opt_lr.step()
        if self.ep == self.args.epochs:
            ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
            self.ep = 0
            self.push_to_buffer(ids_per_cls_train, g, features, dataset, labels, t)
        # no output is required from this function, it only serves to trained the model, and the model is the desired output.
    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        self.net.train()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        self.ep += 1
        self.net.zero_grad()
        fast_weights = None
        meta_losses = []
        if self.ep==1 and t!=0:
            nodes = [self.buffer_node_ids[i] for i in range(t)]
            subgraph_buffer, ids_buffer, [self.train_ids, valid_buffer, test_buffer] = dataset.get_graph(tasks_to_retain = [t*2, (2*t)+1],node_ids=nodes, remove_edges=False)
            aux_features, aux_labels = subgraph_buffer.srcdata['feat'], subgraph_buffer.dstdata['label'].squeeze()
            self.train_ids = []
            if self.args.dataset == 'corafull':
                for i in range(features[train_ids].shape[0]):
                    for j in range(aux_features.shape[0]):
                        if torch.equal(features[i], aux_features[j]):
                            self.train_ids.append(j)
                            break
            else:
                x = [i for i in range(int(0.6*len(ids_buffer[0])))]
                y = [i+ids_buffer[1][0] for i in range(int(0.6*len(ids_buffer[1])))]
                self.train_ids = x + y
            x = features.shape[0]

            for i in range(t):
                buffer_length = []
                buffer_length = [s+x for s in range(len(nodes[i]))]
                self.train_ids.extend(buffer_length)
                x = buffer_length[-1] + 1
            self.dataloader_buffer = dgl.dataloading.NodeDataLoader(subgraph_buffer, self.train_ids, args.nb_sampler, batch_size=args.batch_size, shuffle=True, drop_last=False)
        if t==0:
            for [current_input_nodes, current_output_nodes, current_blocks] in dataloader:
                n_nodes_current = current_input_nodes.shape[0]
                blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in current_blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                fast_weights = self.inner_update_batch(blocks, input_features, train_ids, fast_weights, output_labels, t)
                meta_loss, logits = self.meta_loss_batch(blocks, input_features, train_ids, fast_weights, output_labels, t)
                meta_losses.append(meta_loss)
                self.net.zero_grad()
                self.alpha_lr.zero_grad()
                meta = sum(meta_losses)/len(meta_losses)
        else:
            for [input_nodes, output_nodes, blocks], [current_input_nodes, current_output_nodes, current_blocks] in zip(self.dataloader_buffer, dataloader):
                n_nodes_current = current_input_nodes.shape[0]
                block = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
                current_block = [b.to(device='cuda:{}'.format(args.gpu)) for b in current_blocks]
                input_features = current_block[0].srcdata['feat']
                output_labels = current_block[-1].dstdata['label']
                fast_weights = self.inner_update_batch(current_block, input_features, train_ids, fast_weights, output_labels, t)
                inp_features = block[0].srcdata['feat']
                out_labels = block[-1].dstdata['label']
                meta_loss, logits = self.meta_loss_batch(block, inp_features, self.train_ids, fast_weights, out_labels, t)
                meta_losses.append(meta_loss)
                self.net.zero_grad()
                self.alpha_lr.zero_grad()
                meta = sum(meta_losses)/len(meta_losses)
                
        meta.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2)
        torch.nn.utils.clip_grad_norm_(self.alpha_lr.parameters(), 2)
        self.opt.step()
        self.opt_lr.step()
        if self.ep == self.args.epochs:
            ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
            self.ep = 0
            self.push_to_buffer(ids_per_cls_train, g, features, dataset, labels, t) 