import argparse
from typing import Dict
import os
# from cv2 import mean
import torch
from torch import optim

from datasets import Dataset
# from models import ComplEx
from models_img import ComplEx

from regularizers import F2, N3
from optimizers import KBCOptimizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


big_datasets = ['FB15K', 'WN', 'WN18','WN18RR', 'FB237', 'YAGO3-10','FB15K_1','FB15K_3','FB15K_5','FB15K_7','FB15K_15','FB15K_10','FB15K_20','FB15K_40','FB15K_60','FB15K_80','FB15K_100']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,default='FB15K',
    help="Dataset in {}".format(datasets)
)

models = ['ComplEx']
parser.add_argument(
    '--model', choices=models,default='ComplEx',
    help="Model in {}".format(models)
)

regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD', 'Adadelta']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=1, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0.005, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1.5e-4, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=0.05, type=float,
    help="Learning rate"
)
# lr /= [1 + (epoch-1)* lr_decay]
parser.add_argument(
    '--lr_decay', default=0.99, type=float,
    help="Learning rate decay: lr * lr_decay"
)

parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
args = parser.parse_args()

dataset = Dataset(args.dataset)
train_examples = torch.from_numpy(dataset.get_train().astype('int64'))
valid_examples = torch.from_numpy(dataset.get_valid().astype('int64'))

print(dataset.get_shape())
model = {
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
}[args.model]()

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

device = 'cuda'
model.to(device)



optim_method = {
    'Adadelta' : lambda:optim.Adadelta(model.parameters(), lr= args.learning_rate),
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)

def avg_both(mrs: Dict[str, float], mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    mr = (mrs['lhs'] + mrs['rhs']) / 2.
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.

    return {'MR': mr,'MRR': mrr, 'hits@[1,3,10]': h}

cur_loss = 0
curve = {'train': [], 'valid': []}
for e in range(args.max_epochs):
    # Train step
    model.train()

    cur_loss = optimizer.epoch(e, train_examples)
    # Valid step
    # valid_loss = optimizer.calculate_valid_loss(valid_examples)
    # print(valid_loss)
    # 查看对valid的训练结果
    if (e + 1) % args.valid == 0:
        model.eval()
        valid = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid']
        ]
        curve['valid'].append(valid)
        print("\t VALID : ", valid)
        results = dataset.eval(model, 'test', -1)
        print("TEST : ", results)
        print(avg_both(results[0], results[1],results[2]))

