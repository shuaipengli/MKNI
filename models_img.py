from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from matplotlib.pyplot import flag
import torch
from torch import embedding, gather, nn
import pickle
import torch.nn.functional as F
import numpy as np
from config import alpha,beta,rel_,img_embedding,image_emb,alignment_matrix,alignment_idx


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor, all_scores: torch.Tensor):
        pass
    @abstractmethod
    def compute_all_rel(self):
        pass
    @abstractmethod
    def predict_h(self, queries: torch.Tensor):
        pass
    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1, miss: str = 'tail',
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """

        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))

        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0

                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    ''' scores_str:结构化信息的scores ； score_img:图片信息的scores '''
                    if miss == 'tail':
                        rhs = self.get_rhs(c_begin, chunk_size, flag = 'str')
                        q = self.get_queries(these_queries, flag = 'str')
                        # q @ rhs就是矩阵q和rhs的相乘 (群点击，每个元素相乘再求和)
                        score_str = q @ rhs
                        # # 对数据进行标准化 如[1,2,3] -> [1/sqrt(1**2+2**2+3**2), 2/sqrt(1**2+2**2+3**2), 3/sqrt(1**2+2**2+3**2)]

                        rhs_g = self.get_rhs(c_begin, chunk_size, flag = 'gather')
                        q_g = self.get_queries(these_queries, flag = 'gather')
                        # q @ rhs就是矩阵q和rhs的相乘 (群点击，每个元素相乘再求和)
                        score_gather = q_g @ rhs_g

                        # img_embeddings = self.img_vec
                        # rhs_img = F.normalize(img_embeddings[these_queries[:,2]], p=2, dim=1)
                        # q_img = F.normalize(img_embeddings, p=2, dim=1).transpose(0, 1)
                        # score_img=rhs_img @ q_img
                        
                        img_embeddings = self.img_vec
                        # r_h = score_str.argmax(dim=1)
                        rhs_img = F.normalize(img_embeddings[these_queries[:,0]], p=2, dim=1)
                        q_img = F.normalize(img_embeddings, p=2, dim=1).transpose(0, 1)
                        score_img=rhs_img @ q_img
                        # score_img = self.rel_[these_queries[:,1]].unsqueeze(1) *  score_img
                        # scores = (1-beta) * score_str + beta * score_gather + score_img
                        scores = (1-beta) * score_str + beta * score_gather
                        targets = self.score(these_queries)
                    elif miss == 'head':
                        all_scores = self.compute_all_rel()
                        scores, targets = self.predict_h(these_queries,all_scores)

                    for i, query in enumerate(these_queries):
                        # 找到查询的 (h , r , t)的id
                        # if miss == 'tail': 
                        filter_out = filters[(query[0].item(), query[1].item())]  # filters[h, r], 即取 (h,r, )
                        filter_out += [queries[b_begin + i, 2].item()] # [h,r,t]
                        # elif miss== 'head':
                        #     filter_out = filters[(query[2].item(), query[1].item())]  # filters[h, r], 即取 (h,r, )
                        #     filter_out += [queries[b_begin + i, 0].item()] # [h,r,t]

                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                
                        # targets[i] = torch.tensor(0.) + targets[i]
                        # targets[i] = score_img[i,queries[b_begin + i,2].item()] + targets[i]
                    ''' 如果得分比目标的大就相当于排名要靠后一点： target:batch_size × 1; scores:batch_size × entitys_number '''

                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >=targets).float(), dim=1
                    ).cpu()
                    

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


''' 总的model：
    Args:  
        img_info: 计算完成之后的Img编码向量，可以使用vit、Resnet和Faster R-CNN
        sig_alpha: Img向量与Structure向量融合的权重： alpha*Emb_img + (1-alpha)*Emb_stru, alpha值的设定怎么样更好点？如何评价图片对实体的贡献度？ 可以预先设置一个贡献度，后面将其加入可学习的参数阵列
'''
class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-1,
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank


        self.r_embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * rank, sparse=False),
            nn.Embedding(sizes[1], 2 * rank, sparse=False),
        ])
        

        self.r_embeddings[0].weight.data *= init_size
        self.r_embeddings[1].weight.data *= init_size


        ''' 保持与Structure的维度一致 '''
        self.img_dimension = 1000
        self.img_info = pickle.load(open(img_embedding, 'rb'))
        self.img_vec = torch.from_numpy(self.img_info).float()
        
        self.rel_ = torch.from_numpy(np.array(pickle.load(open(rel_, 'rb')))).float().cuda()

        ''' 初始化一个tensor长度为 [img_dimension, 2*rank]; 并加入参数； 投影矩阵？？？!!! 就是图片的投影矩阵，投影到跟entity Structure空间的矩阵'''
        self.post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.post_mats)
        
        '''读取图片与结构的对齐矩阵alignment_matrix 和 对应的一阶邻居节点'''
        self.alignment_matrix = pickle.load(open(alignment_matrix, 'rb'))            
        self.alignment_idx = pickle.load(open(alignment_idx, 'rb'))   
        # 对齐求聚合后的向量
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)  # [14951, 2000] 如果是Constant就不需要考虑MRP值 得到最后的alpha就是不断训练适应 到的值，首先alpha的设定为0.7
        gather_ = self.alpha * self.img_vec.mm(self.post_mats) + (1-self.alpha) * self.r_embeddings[0].weight
        self.img_vec = self.img_vec.cuda() 
        # embeddings = self.r_embeddings[0].weight.cpu().detach().numpy()
        embeddings = gather_.cpu().detach().numpy()
        gather_scores = [0]*embeddings.shape[0]

        for i in range(embeddings.shape[0]):
            # 如果没有图片的对应图，则直接等于原embedding
            if isinstance(self.alignment_idx[i], int):
                gather_scores[i] = embeddings[i]
            else:
                emb = embeddings[(self.alignment_idx[i][:])]
                gather_scores[i] = np.sum(np.dot(self.alignment_matrix[i]/np.sum(self.alignment_matrix[i]),emb),0)
                # gather_scores[i] = np.sum(np.dot(self.alignment_matrix[i],emb),0)

        self.gather_scores = torch.from_numpy(np.array(gather_scores)).float().cuda()
        self.gather_scores = nn.Parameter(self.gather_scores, requires_grad=True)




    # 计算score 得到Loss
    def score(self, x):
        # 向量投影， 将img投影到stru空间

        embedding = self.r_embeddings[0].weight

        lhs = embedding[(x[:, 0])]
        rel = self.r_embeddings[1](x[:, 1])
        rhs = embedding[(x[:, 2])]

        # 得到的值处于(-1,+1)之间
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        score_str=torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

        ''' 通过与图片对齐的聚合网络提取邻居信息 '''
        # 计算聚合后的分数
        lhs_gather = self.gather_scores[(x[:, 0])]
        lhs_gather = lhs_gather[:, :self.rank], lhs_gather[:, self.rank:]
        rhs_gather = self.gather_scores[(x[:, 2])]
        rhs_gather = rhs_gather[:, :self.rank], rhs_gather[:, self.rank:]

        score_gather=torch.sum(
            (lhs_gather[0] * rel[0] - lhs_gather[1] * rel[1]) * rhs_gather[0] +
            (lhs_gather[0] * rel[1] + lhs_gather[1] * rel[0]) * rhs_gather[1],
            1, keepdim=True
        )
        # # 计算聚合分数和str分数
        scores = (1-beta) * score_str + beta * score_gather
        # scores = score_gather
        # scores = score_str 
        return scores
            
    # 前向传播
    def forward(self, x):

        embedding = self.r_embeddings[0].weight

        lhs = embedding[(x[:, 0])]
        rel = self.r_embeddings[1](x[:, 1])
        rhs = embedding[(x[:, 2])]

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        # # 计算基于structure的分数
        str_scores = embedding
        str_scores = str_scores[:, :self.rank], str_scores[:, self.rank:]
        score_str = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ str_scores[0].transpose(0, 1) +\
              (lhs[0] * rel[1] + lhs[1] * rel[0]) @ str_scores[1].transpose(0, 1)
        

        score_str_predict_h = (rhs[0] * rel[0] - rhs[1] * rel[1]) @ str_scores[0].transpose(0, 1) +\
              (rhs[0] * rel[1] + rhs[1] * rel[0]) @ str_scores[1].transpose(0, 1)
        ''' 通过与图片对齐的聚合网络提取邻居信息 '''

        # 计算聚合后的分数
        lhs_gather = self.gather_scores[(x[:, 0])]
        lhs_gather = lhs_gather[:, :self.rank], lhs_gather[:, self.rank:]
        rhs_gather = self.gather_scores[(x[:, 2])]
        rhs_gather = rhs_gather[:, :self.rank], rhs_gather[:, self.rank:]

        gather_score = self.gather_scores[:, :self.rank], self.gather_scores[:, self.rank:]
        score_gather = (lhs_gather[0] * rel[0] - lhs_gather[1] * rel[1]) @ gather_score[0].transpose(0, 1) +\
              (lhs_gather[0] * rel[1] + lhs_gather[1] * rel[0]) @ gather_score[1].transpose(0, 1)



        # 计算聚合分数和str分数
        scores = (1-beta) * score_str + beta * score_gather
        # scores = score_gather 
        # scores = score_str
        

        return scores,score_str_predict_h,(
                        torch.sqrt(lhs[0]** 2 + lhs[1]** 2),
                        torch.sqrt(rel[0]** 2 + rel[1]** 2),
                        torch.sqrt(rhs[0]** 2 + rhs[1]** 2),
                        # embedding[(x[:, 0])],
                        # self.r_embeddings[1](x[:, 1]),
                        # embedding[(x[:, 2])]
                        # embedding[(x[:, 0])] + self.gather_scores[(x[:, 0])],
                        # 2*self.r_embeddings[1](x[:, 1]),
                        # embedding[(x[:, 2])] + self.gather_scores[(x[:, 2])]
                    )
        # return scores,(
        #                 torch.sqrt(((1-beta) *lhs[0] +  beta *lhs_gather[0])** 2 + ((1-beta) *lhs[1] +  beta *lhs_gather[1]) ** 2),
        #                 torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
        #                 torch.sqrt(((1-beta) *rhs[0] +  beta *rhs_gather[0])** 2 + ((1-beta) *rhs[1] +  beta *rhs_gather[1]) ** 2)
        #             )


    # 获取尾实体 从候选集中选取chunk_size个。 
    def get_rhs(self, chunk_begin: int, chunk_size: int, flag: str):
        embedding = self.r_embeddings[0].weight

        if flag=='gather':
            # gather_scores = embedding
            # for i in range(embedding.shape[0]):
            #     if isinstance(self.alignment_idx[i], int):
            #         continue
            #     emb = embedding[(self.alignment_idx[i][:])]
            #     gather_scores[i] = torch.sum(torch.tensor(self.alignment_matrix[i]).cuda().matmul(emb.double()),dim=0)
            embedding = self.gather_scores
        elif flag=='img':
            embedding = self.img_vec.mm(self.post_mats)

        return embedding[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)


    # 获取计算分数的q
    def get_queries(self, queries: torch.Tensor, flag: str):
        embedding = self.r_embeddings[0].weight

        if flag=='gather':
            # gather_scores = embedding
            # for i in range(embedding.shape[0]):
            #     if isinstance(self.alignment_idx[i], int):
            #         continue
            #     emb = embedding[(self.alignment_idx[i][:])]
            #     gather_scores[i] = torch.sum(torch.tensor(self.alignment_matrix[i]).cuda().matmul(emb.double()),dim=0)
            embedding = self.gather_scores
        elif flag=='img':
            embedding = self.img_vec.mm(self.post_mats)
        lhs = embedding[(queries[:, 0])]
        rel = self.r_embeddings[1](queries[:, 1])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],  # 比如l[0], r[0] 是structure的信息；l[1], r[1] 是image的信息
            lhs[0] * rel[1] + lhs[1] * rel[0]   # 将返回
        ], 1)

    
    # 替换头实体时的预测方法
    def predict_h(self, queries: torch.Tensor, all_scores: torch.Tensor):


        score_str = all_scores[queries[:,1]][:,queries[:,0]] # 500,14951
        target = score_str[queries[:,2]] #500, 1
        return score_str, target
    
    def compute_all_rel(self):
        str_emb = self.r_embeddings[0].weight
        lhs = str_emb
        rel = self.r_embeddings[1].weight[:self.sizes[1]//2] # 对所有的r取一半
        rhs = str_emb

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        all_scores = torch.zeros([self.sizes[1]//2, str_emb.shape[0], str_emb.shape[0]]) # 1345,14951，14951
        # 对每个关系建立
        for i in range(self.sizes[1]//2):
            all_scores[i] = ((lhs[0] * rel[0][i] - lhs[1] * rel[1][i]) @ rhs[0].transpose(0, 1) +\
                (lhs[0] * rel[1][i] + lhs[1] * rel[0][i]) @ rhs[1].transpose(0, 1))   # 所有都和t求分数
        return all_scores