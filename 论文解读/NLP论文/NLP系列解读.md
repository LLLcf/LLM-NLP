###  一.BERT 
[论文地址](https://arxiv.org/abs/1810.04805)
#### Tips
1. 基于Transformer中`Encoder-only`架构，采用双向编码，能够同时考虑句子的上下文信息

2. 预训练采取两个无监督任务(Masked LM 和 NSP)，其中MLM任务为完形填空，NSP任务为判断两个句子是否为连续
![alt text](./画图草稿/bert.png)
3. MLM任务对每个序列中的`WordPiece`标记进行15%的随机[Mask]，但这样会导致预训练MASK任务和下游微调任务无法对齐(微调过程中不存在[MASK]标记)。因此在进行随机[MASK]时，采用80%替换为[MASK]标记，10%随机替换为任意标记实现，10%保持原标记不变
4. NSP任务即为一个二分类任务来理解句子之间的关系，判断句子A和句子B(A->B)是否连续，预训练时，数据对构造一半为`(A->B IS_NEXT)`，另一半为`(A->B NOT_NEXT)`。通过使用分类层将 [CLS] token 的输出转换为 2×1 形状的向量来实现的，然后使用 SoftMax 计算第二句是否跟在第一句后面的概率

    `example: [CLS] [token] [token] [SEP] [token] [token] [SEP]`
    [CLS] token 表示整个句子的输出
5. 同时训练MLM和NSP任务，MLM 帮助 BERT 理解句子内的上下文，NSP 帮助 BERT 掌握句子对之间的联系
6. 模型架构
![alt text](./画图草稿/bert模型架构.png)
[CLS] 取最后一个encoder层的的第一个token输出，即[CLS]对应的token的向量结果。
`pooler_output层`获取的是经过线性变换和 Tanh 激活后的 [CLS] 向量
它通常用于下游任务，是整个输入序列的一个固定维度的语义表示

7. 激活函数
原始Transformer中FeedForward层使用的是RELU激活函数，google-BERT中采用的是`高斯误差线性单元(GELU)`激活函数。GELU可以看成RELU和Drpoout的组合体
GELU激活函数:
$$
GELU(x) = \Phi(x) * 1 x + (1 - \Phi(x)) * 0x = x \Phi(x)
$$
BERT源码中采用近似结果表示：
$$
GELU(x) = 0.5x(1 + tanh(\sqrt{2/\pi}(x+0.044715x^3)))
$$

8. Tokenizer
分词目的：常见的词在token列表中表示为单个token，而罕见的词被分解为多个subword tokens
当前tokenizer分词有word，sub-word，char level三种类型。传统分词如jieba分词等按照word级别进行分词，WordPiece分词为`sub-word`级别，BPE也是sub-word级别
sub-word分词解决word分词遇到的问题：
--    1.超大的词表(vocabulary size)，词表扩充爆炸
--    2.词表中存在大量相似的词
--    3.严重的未登录词(OOV)问题
sub-word分词解决char level分词遇到的问题：
--    1.文本序列变长，输入长度爆炸
--    2.无法对文本语义进行好的表征
![alt text](./画图草稿/wordpiece.png)

1. BERT Embedding 
![alt text](./画图草稿/bert_embedding.png)

   - Token Embedding: 对输入的词进行词编码，每个词映射到高维空间上
![alt text](./画图草稿/token_embedding.png)
   - Position Embedding: 考虑词的位置信息，不同位置的词语给予不同的编码，采用`可学习参数的嵌入方式`(与原始Transformer的绝对余弦位置编码不同)

   ```
   class LearnedPositionalEncoding(nn.Module):
       def __init__(self, seq_len, d_model):
           super(LearnedPositionalEncoding, self).__init__()
           self.position_embeddings = nn.Embedding(seq_len, d_model)

       def forward(self, input_tensor):
           seq_length = input_tensor.size(1)
           position_ids = torch.arange(seq_length, dtype=torch.long, device=input_tensor.device)
           position_ids = position_ids.unsqueeze(0).expand_as(input_tensor[:, :, 0])
           return self.position_embeddings(position_ids)
   ```
   - Token Type Embedding(Segment embedding): 用于区分不同句子的编码(0, 1结果表示两个不同的句子)
![alt text](./画图草稿/segment_embedding.png)
    BERT的inputs为三个embedding直接相加的输入

### 二.Roberta
[论文地址](https://arxiv.org/abs/1907.11692)

#### 与BERT相比的不同点
1. 更多的训练数据(原始BERT训练数据大约为16GB, RoBERTa训练数据额外多160GB数据)
2. 更大的训练Batch Size(原始BERT的Batch Size为256, RoBERTa的Batch Size为8k)
实验证明更大的Batch Size能够有更好的效果，同时训练Steps数减少
3. 取消NSP loss
4. BPE编码(Byte Pair Encoding)
![alt text](./画图草稿/bpe.png)
5. Whole Word Mask 和 动态 Mask
   - Whole Word Mask: 对整个词进行Mask，而不是对单词中的子词进行Mask
   - 动态 Mask: 每次向模型输入一个序列时都会生成新的掩码模式。与BERT不同，BERT是静态Mask，仅在数据预处理阶段进行一次Mask

### 三.Albert
[论文地址](https://arxiv.org/abs/1909.11942)

1. 拥有更少的参数量，但模型性能能够接近BERT的效果
2. 通过对Embedding嵌入的参数进行降维再升维，以达到减少参数量的目的，过去Embedding嵌入维度和Hidden Dim的维度相同，导致大量参数冗余(WordPiece嵌入旨在学习上下文无关的表示，而隐藏层嵌入旨在学习上下文相关的表示,因此，将WordPiece嵌入大小E与隐藏层大小H解耦，使我们能够根据建模需求更有效地利用总模型参数，这些需求规定了H>E)
$$O(V \times H) \rightarrow O(V \times E + E \times H)$$    
3. 跨层参数共享(即所有Transformer层之间共享相同的参数。这样，每一层的自注意力机制和前馈神经网络都使用相同的权重，从而显著减少了模型的参数量)
4. 句子顺序判断(SOP)，由于BERT之后的工作多数发现NSP训练对模型效果影响小或无影响，因此Albert模型去掉了NSP任务，选择SOP来判断句子连贯性，正负样本为句子交换前后的样本

    **note**：
    由于NSP任务的正样本来自两个相邻的segments，而负样本来自不同的segments。因此解决NSP任务可以通过：1.主题预测(语义信息)。2.连贯性预测(BERT的NSP任务本义)

### 四.GPT-1
[论文地址](https://arxiv.org/abs/2305.10435)
