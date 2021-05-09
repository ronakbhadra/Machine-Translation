#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv ("train.csv")
data.head(2)


# In[3]:


data = data[['hindi','english']]
data.head(2)


# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import spacy
import re
import time


# In[5]:


from indicnlp.tokenize import indic_tokenize  


# In[6]:


spacy_english =spacy.load("en_core_web_sm")


# In[7]:


def tokenize_hindi(text):
    return indic_tokenize.trivial_tokenize(text, lang='hi')

def tokenize_english(text):
    return [token.text for token in spacy_english.tokenizer(text)]


# In[8]:


sentence1 = """प्राचीन काल में विक्रमादित्य नाम के एक आदर्श राजा हुआ करते थे।
अपने साहस, पराक्रम और शौर्य के लिए  राजा विक्रम मशहूर थे। 
ऐसा भी कहा जाता है कि राजा विक्रम अपनी प्राजा के जीवन के दुख दर्द जानने के लिए रात्री के पहर में भेष बदल कर नगर में घूमते थे।"""
sentence2 = "Apple is looking at buying U.K. startup for $1 billion."


# In[ ]:





# In[9]:


import sys
from indicnlp import common

# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME=r"indic_nlp_library"

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES=r"indic_nlp_resources"

# Add library to Python path
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))

# Set environment variable for resources folder
common.set_resources_path(INDIC_NLP_RESOURCES)


# In[10]:


spacy_english =spacy.load("en_core_web_sm")


# In[11]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# In[12]:


def tokenize_hindi(text):
    return indic_tokenize.trivial_tokenize(text, lang='hi')

def tokenize_english(text):
    return word_tokenize(text)


# In[13]:


sentence1 = """प्राचीन काल में विक्रमादित्य नाम के एक आदर्श राजा हुआ करते थे।
अपने साहस, पराक्रम और शौर्य के लिए  राजा विक्रम मशहूर थे। 
ऐसा भी कहा जाता है कि राजा विक्रम अपनी प्राजा के जीवन के दुख दर्द जानने के लिए रात्री के पहर में भेष बदल कर नगर में घूमते थे।"""
sentence2 = "Apple is looking at buying U.K. startup for $1 billion."


# In[14]:


maximum=0
new_data = []
for i in range(len(data)):
    #print(i+1)
    text1 = data['hindi'][i]
    text2= data['english'][i]
    #print(text2)
        
    text1=re.sub("[*_|$@&:()\[\]~\{\}<>-]","",text1)
    text1 = re.sub(r'\.+', ".", text1)
    text1 = text1.replace('"', "")
    text1 = text1.replace('♪', "")
    text1 = text1.replace('♫', "")
    
    
    
    
    text2=re.sub("[*_|$@&:()\[\]~\{\}<>-]","",text2)
    text2 = re.sub(r'\.+', ".", text2)
    text2 = text2.replace('"', "")
    text2 = text2.replace('♪', "")
    text2 = text2.replace('♫', "")
    #print(text1)
    #print()
    #print(text2)

    x = tokenize_hindi(text1)
    y = tokenize_english(text2)
    if len(text1)>0 and len(text2)>0:
        if len(text1)==1:
            print("hello i=",i)
            print(text1)
        if len(x)<=70 and len(y)<=70:
            line=[text1,text2]
            new_data.append(line)
    
    if len(x)>maximum:
        maximum=len(x)
        
print("maximum length = ",maximum)


# In[15]:


text = "hello how are you"


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


len(new_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


df = pd.DataFrame(new_data, columns = ['hindi', 'english'])


# In[18]:


len(df)


# In[19]:


df


# In[20]:


#df.sort_values(by=['hindi'], inplace=True)


# In[21]:


s = df.hindi.str.len().sort_values().index
print (s)


# In[22]:


df1 = df.reindex(s)
df1 = df1.reset_index(drop=True)
print (df1)


# In[ ]:





# In[23]:


df1


# In[24]:


data=df1


# In[ ]:





# In[25]:


#temp = "As children, that's how we learn to differentiate ourselves in the world -- through touch."
temp = "It pretends we can isolate ourselves and our n..."
#temp = "(Laughter) W#h@o am {I} to con---tr*ad:ict<> m_y [baby] gi|rl? ~()........."


# In[26]:


x = "[@_!#$%^&*()<>/|}][{~:]"
temp=re.sub("[*_|$@&:()\[\]~\{\}<>-]","",temp)
temp = temp.replace('"', "")
#temp = temp.replace('-', "")
temp = re.sub(r'\.+', ".", temp)
temp


# In[27]:



temp = re.sub(r'\.+', ".", temp)


# In[ ]:





# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


train_data, val_data = train_test_split(data, test_size=0.1,shuffle=False)


# In[30]:


print("Dataset = ",len(data))
print("Train Dataset = ",len(train_data))
print("Validation Dataset = ",len(val_data))


# In[31]:


train_data


# In[32]:


val_data


# In[33]:


import collections
from collections import defaultdict


# In[34]:


hindi_vocab = defaultdict(lambda: 0)


# In[35]:


english_vocab = defaultdict(lambda: 0)


# In[36]:


hindi_vocab['<sos>']=2
hindi_vocab['<eos>']=3
hindi_vocab['<pad>']=1
hindi_vocab['<unk>']=0


# In[37]:


english_vocab['<sos>']=2
english_vocab['<eos>']=3
english_vocab['<pad>']=1
english_vocab['<unk>']=0


# In[38]:


len(hindi_vocab)


# In[39]:


hindi_freq = defaultdict(lambda: 0)
english_freq = defaultdict(lambda: 0)


# In[ ]:





# In[40]:


def build_freq_hindi(freq,data):
    for lines in data:
        for word in tokenize_hindi(lines):
            if word not in freq:
                freq[word]=1
            else:
                freq[word]=freq[word]+1


# In[41]:


def build_freq_english(freq,data):
    for lines in data:
        for word in tokenize_english(lines):
            if word not in freq:
                freq[word]=1
            else:
                freq[word]=freq[word]+1


# In[ ]:





# In[42]:


build_freq_hindi(hindi_freq,data['hindi'])


# In[43]:


build_freq_english(english_freq,data['english'])


# In[ ]:





# In[44]:


count =0
for item in hindi_freq.items():
    key=item[0]
    value=item[1]
    if(value>=2):
        count+=1
        hindi_vocab[key]=len(hindi_vocab)
print(count)


# In[45]:


hindi_vocab


# In[46]:


len(hindi_vocab)


# In[47]:


for item in english_freq.items():
    key=item[0]
    value=item[1]
    if(value>=2):
        english_vocab[key]=len(english_vocab)


# In[ ]:





# In[ ]:





# In[48]:


hindi_reverse = {value : key for (key, value) in hindi_vocab.items()}


# In[49]:


english_reverse = {value : key for (key, value) in english_vocab.items()}


# In[ ]:





# In[50]:


english_vocab['both']


# In[51]:


word = "सालवाडोर"
word_id = hindi_vocab.get(word)
print(word_id)


# In[52]:


hindi_vocab[word]


# In[53]:


len(hindi_vocab)


# In[54]:


print(f"Hindi vocabulary size: {len(hindi_vocab)}")
print(f"English vocabulary size: {len(english_vocab)}")


# In[55]:


hindi = "hindi"
english = "english"


# In[56]:


def line_to_tensor(line,language):
    if language=='hindi':
        tensor = []
        tensor.append(2)
        tokens = tokenize_hindi(line)
        for word in tokens:
            word_id = hindi_vocab[word]
            tensor.append(word_id)
        tensor.append(3)
        return tensor
    elif language=='english':
        tensor = []
        tensor.append(2)
        tokens = tokenize_english(line)
        for word in tokens:
            word_id = english_vocab[word]
            tensor.append(word_id)
        tensor.append(3)
        return tensor


# In[57]:


def data_generator(batch_size,data,line_to_tensor=line_to_tensor):
    
    index = 0
    
    # initialize the list that will contain the current batch
    cur_batch = []
    
    # count the number of lines in data_lines
    num_lines = len(data)
    
    # create an array with the indexes of data_lines that can be shuffled
    lines_index = [*range(num_lines)]
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    while True:
        
        # if the index is greater or equal than to the number of lines in data_lines
        if index>=num_lines:
            # then reset the index to 0
            index = 0
            
        # get a line at the `lines_index[index]` position in data_lines
        line = data.iloc[lines_index[index]]

        
        # append the line to the current batch
        cur_batch.append(line)
            
        # increment the index by one
        index += 1
        
        # if the current batch is now equal to the desired batch size
        if len(cur_batch)==batch_size:
            
            input_batch = []
            target_batch = []
            batch = []
            mask = []
            
            # go through each line (li) in cur_batch
            for li in cur_batch:
                # convert the line (li) to a tensor of integers
                input_tensor = line_to_tensor(li['hindi'],hindi)
                target_tensor = line_to_tensor(li['english'],english)
                
 
                
                # append the padded tensor to the batch
                input_batch.append(input_tensor)
                target_batch.append(target_tensor)
         
            
            
            row_lengths = []
            for row in input_batch:
                row_lengths.append(len(row))
                
            max_length = max(row_lengths)
            
            for row in input_batch:
                while len(row) < max_length:
                    row.append(1)
            
            for row in target_batch:
                row_lengths.append(len(row))
            
            max_length = max(row_lengths)
            
            for row in target_batch:
                while len(row) < max_length:
                    row.append(1)
            
            input_batch=np.array(input_batch)
            target_batch=np.array(target_batch)
            
            
            
            input_batch=np.transpose(input_batch)
            target_batch=np.transpose(target_batch)
            
            #batch = [input_batch,target_batch]
            
            input_batch=torch.tensor(input_batch)
            target_batch=torch.tensor(target_batch)
            
            
            batch=[input_batch,target_batch]
            
    
            yield batch
            
            # reset the current batch to an empty list
            cur_batch = []
            


# In[58]:


batch_size=64


# In[59]:


def train_generator(batch_size):
    return data_generator(batch_size,train_data,line_to_tensor)


# In[60]:


def val_generator(batch_size):
    return data_generator(batch_size,val_data,line_to_tensor)


# In[61]:


train_iterator = train_generator(batch_size)
val_iterator = val_generator(batch_size)


# In[ ]:





# In[ ]:





# In[62]:


next(train_iterator)


# In[63]:


batch=next(train_iterator)


# In[64]:


batch[0].shape[1]


# In[65]:


next(train_iterator)


# In[ ]:





# In[66]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:





# In[ ]:





# In[ ]:





# In[67]:


class Embedder(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        #super(Embedder,self).__init__()
        self.embed=nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        #print("x shape=",x.shape)
        #print(x)
        #print(self.embed)
        return self.embed(x)


# In[ ]:





# In[ ]:





# In[68]:


class PosEncoder(nn.Module):
    def __init__(self,d_model,max_seq_len=300):
        #super().__init__()
        super(PosEncoder,self).__init__()
        self.d_model=d_model
        
        pe=torch.zeros(max_seq_len,d_model)
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos,i]=math.sin(pos/(1000**((2*i)/d_model)))
                
                pe[pos,i+1]=math.cos(pos/(1000**((2*i)/d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        
        #print("hello")
        
        x = x*math.sqrt(self.d_model)
        
        #print("x = ",x.shape)
        #print("seq length = ",x.size(1))
        
        seq_len = x.size(1)
        
        x = x+ Variable(self.pe[:,:seq_len],requires_grad=False)
        
        return x


# In[69]:


def create_masks(input_seq,target_seq):
    input_pad = 1
    input_mask = (input_seq != input_pad).unsqueeze(1)
    
    target_pad = 1

    target_mask = (target_seq != input_pad).unsqueeze(1)

    size = target_seq.size(1)
    
    #print("input_seq = ",input_seq.shape)
    #print("target_seq = ",target_seq.shape)
    #print("seq_len for matrix = ",size)

    nopeak_mask = np.triu(np.ones((1,size,size)),k=1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask)==0)
    
    #print("Target mask = ",target_mask.shape)
    #print("nopeak_mask = ",nopeak_mask.shape)

    target_mask = target_mask & nopeak_mask
    
    #print("input_mask = ",input_mask.shape)
    #print("target_mask = ",target_mask.shape)
    
    return input_mask,target_mask
    


# In[70]:


input_seq = batch[0].transpose(0,1)
input_pad = 1

input_mask = (input_seq != input_pad).unsqueeze(1)


# In[71]:


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy


# In[72]:


target_seq = batch[0].transpose(0,1)
target_pad = 1

target_mask = (input_seq != input_pad).unsqueeze(1)

size = target_seq.size(1)

nopeak_mask = np.triu(np.ones((1,size,size)),k=1)
nopeak_mask = Variable(torch.from_numpy(nopeak_mask)==0)

target_mask = target_mask & nopeak_mask


# In[ ]:





# In[73]:


class MultiHeadAttention(nn.Module):
    def __init__(self,heads,d_model,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        
        self.d_model = d_model
        
        self.d_k = d_model//heads
        self.h = heads
        
        self.q_linear=nn.Linear(d_model,d_model)
        self.v_linear=nn.Linear(d_model,d_model)
        self.k_linear=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
        self.out=nn.Linear(d_model,d_model)
    
    def forward(self,q,k,v,mask = None):
        
        bs = q.size(0)
        
        k = self.k_linear(k).view(bs,-1,self.h,self.d_k)
        q = self.q_linear(q).view(bs,-1,self.h,self.d_k)
        v = self.v_linear(v).view(bs,-1,self.h,self.d_k)
        
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        #print("k type=",type(k))
        #print("K shape = ",k.shape)
        
        scores = attention(q,k,v,self.d_k,mask,self.dropout)
        
        concat = scores.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        
        output = self.out(concat)
        
        return output
        
        


# In[74]:


def attention(q,k,v,d_k,mask=None,dropout=None):
    
    scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)
    
    scores = F.softmax(scores,dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    
    output = torch.matmul(scores,v)
    
    return output


# In[ ]:





# In[75]:


class FeedForward(nn.Module):
    def __init__(self,d_model,hidden_size=1024,dropout=0.1):
        super(FeedForward,self).__init__()
        
        self.linear_1=nn.Linear(d_model,hidden_size)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(hidden_size,d_model)
    
    def forward(self,x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# In[76]:


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super(Norm,self).__init__()
    
        self.size = d_model
        
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))/ (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# In[ ]:





# In[77]:


class EncoderLayer(nn.Module):
    def __init__(self,d_model,heads,dropout=0.1):
        super(EncoderLayer,self).__init__()
        
        self.norm_1=Norm(d_model)
        self.norm_2=Norm(d_model)
        
        self.attention=MultiHeadAttention(heads,d_model)
        
        
        
        self.ff=FeedForward(d_model)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)
        
        #self.LSTM = nn.LSTM(d_model,256,1,bidirectional=False,dropout = 0.1)
        
        #self.linear = nn.Linear(1*1024,256)
        
        #print("ïnside encoder ",self.LSTM)
    
    def forward(self,x,mask):
        x2=self.norm_1(x)
        x=x+self.dropout_1(self.attention(x2,x2,x2,mask))
        
        #print("ïnside encoder layer x type = ",type(x))
        
        #print("ïnside encoder layer x shape = ",x.shape)
        
        
        #outputs, (hidden_state, cell_state) = self.LSTM(x)
        
        
        
        #print("output size=",outputs.shape)
        
        
        #y = self.linear(outputs)
        
        #z=x+outputs
        
        #print("y=",y.shape)
        
        x2=self.norm_2(x)
        x=x+self.dropout_2(self.ff(x2))
        
        return x


# In[ ]:





# In[78]:


class DecoderLayer(nn.Module):
    def __init__(self,d_model,heads,dropout=0.1):
        super(DecoderLayer,self).__init__()
        
        self.norm_1=Norm(d_model)
        self.norm_2=Norm(d_model)
        self.norm_3=Norm(d_model)
        
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)
        self.dropout_3=nn.Dropout(dropout)
        
        self.attention_1=MultiHeadAttention(heads,d_model)
        self.attention_2=MultiHeadAttention(heads,d_model)
        
        self.ff = FeedForward(d_model)
        
    def forward(self,x,encoder_output,src_mask,trg_mask):
        
        x2=self.norm_1(x)
        x=x+self.dropout_1(self.attention_1(x2,x2,x2,trg_mask))
        x2=self.norm_2(x)
        x=x+self.dropout_2(self.attention_2(x2,encoder_output,encoder_output,src_mask))
        x2=self.norm_3(x)
        x=x+self.dropout_3(self.ff(x2))
        
        return x


# In[ ]:





# In[79]:


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# In[ ]:





# In[80]:


class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,Embedder,PosEncoder,Norm,get_clones):
        super().__init__()
        #print("check 2 inside Encoder")
        self.N=N
        self.embed=Embedder(vocab_size,d_model)
        #print(self.embed)
        self.pe=PosEncoder(d_model)
        self.layers=get_clones(EncoderLayer(d_model,heads),N)
        self.norm=Norm(d_model)
        #print("check 2 inside Encoder")
        
    def forward(self,src,mask):
        #print("check 2 inside EncoderA")
        #print("check 2 inside EncoderA*******")
        #print("src shape which will be embedded= ",src.shape)
        #print(src)
        x=self.embed(src)
        #print("check 2 inside EncoderA*******")
        x=self.pe(x)
        for i in range(self.N):
            x=self.layers[i](x,mask)
        #print("check 2 inside EncoderA")
        return self.norm(x)


# In[81]:


d_model = 256
heads = 8
N = 6
src_vocab = len(hindi_vocab)
trg_vocab = len(english_vocab)


# In[82]:


len(hindi_vocab)


# In[83]:


encoder = Encoder(len(hindi_vocab),d_model,N,heads,Embedder,PosEncoder,Norm,get_clones)
#print(encoder)


# In[84]:


class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,Embedder,PosEncoder,Norm,get_clones):
        super(Decoder,self).__init__()
        #print("check 3 inside Decoder")
        self.N=N
        self.embed=Embedder(vocab_size,d_model)
        print(self.embed)
        self.pe=PosEncoder(d_model)
        self.layers=get_clones(DecoderLayer(d_model,heads),N)
        self.norm=Norm(d_model)
        #print("check 3 inside Decoder")
    
    def forward(self,trg,encoder_output,src_mask,trg_mask):
        #print("check 3 inside DecoderA")
        x=self.embed(trg)
        x=self.pe(x)
        for i in range(self.N):
            x=self.layers[i](x,encoder_output,src_mask,trg_mask)
        #print("check 3 inside DecoderA")
        return self.norm(x)


# In[85]:


decoder = Decoder(len(english_vocab),d_model,N,heads,Embedder,PosEncoder,Norm,get_clones)
#print(decoder)


# In[86]:


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads,encoder,decoder):
        #def __init__(self,Encoder_LSTM,Decoder_LSTM):
        super(Transformer,self).__init__()
        #print("check 1 inside TransformerA")
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out = nn.Linear(d_model, trg_vocab)
        #print("check 1 inside TransformerA")
    def forward(self, src, trg, src_mask, trg_mask):
        #print("check 1 inside TransformerB")
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


# In[ ]:





# In[87]:


model = Transformer(src_vocab, trg_vocab, d_model, N, heads,encoder,decoder)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


# In[88]:


#print(model)


# In[ ]:





# In[89]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')


# In[90]:


import math
import copy


# In[91]:


optim = torch.optim.Adam(model.parameters(), lr=0.0003)


# In[ ]:





# In[92]:


def train_model(epochs, print_every=100):
    
    model.train()
    
    start_time = time.time()
    
    total_loss = 0
    
    counter=0
    epoch_loss=0
    
    for epoch in range(epochs):
       
        for i in range(int(len(train_data)/int(batch_size))+1):
            
            
            counter=counter+1
        
            batch = next(train_iterator)
        
            source = batch[0]
            target = batch[1]
            
            source,target=source.type(torch.LongTensor),target.type(torch.LongTensor)
            
            src = source.transpose(0,1)
            trg = target.transpose(0,1)
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next
            
            trg_input = trg[:, :-1]
            
            # the words we are trying to predict
            
            targets = trg[:, 1:].contiguous().view(-1)
            
            # create function to make masks using mask code above
            
            src_mask, trg_mask = create_masks(src, trg_input)
            
            preds = model(src, trg_input, src_mask, trg_mask)
            
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),targets, ignore_index=target_pad)
            loss.backward()
            optim.step()
            
            total_loss += loss.data
            epoch_loss += loss.data
            
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("iter = %d, loss = %.3f" % (i + 1, loss_avg))
                total_loss = 0
    
    print("")
    end_time=time.time()
    loss_avg=epoch_loss/counter
    print('Train Loss {:.4f}'.format(loss_avg))
    print("TIME = ",end_time-start_time)
        


# In[93]:


len(val_data)


# In[94]:


def test(epoch):
    model.eval()
    
    total_loss = 0
    
    start_time = time.time()
    
    counter = 0
    
    for epoch in range(epoch):
       
        for i in range(int(len(val_data)/int(batch_size))+1):
            
            counter+=1
            print("Iteration = ",counter)
            
            
            
            batch = next(val_iterator)
        
            source = batch[0]
            target = batch[1]
            
            source,target=source.type(torch.LongTensor),target.type(torch.LongTensor)
            
            src = source.transpose(0,1)
            trg = target.transpose(0,1)

            # the Target sentence we input has all words except
            # the last, as it is using each word to predict the next

            trg_input = trg[:, :-1]

            # the words we are trying to predict

            targets = trg[:, 1:].contiguous().view(-1)

                    # create function to make masks using mask code above

            src_mask, trg_mask = create_masks(src, trg_input)
            
            #print(src)

            preds = model(src, trg_input, src_mask, trg_mask)

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),targets, ignore_index=target_pad)

            total_loss += loss.data
            
            
        loss_avg = total_loss /(counter)
        
        print("")
        end_time=time.time()
        print('Val Loss {:.4f}'.format(loss_avg))
        print("TIME = ",end_time-start_time)
        print("")


# In[97]:


no_epoch=5


# In[98]:


for i in range(no_epoch):
    print("EPOCH = ",i+1)
    train_model(1, print_every=1)
    test(1)


# In[ ]:


import pickle
filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))


# In[ ]:


loaded_model


# In[ ]:



import pickle
  
# Save the trained model as a pickle string.
saved_model = pickle.dumps(model)
  
# Load the pickled model
test_model = pickle.loads(saved_model)
  
# Use the loaded pickled model to make predictions
#knn_from_pickle.predict(X_test)


# In[106]:


PATH = "normal_transformer_entire_model.pt"

# Save
torch.save(model, PATH)

# Load
model = torch.load(PATH)
model.eval()


# In[ ]:


val_data


# In[ ]:


model = loaded_model


# In[ ]:


val_data['hindi'][81857]


# In[ ]:


test=train_data['hindi'].iloc[0]
test


# In[ ]:


src = tokenize_hindi(test)
src


# In[ ]:


src = Variable(torch.LongTensor([[hindi_vocab[tok] for tok in src]]))


# In[ ]:


src


# In[100]:


def translate(model, src, max_len=30, custom_string=True):
    model.eval()

    if custom_string:
        src = tokenize_hindi(src)  # .transpose(0,1)
        # sentence = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in sentence]])) #.cuda()
        if device.type == 'cuda':
            src = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in src]])).cuda()
        else:
            src = Variable(torch.LongTensor([[hindi_vocab[tok] for tok in src]]))
        src_mask = (src != hindi_vocab['<pad>']).unsqueeze(-2)

    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([english_vocab['<sos>']])
    for i in range(1, max_len):

        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype("uint8")
        if device.type == 'cuda':
            trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()
        else:
            trg_mask = Variable(torch.from_numpy(trg_mask) == 0)
        out = model.out(model.decoder(outputs[:i].unsqueeze(0),
                                      e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == english_vocab['<eos>']:
            #print("EOS is generated")
            break
    temp = len('<sos> ')
    return ' '.join([english_reverse[int(ix)] if int(ix)!=0 else '<unk>' for ix in outputs[:i]])[temp:]


# In[ ]:


test1=train_data['hindi'].iloc[1]
test1


# In[ ]:


translate(model,test1)


# In[ ]:


translate(loaded_model,test1)


# In[103]:


check_data = pd.read_csv ("testhindistatements.csv")


# In[104]:


check_data=check_data['hindi']


# In[ ]:


#f = open("results.txt", "w",encoding="utf-8")
for i in range(len(check_data)):
    print(i+1)
    test1= check_data.iloc[i]
    #test2=train_data['english'].iloc[i]
    #print(test2)
    x = translate(model,test1)
    #print()
    print(x)
    #print(len(x))
 #   f.write(x+"\n")
    print()
#f.close()


# In[ ]:


#f = open("results.txt", "w",encoding="utf-8")
for i in range(len(train_data)):
    print(i+1)
    test1= train_data['hindi'].iloc[i]
    test2=train_data['english'].iloc[i]
    print(test2)
    x = translate(model,test1)
    #print()
    print(x)
    #print(len(x))
 #   f.write(x+"\n")
    print()
#f.close()


# In[ ]:


f = open("results5.txt", "w",encoding="utf-8")
for i in range(len(check_data)):
    print(i+1)
    test1=check_data.iloc[i]
    print(test1)
    x = translate(model,test1)
    print(x)
    f.write(x+"\n")
    print()
f.close()


# In[102]:


y = pd.read_csv ("testhindieng.csv")


# In[ ]:


y=y['english']


# In[105]:


f = open("transformer_results.txt", "w",encoding="utf-8")
for i in range(len(check_data)):
    print(i+1)
    test1=check_data.iloc[i]
    print(test1)
    x = translate(model,test1)
    print(x)
    f.write(x+"\n")
    print()
f.close()


# In[ ]:





# In[ ]:


check_data.iloc[1]


# In[ ]:


f = open("results.txt", "w",encoding="utf-8")
for i in range (len(check_data)):       #len(check_data)
    #test_sentence_en = check_data['english'][i]
    test= check_data.iloc[i]
    #print("Test Sentense English: ",test_sentence_en)
    print(i+1)
    print("Test Sentense Hindi: ",test)
    print("Translated Sentence: ",translate(model,test))
    x = translate(model,test)
    f.write(x+"\n")
    print()
f.close()


# In[ ]:


for i in range(len(check_data)):
    test=check_data.iloc[i]
    print(translate(model,test))


# In[ ]:





# In[107]:


for i in range(len(check_data)):
    print(i+1)
    test1=check_data.iloc[i]
    print(test1)
    x = translate(model,test1)
    print(x)
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




