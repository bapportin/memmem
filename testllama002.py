import argparse
import torch
import traceback
import transformers
import numpy as np
import collections
import pickle
import os
import time
import random
import numpy as np
import proxy
import torchao
import bitsandbytes
import math
import mlsh


class SharedBuffer:
    @staticmethod
    def get_buffer(name,initmethod):
        if not hasattr(SharedBuffer,name):
            setattr(SharedBuffer,name,initmethod())
        return getattr(SharedBuffer,name)

class JRMSNorm(torch.nn.Module):
    def __init__(self, orig):
        """
        JRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = orig.weight
        self.variance_epsilon = orig.variance_epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class JMLP(torch.nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.hidden_size = orig.hidden_size
        self.intermediate_size = orig.intermediate_size
        self.gate_proj = orig.gate_proj
        self.up_proj = orig.up_proj
        self.down_proj = orig.down_proj
        self.act_fn = orig.act_fn

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class DataCache:
    def __init__(self,seq_size=2048):
        self.data={}#layer->[]
        self.seq_size=seq_size
        self.mem_size=8
        self.mem_layers={i:mlsh.Tree() for i in [4,6,8]}
        

    def pushDown(self):
        pass

    def _update(self,layer,query):
        #query memory with queries rotated to seq_size and prepend results
        return [],[]

    def update(self,layer,query,key,value):
        #print("dcache",query.shape,key.shape,value.shape)
        
        if not layer.layer_idx in self.data:
            self.data[layer.layer_idx]=[]
        #split and rejoin on sequence dim
        for i in range(key.shape[1]):
            self.data[layer.layer_idx].append((key[:,i:i+1,:,:],value[:,i:i+1,:,:]))
        if len(self.data[layer.layer_idx])>self.seq_size:
            kh,vh=self.data[layer.layer_idx].pop(0)
            #move old data to the memory
            if layer.layer_idx in self.mem_layers:
                for k,v in zip(kh.view(-1,kh.shape[-1]),vh.view(-1,vh.shape[-1])):
                    self.mem_layers[layer.layer_idx].put(k,v,True)
        mklist,mvlist=self._update(layer,query)
        klist,vlist=[],[]
        for k,v in self.data[layer.layer_idx]:
            klist.append(k)
            vlist.append(v)
            #wlist.append(torch.zeros(1,1,1,1))
        return torch.cat(klist,1),torch.cat(vlist,1),mklist,mvlist
        

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        
        

class JAttention(torch.nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.layer_idx = orig.layer_idx
        self.attention_dropout = orig.attention_dropout
        self.hidden_size=orig.q_proj.in_features
        self.head_dim = orig.head_dim
        self.num_key_value_heads = orig.k_proj.out_features//self.head_dim
        self.num_key_value_groups = orig.num_key_value_groups
        self.is_causal = orig.is_causal

        key_size=self.head_dim

        self.q_proj = orig.q_proj
        self.k_proj = orig.k_proj
        self.v_proj = orig.v_proj
        self.o_proj = orig.o_proj
        if self.is_causal:
            def getMask():
                return torch.tril(torch.ones(1, 1, 4096, 4096))
            self.register_buffer(
                "mask", SharedBuffer.get_buffer("mask",getMask), persistent=False
            )
        def getInvFreq():
            return 1.0 / (10000 ** (torch.arange(0, key_size, 2, dtype=torch.int64).float() / key_size))
        self.register_buffer(
            "inv_freq",SharedBuffer.get_buffer("inv_freq",getInvFreq), persistent=False
        )
        def getCos():
            pids=torch.arange(0,4096)+8192
            freq=torch.cat([self.inv_freq,self.inv_freq],-1)
            return torch.cos(pids[None,:,None,None]*freq[None,None,None,:])
        self.register_buffer(
            "cos",SharedBuffer.get_buffer("cos",getCos), persistent=False
        )        
        def getSin():
            pids=torch.arange(0,4096)+8192
            freq=torch.cat([self.inv_freq,self.inv_freq],-1)
            return torch.sin(pids[None,:,None,None]*freq[None,None,None,:])
        self.register_buffer(
            "sin",SharedBuffer.get_buffer("sin",getSin), persistent=False
        )


    def doRotate(self,q,offset=0):
        #batch_size, seq_len,self.num_heads,self.key_size
        #print("doRotate",q.shape,offset)
        batch_size, seq_len,h,f=q.shape
        q1=q*self.cos[:,offset:offset+seq_len,:,:]+(rotate_half(q)*self.sin[:,offset:offset+seq_len,:,:])
        return q1

    def rotateTo(self,q,idx):
        #batch_size, seq_len,self.num_heads,self.key_size
        batch_size, seq_len,h,f=q.shape
        freq=torch.cat([self.inv_freq,self.inv_freq],-1)[None,None,None,:]*idx
        cos=torch.cos(freq)
        sin=torch.sin(freq)
        q1=q*cos+(rotate_half(q)*sin)
        return q1

    def forward(self,x,cache):
        batch_size, seq_len, _ = x.shape

        #print(0,x.shape)
        
        query=self.q_proj(x).view(batch_size, seq_len, -1, self.head_dim)
        
        key=self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value=self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        #print(1,x.shape,key.shape,value.shape,query.shape)

        #rotate to end of list
        query=self.doRotate(query,4096-query.shape[1])

        key,value,kext,vext=cache.update(self,query,key,value)

        #print(2,x.shape,key.shape,value.shape,query.shape,qext.shape,kext.shape)

        #rotate to end of list
        key=self.doRotate(key,4096-key.shape[1])

        #concatenate to beginning of key and value, we don't need to rotate extension, since they are at position 0, long in the past
        if len(kext)>0:
            key=torch.cat(kext+[key],1)
            value=torch.cat(vext+[value],1)

        #transpose for matmul
        query=query.transpose(1, 2)
        key=key.transpose(1, 2)
        value=value.transpose(1, 2)

        #repeat for grouped query attention
        key=torch.repeat_interleave(key, dim=1, repeats=self.num_key_value_groups)
        value=torch.repeat_interleave(value, dim=1, repeats=self.num_key_value_groups)

        #calculate the attention weights
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        if self.is_causal:
            attn_weights = attn_weights.masked_fill(
                self.mask[:, :, -attn_weights.shape[2]:, -attn_weights.shape[3]:] == 0, float("-inf")
            )


        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_output = torch.matmul(attn_weights, value)

        #print(12,attn_output.shape)

        attn_output=attn_output.transpose(1, 2)

        #print(13,attn_output.shape)

        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        #print(14,attn_output.shape)


        return self.o_proj(attn_output)
        

class JDecoder(torch.nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.hidden_size=orig.hidden_size
        self.self_attn=JAttention(orig.self_attn)
        self.mlp=JMLP(orig.mlp)
        self.input_layernorm=JRMSNorm(orig.input_layernorm)
        self.post_attention_layernorm=JRMSNorm(orig.post_attention_layernorm)
        
    def forward(self,x,cache):
        res=x
        x=self.input_layernorm(x)
        x=self.self_attn(x,cache)
        x=res+x
        res=x
        x=self.post_attention_layernorm(x)
        x=self.mlp(x)
        return x+res

class Model(torch.nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.vocab_size=orig.vocab_size
        self.embed_tokens=orig.embed_tokens
        self.padding_idx=orig.padding_idx
        self.norm=JRMSNorm(orig.norm)
        self.lm_head=torch.nn.Linear(orig.embed_tokens.weight.shape[1],orig.embed_tokens.weight.shape[0],bias=False)
        self.lm_head.weight=self.embed_tokens.weight
        self.layers=torch.nn.ModuleList()
        for l in orig.layers:
            self.layers.append(JDecoder(l))

    def forward(self,x,cache):
        x=self.embed_tokens(x)
        for l in self.layers:
            x=l(x,cache)
        x=self.norm(x)
        x=self.lm_head(x)
        return x

    def nextToken(self,x,cache):
        logits=self(x,cache)
        return torch.multinomial(torch.softmax(logits[0][-1],-1),1).item()



        

if __name__=="__main__":
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    orig=transformers.AutoModel.from_pretrained(model_id)
    m=Model(orig)
    tok=transformers.AutoTokenizer.from_pretrained(model_id)
    cache=DataCache()
    toks=tok.encode("""<|start_header_id|>system<|end_header_id|>
You are a python programming expert and are trying to give useful examples.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Schreib mir ein python programm das alle primzahlen von 0 bis 100 ausgibt.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")
    with torch.no_grad():
        t=m.nextToken(torch.tensor([toks]),cache)
    print(tok.decode(toks))
    while True:
        print(repr(tok.decode([t])),t)
        toks.append(t)
        if t==128009:
            break
        with torch.no_grad():
            t=m.nextToken(torch.tensor([[t]]),cache)
    print(tok.decode(toks))
        

    
