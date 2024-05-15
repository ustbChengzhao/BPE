from collections import OrderedDict
import pickle
import re
from tqdm import tqdm

class BPE:
    def __init__(self):
        self.b2i = OrderedDict() # bytes to index
        self.i2b = OrderedDict() # index to bytes
        self.next_id = 0
        
        # special tokens
        self.sp_s2i = {}
        self.sp_i2s = {}
        
    # 相邻token统计
    def _pair_stats(self, token, stats):
        for i in range(len(token) - 1):
            new_token = token[i] + token[i + 1]
            if new_token not in stats:
                stats[new_token] = 0
            stats[new_token] += 1
    
    def _merge_pair(self, tokens, new_token):
        merged_tokens = []
        
        i = 0
        while i < len(tokens):
            if (i + 1) < len(tokens) and tokens[i] + tokens[i + 1] == new_token:
                merged_tokens.append(new_token)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens
    
    def vocab_size(self):
        return self.next_id
    
    def train(self, text_list, vocab_size):
        # 单字节是最基础的token
        for i in range(256):
            self.b2i[bytes([i])] = i
        self.next_id = 256
        
        # 语料转byte
        tokens_list = []
        for text in text_list:
            tokens = [bytes([b]) for b in text.encode('utf-8')]
            tokens_list.append(tokens)
        
        # 进度条
        progress = tqdm(total=vocab_size - 256)
        
        while True:
            # 词表足够大，退出训练
            if self.next_id >= vocab_size:
                break
            
            stats = {}
            for tokens in tokens_list:
                self._pair_stats(tokens, stats)
            
            # 没有更多相邻token，无法生成更多token
            if not stats:
                break
            
            # 合并最高频的相邻token，作为新的token加入词表
            best_token = max(stats, key=stats.get)  # 最高频的相邻token

            new_tokens_list = []
            for tokens in tokens_list:
                new_tokens_list.append(self._merge_pair(tokens, best_token))
            tokens_list = new_tokens_list
            
            # new token加入词表
            self.b2i[best_token] = self.next_id
            self.next_id += 1
            
            # 刷新进度条
            progress.update(1)
            
        self.i2b = {i: b for b, i in self.b2i.items()}
    
    # 返回词表
    def vocab(self):
        v = {}
        v.update(self.b2i)
        v.update({token.encode('utf-8'): id for id, token in self.sp_i2s.items()})
        return v

    # 特殊token
    def add_special_tokens(self, tokens):
        for token in tokens:
            if token not in self.sp_s2i:
                self.sp_s2i[token] = self.next_id
                self.sp_i2s[self.next_id] = token
                self.next_id += 1
    
    # 编码
    def encode(self, text):
        # 特殊token先分离
        pattern = "(" + "|".join([re.escape(token) for token in self.sp_s2i.keys()]) + ")"
        splits = re.split(pattern, text)
        
        # encode结果
        enc_ids = []
        enc_tokens = []
        for sub_text in splits:
            # 特殊字符编码
            if sub_text in self.sp_s2i:
                enc_ids.append(self.sp_s2i[sub_text])
                enc_tokens.append(sub_text.encode('utf-8'))
            # 普通字符编码
            else:
                tokens = [bytes([b]) for b in sub_text.encode('utf-8')]
                while True:
                    # 统计相邻token频率
                    stats = {}
                    self._pair_stats(tokens, stats)
                    
                    # 选择合并后id最小的pair合并（优先合并最短的）
                    new_token = None
                    for merge_token in stats:
                        if merge_token in self.b2i and (new_token is None or self.b2i[merge_token] < self.b2i[new_token]):
                            new_token = merge_token
                    
                    # 没有可以合并的pair，退出
                    if new_token is None:
                        break
                    
                    # 合并pair
                    tokens = self._merge_pair(tokens, new_token)
                enc_ids.extend([self.b2i[token] for token in tokens])
                enc_tokens.extend(tokens)
        return enc_ids, enc_tokens
    
    def decode(self, ids):
        bytes_list = []
        for id in ids:
            if id in self.sp_i2s:
                bytes_list.append(self.sp_i2s[id].encode('utf-8'))
            else:
                bytes_list.append(self.i2b[id])
        return b''.join(bytes_list).decode('utf-8')
    
    def save(self, file):
        with open(file, 'wb') as f:
            f.write(pickle.dumps((self.b2i, self.sp_s2i, self.next_id)))
            
    def load(self, file):
        with open(file, 'rb') as f:
            self.b2i, self.sp_s2i, self.next_id = pickle.loads(f.read())
        self.i2b = {i: b for b, i in self.b2i.items()}
        self.sp_i2s = {i: s for s, i in self.sp_s2i.items()}
        

if __name__ == "__main__":
    cn=open('dataset/train-cn.txt','r').read()
    en=open('dataset/train-en.txt','r').read()
    
    # 训练
    tokenizer=BPE()
    tokenizer.train(text_list=[cn,en],vocab_size=300)
    
    # 特殊token
    tokenizer.add_special_tokens((['<|im_start|>','<|im_end|>','<|endoftext|>','<|padding|>']))
    
    # 保存
    tokenizer.save('tokenizer.bin')
    
    # 还原
    tokenizer=BPE()
    tokenizer.load('tokenizer.bin')
    print('vocab size:',tokenizer.vocab_size())
    
    # 编码
    ids,tokens=tokenizer.encode('<|im_start|>system\nyou are a helper assistant\n<|im_end|>\n<|im_start|>user\n今天的天气\n<|im_end|><|im_start|>assistant\n')
    print('encode:',ids,tokens)
    
    # 解码
    s=tokenizer.decode(ids)
    print('decode:',s)
    
    # 打印词典
    print('vocab:',tokenizer.vocab())