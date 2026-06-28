# OPD (On-Policy Distillation) 实现
将标准的基于解码器的知识蒸馏（Decoder KD）方法转换为 **OPD (On-Policy Distillation)**，用于语言模型蒸馏。

### Decoder KD (基准方法)
- 教师生成"真实"回答作为目标
- 学生学习模仿教师的输出
- 教师的生成是固定的，作为训练目标

### OPD (On-Policy Distillation)
- 学生自己生成回答（on-policy）
- 学生和教师都处理学生生成的序列
- 在它们的输出分布之间计算 KL 散度
- 学生学习在自己的生成上与教师分布对齐

## 实现变更
### 1. 自定义 Trainer 类
主要变化发生在 `compute_loss` 方法中：
#### 修改前（标准 KD）：
```python
class KDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 教师生成回答
        with torch.no_grad():
            teacher_output = self.teacher_model.generate(inputs)
        
        # 学生从教师的输出中学习
        student_output = model(inputs, teacher_output)
        loss = compute_loss(student_output, teacher_output)
        return loss
```
#### 修改后（OPD）：
```python
class OPDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
    
    @torch.no_grad()
    def generate_sequences(self, input_ids, attention_mask):
        """学生自己生成回答"""
        self.model.eval()
        sequences = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id
        )
        self.model.train()
        return sequences
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 学生生成回答（on-policy）
        prompt_ids = inputs["input_ids"].to(model.device)
        prompt_mask = inputs["attention_mask"].to(self.model.device)
        sequences = self.generate_sequences(prompt_ids, prompt_mask)
        
        # 为生成的序列构建注意力掩码
        attention_mask = (sequences != self.processing_class.pad_token_id).long()
        
        # 学生在自己生成的序列上进行前向传播
        logits = model(sequences, attention_mask=attention_mask).logits[
            :, prompt_ids.shape[-1]:
        ]
        
        # 教师在学生生成的序列上进行前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher_model(sequences, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits[:, prompt_ids.shape[-1]:]
        
        # 处理词汇表大小不匹配的情况
        if logits.shape[-1] != teacher_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        # 计算反向 KL 散度
        completion_ids = sequences[:, prompt_ids.shape[-1]:]
        kl = compute_rkl(
            logits,
            teacher_logits,
            completion_ids,
            padding_id=self.processing_class.pad_token_id,
            reduction="mean"
        )
        
        loss = kl.mean()
        return loss
```
### ‘compute_rkl'
  增加了均值处理情况
### 数据处理
```
原始输入
   ↓
分词 (Tokenization)
   ↓
教师生成回答
   ↓
组合: [输入 + 教师回答]
   ↓
学生前向传播 → 计算损失
```
变为了
```
原始输入
   ↓
分词 (Tokenization)
   ↓
学生生成回答 (on-policy，每批次动态生成)
   ↓
组合: [输入 + 学生回答]
   ↓
双重前向传播:
   - 学生 logits
   - 教师 logits
   ↓
计算 KL 散度损失
```

## 实验结果
目前只运行了epoch=0.1的情况：测试结果全为一类，可能由于epoch不够大，之后继续尝试。


