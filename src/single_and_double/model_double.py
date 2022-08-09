import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import BertModel
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from category_id_map import CATEGORY_ID_LIST
from VIT import vit


from torch.nn import init
from collections import OrderedDict



class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, s, c, h, w = x.size()
        x = x.view(b*s, c, h, w)
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        x = x * y.expand_as(x)

        return x.view(b, s, c, h, w)


# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # x: input features with shape [b, s, c, h, w]
#         b, s, c, h, w = x.size()
#         x = x.view(b*s, c, h, w)

#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)

#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#         x = x * y.expand_as(x)

#         return x.view(b, s, c, h, w)

class MultiModal_double(nn.Module):
    def __init__(self, args, from_pretrain = True):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        # self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        # self.visual_backbone = swin_base()
        # self.visual_backbone = vit()
        self.visual_encoder = nn.Linear(512, 768)
        #self.visual_ECA_net = ECAAttention()
        
        
        bert_output_size = 768
        
        # cross-attention
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)

        self.classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))
            

    # def forward(self, inputs, inference=False):
    def forward(self, title_input, text_mask, frame_input, vedio_mask):
        # import pdb
        # pdb.set_trace()
        # torch.Size([bs, seqlen, 3, 224, 224])->torch.Size([bs, seqlen, 768])
        # frame_input = self.visual_ECA_net(frame_input)
        # frame_input = self.visual_backbone(frame_input)
        vision_embedding = self.visual_encoder(frame_input)
        # vision_embedding = frame_input
        # frame_input = self.visual_fc(frame_input)
        # vision_embedding = frame_input       
    
        # text_mask = inputs['title_mask']
        # vedio_mask = inputs['frame_mask']
        
        # torch.Size([8, 50])->torch.Size([8, 50, 768])
        bert_semantics_embedding = self.bert(input_ids=title_input, attention_mask=text_mask)['last_hidden_state']

        # vision_embedding = self.enhance(vision_embedding)
        # final_embedding = self.fusion([vision_embedding, bert_embedding])
        
        cross_attn_result_text = self.multi_head_decoder(tgt=vision_embedding.permute(1,0,2), memory=bert_semantics_embedding.permute(1,0,2),
                                                   tgt_key_padding_mask=vedio_mask==0, memory_key_padding_mask=text_mask==0)
        
        cross_attn_result_text = cross_attn_result_text.permute(1,0,2).mean(dim=1)
            
        prediction = self.classifier(cross_attn_result_text)
  
        return prediction
        # if inference :
        #     return torch.argmax(prediction, dim=1)
        # else:
        #     return self.cal_loss(prediction, inputs['label'])
    
    def CrossEntropyLoss_label_smooth(outputs, targets, num_classes=200, epsilon=0.2):
        N = targets.size(0)
        # torch.Size([8, 10])
        # 初始化一个矩阵, 里面的值都是epsilon / (num_classes - 1)
        smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))
        targets = targets.data

        # 为矩阵中的每一行的某个index的位置赋值为1 - epsilon
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)
        # 调用torch的log_softmax
        log_prob = nn.functional.log_softmax(outputs, dim=1)
        # 用之前得到的smoothed_labels来调整log_prob中每个值
        print(smoothed_labels)
        loss = - torch.sum(log_prob * smoothed_labels) / N
        return loss


    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        # loss = CrossEntropyLoss_label_smooth(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    

class InferenceModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.video_backbone = vit()
        self.model1 = MultiModal(args)
        self.model2 = MultiModal(args)
        self.model3 = MultiModal(args)
        self.model4 = MultiModal(args)
        self.model5 = MultiModal(args)

    def forward(self, title_input, title_mask, frame_input, frame_mask):
        # frame_input = self.video_fc(self.video_backbone(frame_input))
        frame_input = self.video_backbone(frame_input)
        pred1 = self.model1(title_input, title_mask, frame_input, frame_mask)
        pred2 = self.model2(title_input, title_mask, frame_input, frame_mask)
        pred3 = self.model3(title_input, title_mask, frame_input, frame_mask)
        pred4 = self.model4(title_input, title_mask, frame_input, frame_mask)
        pred5 = self.model5(title_input, title_mask, frame_input, frame_mask)
        pred = (pred1 + pred2 + pred3 + pred4 + pred5) / 5
        # pred = torch.argmax(pred, dim = 1)
        return pred



class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) *param.data +self.decay *self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.75, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None) and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None) and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='bert.embeddings.word_embeddings.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None) and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='bert.embeddings.word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None) and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]