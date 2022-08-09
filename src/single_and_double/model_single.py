import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead, BertPooler
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer
from VIT import vit
from category_id_map import CATEGORY_ID_LIST


class MultiModal_single(nn.Module):
    def __init__(self, args, from_pretrain = True):
        super().__init__()
        bert_output_size=768
        config = BertConfig.from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
        if from_pretrain:
            self.bert = UniBert(config).from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
        else:
            self.bert = UniBert(config)
        self.video_fc = nn.Linear(512, bert_output_size)
        # self.video_backbone = vit()
            
        self.classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))
        #self.visual_backbone = swin_tiny(args.swin_pretrained_path)

    # def forward(self, inputs, inference=False):
    def forward(self, title_input, title_mask, frame_input, frame_mask):
        # frame_input = self.video_fc(self.video_backbone(frame_input))
        # last_hidden = self.bert(inputs['frame_input'], inputs['frame_mask'], inputs['title_input'], inputs['title_mask']).mean(1)
        frame_input = self.video_fc(frame_input)
        last_hidden = self.bert(frame_input, frame_mask, title_input, title_mask).mean(1)

        pred = self.classifier(last_hidden)
        return pred
        # if inference:
        #     return torch.argmax(pred, dim=1)
        # else:
        #     return self.cal_loss(pred, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    
class InferenceModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.video_backbone = vit()
        self.model1 = MultiModal(args, False)
        self.model2 = MultiModal(args, False)

    def forward(self, title_input, title_mask, frame_input, frame_mask):
        # frame_input = self.video_fc(self.video_backbone(frame_input))
        frame_input = self.video_backbone(frame_input)
        pred1 = self.model1(title_input, title_mask, frame_input, frame_mask)
        pred2 = self.model2(title_input, title_mask, frame_input, frame_mask)
        pred3 = self.model3(title_input, title_mask, frame_input, frame_mask)
        pred4 = self.model4(title_input, title_mask, frame_input, frame_mask)
        pred = (pred1 + pred2 + pred3 + pred4) / 4
        # pred = torch.argmax(pred, dim = 1)
        return pred
# class InferenceModal(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.video_backbone = vit()
#         self.model1 = MultiModal(args, False)
#         self.model2 = MultiModal(args, False)
#         self.model3 = MultiModal(args, False)
#         self.model4 = MultiModal(args, False)

#     def forward(self, title_input, title_mask, frame_input, frame_mask):
#         # frame_input = self.video_fc(self.video_backbone(frame_input))
#         frame_input = self.video_backbone(frame_input)
#         pred1 = self.model1(title_input, title_mask, frame_input, frame_mask)
#         pred2 = self.model2(title_input, title_mask, frame_input, frame_mask)
#         pred3 = self.model3(title_input, title_mask, frame_input, frame_mask)
#         pred4 = self.model4(title_input, title_mask, frame_input, frame_mask)
#         pred = (pred1 + pred2 + pred3 + pred4) / 4
#         # pred = torch.argmax(pred, dim = 1)
#         return pred
        

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

class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None):
        text_emb = self.embeddings(input_ids=text_input_ids)

        video_emb = self.embeddings(inputs_embeds=video_feature)

        embedding_output = torch.cat([video_emb, text_emb], 1)

        mask = torch.cat([video_mask, text_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        return encoder_outputs
