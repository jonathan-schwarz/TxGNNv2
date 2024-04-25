"""LLM Models."""
import torch
import torch.nn.functional as F
from torch import nn

from transformers import MistralModel, MistralPreTrainedModel
from transformers import modeling_outputs
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from typing import List, Optional, Tuple, Union


class MistralForCustomSequenceClassification(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MistralModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fromage_settings: Optional[dict] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if fromage_settings['use_fromage'] and 'top_only' != fromage_settings['fromage_type']:
            inputs_embeds = self.model.embed_tokens(input_ids)
            input_ids = None
            if 'p_tuning' in fromage_settings['fromage_type']:
                prefix_inputs_embeds = gnn_embeddings
                prefix_attention_mask = torch.ones_like(attention_mask[:, :2]).to(self.model.device)
            elif 'text_tuning' in fromage_settings['fromage_type']:
                explanation_ids = [self.model.embed_tokens(f)
                                   for f in fromage_settings['text_ids']]
                prefix_inputs_embeds = torch.concat([
                    explanation_ids[0], gnn_embeddings[:, :1, :],  # Drug Embedding
                    explanation_ids[1], gnn_embeddings[:, 1:, :],  # Disease Embedding
                    explanation_ids[2]], axis=1)  # Query
                prefix_attention_mask = torch.ones_like(attention_mask[:, :prefix_inputs_embeds.shape[1]])
            else:
                assert False

            inputs_embeds = torch.concat([prefix_inputs_embeds, inputs_embeds], axis=1)
            attention_mask = torch.concat([prefix_attention_mask, attention_mask], axis=1)

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # torch.Size([bs, seq_len, embedding size])
        hidden_states = transformer_outputs[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        # torch.Size([bs, embedding size])
        pooled_hidden_states = hidden_states[
                torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        return pooled_hidden_states
