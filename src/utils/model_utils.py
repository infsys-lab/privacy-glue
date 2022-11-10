#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


# adapted from
# https://towardsdatascience.com/
# how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
class MultiTaskModel(nn.Module):
    def __init__(
        self,
        encoder_name_or_path,
        tasks: List,
        label_names: Dict,
        from_tf,
        config: AutoConfig,
        cache_dir,
        revision,
        max_output_layer_size,
    ):
        super().__init__()
        self.max_output_layer_size = max_output_layer_size
        self.encoder = AutoModel.from_pretrained(
            encoder_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir,
            revision=revision,
        )
        self.output_heads = nn.ModuleDict()
        # one model several heads
        for task_id, task in enumerate(tasks):
            decoder = TokenClassificationHead(
                self.encoder.config.hidden_size, len(label_names[task])
            )
            # ModuleDict requires keys to be strings
            self.output_heads[str(task_id)] = decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output, pooled_output = outputs[:2]
        unique_task_ids_list = torch.unique(task_ids).tolist()
        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:
            task_id_filter = task_ids == unique_task_id
            # send the encoder outputs to the correct classification head depending on
            # task id.
            task_logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )
            # on first iteration create a new Tensor with the same dtype and device as
            # the classification head output but scale to the appropriate dimensions.
            if logits is None:
                logits = task_logits.new_empty(
                    (input_ids.shape[0], input_ids.shape[1], self.max_output_layer_size)
                )
            # Pad the smaller output layers so the outputs can be fit into the logits
            # tensor.
            if task_logits.shape[2] < self.max_output_layer_size:
                task_logits = nn.functional.pad(
                    task_logits,
                    (0, self.max_output_layer_size - task_logits.shape[2], 0, 0, 0, 0),
                    "constant",
                    -100,
                )
            # fill in the outputs in the right places
            logits[task_id_filter] = task_logits

            if labels is not None:
                loss_list.append(task_loss)

        outputs = (logits, outputs[2:])

        if len(loss_list) > 0:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs
        return outputs


class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(
            hidden_size, num_labels, bias=kwargs.get("bias", None)
        )
        self.num_labels = num_labels
        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
    ):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            labels = labels.long()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss
