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
    ):
        super().__init__()

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
        logits_list = []
        for unique_task_id in unique_task_ids_list:
            task_id_filter = task_ids == unique_task_id
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)

            logits_list.append(logits)
        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used

        # zip and flatten subtasks logit lists to interleave them and have all examples
        # in one list
        logits_list = [
            logits
            for zipped_subtask_logits in zip(*logits_list)
            for logits in zipped_subtask_logits
        ]

        # Pad the smaller output layers so logit outputs can be transformed to tensor
        max_output_layer_size = max([logits.shape[1] for logits in logits_list])
        logits_list = [
            nn.functional.pad(
                logits, (0, max_output_layer_size - logits.shape[1]), "constant", -100
            )
            for logits in logits_list
        ]

        outputs = (torch.stack(logits_list), outputs[2:])

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
