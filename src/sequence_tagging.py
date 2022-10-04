#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.pipeline_utils import Privacy_GLUE_Pipeline


class Sequence_Tagging_Pipeline(Privacy_GLUE_Pipeline):
    def _retrieve_data(self) -> None:
        pass

    def _load_pretrained_model_and_tokenizer(self) -> None:
        pass

    def _apply_preprocessing(self) -> None:
        pass

    def _set_metrics(self) -> None:
        pass

    def _run_train_loop(self) -> None:
        pass
