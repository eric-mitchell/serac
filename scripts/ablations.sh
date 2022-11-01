#!/bin/bash

## These should be the default configs we ran for the main experiments
python -m run +alg=rep +experiment=fnli +model=bert-base batch_size=10 val_batch_size=10 rep.cross_attend=True rep.cls_name=distilbert-base-cased
python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true  rep.cross_attend=False rep.cls_name=distilbert-base-cased


python -m run +alg=rep +experiment=fnli +model=bert-base batch_size=10 val_batch_size=10 rep.cross_attend=False rep.cls_name=distilbert-base-cased
python -m run +alg=rep +experiment=fnli +model=bert-base batch_size=10 val_batch_size=10 rep.cross_attend=False rep.cls_name=bert-base-cased rep.checkpoint_grad=True
python -m run +alg=rep +experiment=fnli +model=bert-base batch_size=10 val_batch_size=10 rep.cross_attend=True rep.cls_name=bert-base-cased rep.checkpoint_grad=True


python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true  rep.cross_attend=True rep.cls_name=distilbert-base-cased
python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true  rep.cross_attend=False rep.cls_name=bert-base-cased rep.checkpoint_grad=True
python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true  rep.cross_attend=True rep.cls_name=bert-base-cased rep.checkpoint_grad=True

