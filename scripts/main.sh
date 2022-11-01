#!/bin/bash

python -m run +alg=enn +experiment=fnli +model=bert-base batch_size=10 val_batch_size=10
python -m run +alg=gtn +experiment=fnli +model=bert-base batch_size=10 val_batch_size=10 gtn.descent=True
python -m run +alg=rep +experiment=fnli +model=bert-base batch_size=10 val_batch_size=10 rep.cross_attend=True

python -m run +alg=enn +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=false data.zsre_yn=false data.hard_neg=false
python -m run +alg=gtn +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=false data.zsre_yn=false data.hard_neg=false gtn.descent=True
python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=false data.zsre_yn=false data.hard_neg=false

python -m run +alg=enn +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true
python -m run +alg=gtn +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true gtn.descent=True
python -m run +alg=rep +experiment=qa +model=t5large batch_size=10 val_batch_size=10 data.zsre_impl=true data.zsre_yn=true data.hard_neg=true

python -m run +alg=enn +experiment=sent +model=blender-small batch_size=5 val_batch_size=5
python -m run +alg=gtn +experiment=sent +model=blender-small batch_size=5 val_batch_size=5 gtn.descent=True
python -m run +alg=rep +experiment=sent +model=blender-small batch_size=5 val_batch_size=5
