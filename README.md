# UpDown Prediction Language Model using Twitter

## Modules
```c_dataset.py``` : train/test에 활용될 Dataset 구성에 사용되는 CustomDataset class 정의. filtering할 데이터인지, preprocess된 데이터인지에 따라 데이터셋 디렉토리 및 전처리 진행 여부 결정\
```config.py``` : seed, dataset directory, start & end date, hyper-parameters 설정 가능\
```module.py``` : bert fine-tuning module. pytorch-lightning 형식으로 구성. 각 step, epoch이 끝날 때마다 어떤 job을 수행할 지 구현\
```main.py``` : 학습 및 test시 실행. \
```tune.py``` : ray tune을 사용한 hyperparameter tuning시 실행. tune_asha 모듈 내부의 config 내부를 수정하여 tuning할 파라미터 설정 가능.

## Demo
### Train & Test
```python
>>> python main.py 
train, test, 및 filtering 진행
```
### Hyper-parameter Tuning
```python
>>> python tune.py
hyper-parameter tuning 진행. 마지막에 best hyper-parameter combination 출력
```
