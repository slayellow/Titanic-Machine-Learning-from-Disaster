Titanic-Machine-Learning-from-Disaster
======================================

-	Site: https://www.kaggle.com/c/titanic

Description
-----------

-	타이타닉과 같은 상황이 발생시 구명정이 충분치 않으므로 여성, 아이들, 상류층과 같은 사람들이 생존할 가능성이 높다.
-	어떤 부류의 사람들이 생존 가능성이 있는지에 대한 분석 진행

Goal
----

-	승객이 타이타닉 침몰에서 살아남을 수 있는지 여부를 예측
-	Test Dataset에 대해 0 or 1로 예측

Submission File Format
----------------------

![Submission_File_Format](./Image/Submission_File_Format.png)

Dataset
-------

-	Training Set: 891개의 데이터 / 12개의 Column으로 구성
-	Test Set: 418개의 데이터 / Target Data를 제외한 11개의 Column으로 구성

Model
-----

![Model](./Image/Model.png)

Result
------

### Hyper Parameter

-	Epoch: 300
-	Batch Size : 1000

![Result](./Image/Result.png)

Discussion
----------

-	Deep Neural Network로 Categorial Classification을 구현
-	성능이 좋지는 않다.
