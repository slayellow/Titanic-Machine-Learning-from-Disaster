import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data():
    def __init__(self, train_path, test_path):
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)
        self.make_columns()
        self.One_Hot_Encoding()
        self.Drop_Columns()

    def category_age(self, x):
        if x < 10:
            return 0
        elif x < 20:
            return 1
        elif x < 30:
            return 2
        elif x < 40:
            return 3
        elif x < 50:
            return 4
        elif x < 60:
            return 5
        elif x < 70:
            return 6
        else:
            return 7

    def make_columns(self):
        self.df_train['FamilySize'] = self.df_train['SibSp'] + self.df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
        self.df_test['FamilySize'] = self.df_test['SibSp'] + self.df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
        # Fare에 Null값은 전체 Fare의 평균
        self.df_test.loc[self.df_test.Fare.isnull(), 'Fare'] = self.df_test['Fare'].mean()
        # Fare의 분포를 맞추기위해 log사용
        self.df_train['Fare'] = self.df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
        self.df_test['Fare'] = self.df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
        # Age에 Null값이 많기 때문에 Name을 이용하여 Null값 채우기
        self.df_train['Initial'] = self.df_train.Name.str.extract('([A-Za-z]+)\.', expand=False)
        self.df_test['Initial'] = self.df_test.Name.str.extract('([A-Za-z]+)\.', expand=False)
        self.df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                                ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
        self.df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                                ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

        # Null로 이루어진 Age값 대입
        self.df_train.loc[(self.df_train.Age.isnull())&(self.df_train.Initial=='Mr'),'Age'] = 33
        self.df_train.loc[(self.df_train.Age.isnull())&(self.df_train.Initial=='Mrs'),'Age'] = 36
        self.df_train.loc[(self.df_train.Age.isnull())&(self.df_train.Initial=='Master'),'Age'] = 5
        self.df_train.loc[(self.df_train.Age.isnull())&(self.df_train.Initial=='Miss'),'Age'] = 22
        self.df_train.loc[(self.df_train.Age.isnull())&(self.df_train.Initial=='Other'),'Age'] = 46
        self.df_test.loc[(self.df_test.Age.isnull())&(self.df_test.Initial=='Mr'),'Age'] = 33
        self.df_test.loc[(self.df_test.Age.isnull())&(self.df_test.Initial=='Mrs'),'Age'] = 36
        self.df_test.loc[(self.df_test.Age.isnull())&(self.df_test.Initial=='Master'),'Age'] = 5
        self.df_test.loc[(self.df_test.Age.isnull())&(self.df_test.Initial=='Miss'),'Age'] = 22
        self.df_test.loc[(self.df_test.Age.isnull())&(self.df_test.Initial=='Other'),'Age'] = 46
        # Embarked에서 Null값은 'S'로 통일
        self.df_train['Embarked'].fillna('S', inplace=True)

        # Age을 Categroial하게 변경후 기존 Age는 삭제
        self.df_train['Age_cat'] = self.df_train['Age'].apply(self.category_age)
        self.df_test['Age_cat'] = self.df_test['Age'].apply(self.category_age)
        self.df_train.drop(['Age'], axis=1, inplace=True)
        self.df_test.drop(['Age'], axis=1, inplace=True)

        # Initial, Embarked, Sex 값을 Numerical하게 변경
        # Initial 변경
        self.df_train['Initial'] = self.df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
        self.df_test['Initial'] = self.df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

        # Embarked 변경
        self.df_train['Embarked'] = self.df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        self.df_test['Embarked'] = self.df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        # Sex 변경
        self.df_train['Sex'] = self.df_train['Sex'].map({'female': 0, 'male': 1})
        self.df_test['Sex'] = self.df_test['Sex'].map({'female': 0, 'male': 1})

    def One_Hot_Encoding(self):
        # One-Hot Encoding
        self.df_train = pd.get_dummies(self.df_train, columns=['Initial'], prefix='Initial')
        self.df_test = pd.get_dummies(self.df_test, columns=['Initial'], prefix='Initial')
        self.df_train = pd.get_dummies(self.df_train, columns=['Embarked'], prefix='Embarked')
        self.df_test = pd.get_dummies(self.df_test, columns=['Embarked'], prefix='Embarked')

    def Drop_Columns(self):
        #  Drop Columns
        self.df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
        self.df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

    def Split_Input_Output(self):
        # Input Output 분리
        X_train = self.df_train.drop('Survived', axis=1).values
        target_label = self.df_train['Survived'].values
        X_test = self.df_test.values
        # Training, Validation Set 분리
        # X_tr, X_vid, y_tr, y_vid = train_test_split(X_train, target_label, test_size=0.3, random_state=2019)
        return X_train, target_label, X_test

