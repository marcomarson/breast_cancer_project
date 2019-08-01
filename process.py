__author__ = "Marco Marson"
__version__ = "1.0"
__maintainer__ = "Marco Marson"
__email__ = "vollet.marson@gmail.com"
__status__ = "Development"

from sklearn.model_selection import train_test_split
import pandas as pd

pd.set_option('display.max_columns', 50)


class Process_Data:
    def __init__(self):
        self.data = pd.read_csv("breast-cancer.data")
        # self.data.drop('id', axis=1)

    # 1 = 0.25
    # 2=0.5
    # 3=1
    # deg - malig: 1, 2, 3.
    def normalize(self):
        self.y = self.data['Class']
        self.X = self.data.drop('Class', axis=1)
        self.X['deg-malig'] = self.X['deg-malig'].apply(
            lambda x: 0.25 if x == 1 else (0.5 if x == 2 else 1))
        self.X = pd.get_dummies(self.X, drop_first=True)

    def get_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
