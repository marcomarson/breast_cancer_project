__author__ = "Marco Marson"
__version__ = "1.0"
__maintainer__ = "Marco Marson"
__email__ = "vollet.marson@gmail.com"
__status__ = "Development"

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

class CompareAndSave:
    def get_confusion_matrix_result(self, algorithm1, name, y_test, y_pred):
        # #Avaliar o modelo: Acurácia e matriz de contingência
        print("Resultado da Avaliação do Modelo --->>>"+name)
        print("Matriz de confusão")
        print(confusion_matrix(y_test, y_pred))
        print('')
        print("Relatório de classificação")

        print(classification_report(y_test, y_pred))
        print('')

        # Salvar o modelo para uso posterior
        joblib.dump(algorithm1, name+'.joblib')
        print('Gerado modelo '+name+'.joblib com sucesso\n\n')
