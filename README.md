# Classificazione di caratteri Manoscritti


Utilizzando l'implentazione del Decision Tree fornita dalla libreria scikit-learn, si è realizato un classificatore di caratteri manoscritti.
Si è mostrato particolare interesse nel rielaborare il dataset EMNIST rendendolo compatibile con l’input atteso dalle funzioni di fit e di predict, si è garantito inoltre che al variare delle dimensioni del training set esso risultasse bilanciato.\
Quindi al crescere delle dimensioni del training set si sono effettuati alcuni test mirati ad evidenziare l'accuratezza nella predizione ottenute sul Test Set e sul Training Set.

## Organizzazione dell'elaborato

L'elaborato si articola in due file:

 * **elaborateDataset:** In questo file vengono implementate le fonzioni che permettono la gestione del Data Set.
  In seguito vengono riportate le signatures delle funzioni specificandone i parametri richiesti.
 
   ```manageDataSet(dimension, X, y)```
   
   * dimension: numero di samples contenuti.
   * X: matrice delle immagini. 
   * y: vettore delle labels.
     
   ```balanceDataSet(X, y)```
   
   * X: matrice delle immagini. 
   * y: vettore delle labels.
        
 * **lettersTest:** In questo file vengono implementati i test. 

