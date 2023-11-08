> ***Disclaimer: Daten dürfen nur im Zusammenhang mit Evaluierung der Thesis genutzt werden***

**Masterarbeit Textklassifikation des Histologiebefundes nach ICD-O per Deep Learning in Kooperation mit dem klinischen Krebsregister Rheinland-Pfalz**

________________
**Code-Repository inkl. Datensätze und trainierter Modelle**

Vorgehen:

1. * [ ]   clone gitlab repo
2. * [ ]   Ausführen jeweiliger [modell]_run_v10.py im gleichen directory; dort müssen auch die trainierten Modelle in den jeweiligen Ordnern fm_v10, cm100_v10, cm50_v10 liegen

*Code wurde für Windows bzw. sonstige Plattformen angepasst, es sollten keine weiteren Pfadanpassungen notwendig sein.*

________________

*1. nb_svm_run_v10.py*

*  enthält Naive-Bayes- sowie Support-Vector-Machine-Implementierung der Klassifikatoren

*  optimiert per GridSearchCV mit Parametern für ngram_range, alpha (SVM), tf_idf-use

*  ausgewertet mit FB (b=0,5), Accuracy, Precision, TP, FP, FN

*2. fm_run_v10.py*

* run-file für *fm_obj_v10.py*

*3. fm_obj_v10.py*

* *individuelle Metriken:* Neben scikit-learn Metriken (für Testergebnisse) werden eigene Metriken für Training und Validierung genutzt; *precision_, recall_, fbeta_score_* waren
in keras 1.0 noch vorhanden, werden allerdings nur batch-wise berechnet und wurden deshalb entfernt; per Callback kann am Epochenende die jeweilige Metrik gesondert gespeichert werden (vgl. taken from https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2),
was in der Klasse Metrics_ umgesetzt wurde; diese wird als Callback während des model.fit aufgerufen und ermöglicht so die jeweilige Auswertung gesamthaft (und nicht nur batch-wise)

* *Load Data / Preprocessing:* Datacleaning, Label-Encoding, Train-Test-Validation-Split; gesondert gespeichert werden der Tokenizer() sowie das reversed_label dict,
um neue Daten genauso wie die Trainingsdaten zu behandeln (gleiches Zuordnen von Labeln, Tokens etc.)

* *create_model:* Modell wird aufgebaut, kompiliert; Training ist auskommentiert, dafür wird bereits trainiertes Modell geladen; dann Evaluation von Testdaten sowie neuen Daten mit
FB (b=0,5), Accuracy, Precision, TP, FP, FN

*4. cm100_run_v10.py*

* run-file für *cm100_obj_v10.py*

*5. cm100_obj_v10.py*

* siehe Punkt 3

* neu hinzu kommt eine Preprocessing-Routine, die dynamisch Regeln erstellt. Regeln enthalten eindeutige Kombi aus Klasse + Histologiebefund, die so mindestens 100 mal vorkommt;
Auswertung erfolgt, in dem die Sequenz des jeweiligen Befundes (tokenisiert) hintereinander weg an das Label gehängt wird (bspw. 493_10_2_3) und dann das Vorkommen dieser Kombination
gezählt sowie auf Eindeutigkeit hin geprüft wird; es findet eine Wiedereinsteuerung der betroffenen Einträge statt, welche Rechtschreibfehler abfangen soll; Performance wird
so nur geringfügig beinflusst

*6. cm50_run_v10.py*

* run-file für *cm50_obj_v10.py*

*7. cm50_obj_v10.py*

* siehe Punkt 5, statt 100-maligen Vorkommen jetzt 50

