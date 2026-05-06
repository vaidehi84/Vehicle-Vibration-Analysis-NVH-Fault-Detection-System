.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    
    from scipy.fft import fft
    from scipy.stats import skew, kurtosis
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_curve,
        auc
    )
    
    import joblib

.. code:: ipython3

    np.random.seed(42)
    
    def generate_signal(label, size=300):
        t = np.linspace(0, 1, size)
    
        if label == 0:
            # Normal vibration
            signal = np.sin(2*np.pi*5*t) + np.random.normal(0, 0.2, size)
        else:
            # Fault vibration
            signal = (
                np.sin(2*np.pi*5*t) +
                np.sin(2*np.pi*18*t) +
                np.random.normal(0, 1.0, size)
            )
        return signal
    
    
    X, y = [], []
    
    for _ in range(2000):
        X.append(generate_signal(0))
        y.append(0)
    
        X.append(generate_signal(1))
        y.append(1)
    
    X = np.array(X)
    y = np.array(y)

.. code:: ipython3

    def extract_features(signal):
        fft_vals = np.abs(fft(signal))[:len(signal)//2]
    
        return [
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.sqrt(np.mean(signal**2)),   # RMS
            skew(signal),
            kurtosis(signal),
            np.mean(fft_vals),
            np.max(fft_vals),
            np.sum(fft_vals**2),
            np.argmax(fft_vals)
        ]
    
    X_feat = np.array([extract_features(s) for s in X])

.. code:: ipython3

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42
    )

.. code:: ipython3

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

.. code:: ipython3

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


.. parsed-literal::

    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1 Score: 1.0
    
    Classification Report:
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       418
               1       1.00      1.00      1.00       382
    
        accuracy                           1.00       800
       macro avg       1.00      1.00      1.00       800
    weighted avg       1.00      1.00      1.00       800
    
    

.. code:: ipython3

    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()



.. image:: output_6_0.png


.. code:: ipython3

    y_prob = model.predict_proba(X_test)[:,1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.show()



.. image:: output_7_0.png


.. code:: ipython3

    joblib.dump(model, "nvh_model.pkl")




.. parsed-literal::

    ['nvh_model.pkl']



.. code:: ipython3

    def live_signal(label=None):
        t = np.linspace(0, 1, 300)
    
        if label is None:
            label = np.random.choice([0,1])
    
        if label == 0:
            return np.sin(2*np.pi*5*t) + np.random.normal(0,0.2,300)
        else:
            return np.sin(2*np.pi*5*t) + np.sin(2*np.pi*18*t) + np.random.normal(0,1.0,300)
    
    
    def predict_live(signal):
        features = np.array(extract_features(signal)).reshape(1,-1)
        return model.predict(features)[0]
    
    
    for i in range(10):
        signal = live_signal()
        pred = predict_live(signal)
    
        plt.figure(figsize=(10,3))
        plt.plot(signal)
    
        if pred == 0:
            plt.title("NORMAL VIBRATION ✅")
        else:
            plt.title("FAULT DETECTED ⚠")
    
        plt.show()
    
        time.sleep(1)



.. image:: output_9_0.png



.. image:: output_9_1.png



.. image:: output_9_2.png



.. image:: output_9_3.png



.. image:: output_9_4.png



.. image:: output_9_5.png


.. parsed-literal::

    C:\Users\Vaidehi Sharma\anaconda3\Lib\site-packages\IPython\core\pylabtools.py:170: UserWarning: Glyph 9989 (\N{WHITE HEAVY CHECK MARK}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)
    


.. image:: output_9_7.png


.. parsed-literal::

    C:\Users\Vaidehi Sharma\anaconda3\Lib\site-packages\IPython\core\pylabtools.py:170: UserWarning: Glyph 9989 (\N{WHITE HEAVY CHECK MARK}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)
    


.. image:: output_9_9.png



.. image:: output_9_10.png


.. parsed-literal::

    C:\Users\Vaidehi Sharma\anaconda3\Lib\site-packages\IPython\core\pylabtools.py:170: UserWarning: Glyph 9989 (\N{WHITE HEAVY CHECK MARK}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)
    


.. image:: output_9_12.png


