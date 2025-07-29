# Grokking

This project demonstrates the **grokking phenomenon** using both MLP and Transformer models trained to perform **modulo division**.

* The **MLP** was trained on modulo **23**, as it showed poor performance on larger moduli.
* The **Transformer** was trained on modulo **97**.

The plots below show accuracy dynamics on the train and test sets for both models:

<img width="600" height="400" alt="MLP accuracy" src="https://github.com/user-attachments/assets/4013e9a5-668c-4286-b7aa-1c57b3ba9282" />

<img width="600" height="400" alt="Transformer accuracy" src="https://github.com/user-attachments/assets/1f317256-2c5d-4b03-b66a-281b2c9d92bd" />

As seen in the second plot, the **Transformer begins to generalize** after approximately **20,000 optimization steps**, clearly exhibiting grokking behavior.
In contrast, the **MLP fails to generalize**, even with significantly more training steps and a smaller modulo.
