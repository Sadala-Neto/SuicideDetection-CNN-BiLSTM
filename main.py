#region Bibliotecas utilizadas
import pandas as pd
import numpy as np
import re
import nltk
import time
import psutil
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, classification_report, f1_score,roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, Dropout, Input, BatchNormalization, MaxPooling1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from keras.optimizers import Adam


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# endregion


# region Seleção do dataset

print("====================================")
print("Inicio da fase de seleção do dataset")
print("====================================")

# Definindo os caminhos para os arquivos do dataset de suicídio
df_suicide_data = pd.read_csv("../Suicide_Detection.csv")

# Mapeando as colunas do dataset de suicídio para 'text' e 'class'
df_suicide_data = df_suicide_data[['text', 'class']]

print("Primeiras 5 linhas do DataFrame original")
print(df_suicide_data.head())

# Trocando os valores da classe para o padrão numérico (0 para não-suicídio, 1 para suicídio)
# classe: 'non-suicide' -> 0, 'suicide' -> 1
label_map = {'non-suicide': 0, 'suicide': 1}

df_suicide_data['feeling'] = df_suicide_data['class'].replace(label_map)

# Descartando a coluna 'class' original e renomeando a nova coluna de mapeamento
df_suicide_data = df_suicide_data.drop(columns='class')

print("\nDataFrame após mapeamento de classes")
print(df_suicide_data.head())

print(f"\nShape do DataFrame combinado (único arquivo): {df_suicide_data.shape}")
print("\nContagem de classes no DataFrame combinado (antes da remoção de duplicatas)")
print(df_suicide_data['feeling'].value_counts())

# Remover duplicatas da coluna 'text'
print("\nRemovendo duplicatas da coluna 'text'")
initial_rows = len(df_suicide_data)
df_suicide_data.drop_duplicates(subset=['text'], inplace=True)
print(f"Número de linhas antes: {initial_rows}")
print(f"Número de linhas após remover duplicatas: {len(df_suicide_data)}")
print("\nContagem de classes após remover duplicatas:")
print(df_suicide_data['feeling'].value_counts())

# Balanceamento após a remoção de duplicatas
print("\nContagem de classes após remover duplicatas (antes do balanceamento):")
initial_counts = df_suicide_data['feeling'].value_counts()
print(initial_counts)

# Separa as classes
df_class_0 = df_suicide_data[df_suicide_data['feeling'] == 0]  # non-suicide
df_class_1 = df_suicide_data[df_suicide_data['feeling'] == 1]  # suicide

# Encontra o tamanho da menor classe para balanceamento
min_class_size = 10000 #min(len(df_class_0), len(df_class_1))

# Reduz o número de amostras das classes para o tamanho da menor
df_class_0_balanced = df_class_0.sample(n=min_class_size, random_state=42) #n=min_class_size
df_class_1_balanced = df_class_1.sample(n=min_class_size, random_state=42) #n=min_class_size

# Concatena as classes balanceadas e embaralha o dataset final
df_balanced = pd.concat([df_class_0_balanced, df_class_1_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

# Atualiza df_suicide_data com o dataset balanceado
df_suicide_data = df_balanced

print("\nContagem de classes após o balanceamento:")
print(df_suicide_data['feeling'].value_counts())
print(f"Shape do DataFrame final após balanceamento: {df_suicide_data.shape}")

print("====================================")
print("Fim da fase de seleção do dataset")
print("====================================")

# endregion


# region Pre-processamento

from bs4 import BeautifulSoup

print("\n====================================")
print("Inicio do Pré-processamento")
print("====================================")

# Pré-processamento dos dados
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remover tags HTML (o dataset Amazon não tem, mas manter é seguro)
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    # Remover links
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text), flags=re.MULTILINE)
    # Remover menções a usuários (@)
    text = re.sub(r'\@\w+|\#', '', str(text))
    # Converter para minúsculas
    text = text.lower()
    # Remover pontuações e caracteres especiais
    text = re.sub(r'[^\w\s]', '', str(text))
    # Remover múltiplos espaços em branco e espaços nas bordas
    text = ' '.join(text.split())
    # Remover stop words
    # ⚠️ IMPORTANTE:
    # - Mantenha esta linha ativa → versão SEM stopwords
    # - Comente esta linha → versão COM stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df_suicide_data['text'] = df_suicide_data['text'].apply(preprocess_text)
# Cópia dos textos pré-processados para análise qualitativa
texts_clean = df_suicide_data['text'].values

print("\nDataFrame após pré-processamento de texto:")
print(df_suicide_data.head())

# Total de amostras após duplicatas no df_suicide_data
total_samples = len(df_suicide_data)

# Teste (20% do total)
test_size_final = 0.2
# Validação (25% do restante, que é 80% do total, então 20% do total)
val_size_final = 0.25 / (1 - test_size_final) * test_size_final # 0.25 * 0.8 = 0.2 do total
val_size_from_train = 0.25

# Dados para divisão
X_text = df_suicide_data['text'].values  # texto legível
X = X_text.copy()                        # usado para tokenização
y = df_suicide_data['feeling'].values
le = LabelEncoder()
y = le.fit_transform(y)

# Primeira divisão: separa Teste (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size_final, random_state=5, stratify=y)
# Segunda divisão: do restante (X_temp, y_temp), separa Validação (25% dele, que é 20% do total) e Treino (75% dele)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_from_train, random_state=37, stratify=y_temp)

# Divisão dos textos legíveis para análise qualitativa
X_text_temp, X_text_test, y_temp, y_test = train_test_split(X_text, y, test_size=test_size_final, random_state=5, stratify=y)
X_text_train, X_text_val, y_train, y_val = train_test_split(X_text_temp, y_temp, test_size=val_size_from_train, random_state=37, stratify=y_temp)

print(f"\nNúmero de amostras no conjunto de Treino: {len(X_train)}")
print(f"Número de amostras no conjunto de Validação: {len(X_val)}")
print(f"Número de amostras no conjunto de Teste: {len(X_test)}")

print("\nDistribuição de sentimentos no conjunto de Treino:")
print(pd.Series(y_train).value_counts(normalize=True))
print("Distribuição de sentimentos no conjunto de Validação:")
print(pd.Series(y_val).value_counts(normalize=True))
print("Distribuição de sentimentos no conjunto de Teste:")
print(pd.Series(y_test).value_counts(normalize=True))


# Vetorização dos textos
tokenizer = Tokenizer(num_words=50000, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)

# Convertendo os textos pré-processados em sequências numéricas de tokens
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)

print("\n=== Análise do Comprimento das Sequências ===")

# Calcular os comprimentos das sequências em cada conjunto
train_lengths = [len(seq) for seq in X_train]
val_lengths = [len(seq) for seq in X_val]
test_lengths = [len(seq) for seq in X_test]

vocab_size = len(tokenizer.word_index) + 1
max_len = 160 # sem stopwords = 160; com stopwords = 320

# Aplica o padding
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
X_val = pad_sequences(X_val, padding='post', maxlen=max_len)

print(f"\nShape de X_train após padding: {X_train.shape}")
print(f"Shape de X_test após padding: {X_test.shape}")
print(f"Shape de X_val após padding: {X_val.shape}")

print("\n====================================")
print("Fim do Pré-processamento")
print("====================================")

# endregion


# region Analise usando a RNA

#criação do modelo
model = Sequential()
model.add(Input(shape=(max_len,)))
model.add(Embedding(vocab_size, 100))
model.add(Dropout(0.3))
model.add(Conv1D(32, 5, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001))))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'precision', 'recall', 'auc'])

model.summary()

# Calcular pesos de classe automaticamente
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

print("Pesos de classe:", class_weights)
print()

#criterio de parada & Define o callback EarlyStopping
early_stopping_loss = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
callbacks_list = [early_stopping_loss, reduce_lr]

#Treinamento do modelo
print("Inicio do treinamento")

# ===============================
# Inicio da medição de tempo e memória (antes do treino)
# ===============================
process = psutil.Process()
mem_before = process.memory_info().rss / (1024 ** 2)  # MB
start_time = time.time()

rna = model.fit(X_train, y_train, epochs=25, batch_size=1000, validation_data=(X_val, y_val), callbacks=callbacks_list, class_weight=class_weights)

# ===============================
# Final da medição de tempo e memória (após o treino)
# ===============================

end_time = time.time()
mem_after = process.memory_info().rss / (1024 ** 2)  # MB

training_time = end_time - start_time
memory_used = mem_after - mem_before

print(f"\n⏱ Tempo total de treino: {training_time:.2f} segundos")
print(f"🧠 Consumo aproximado de memória: {memory_used:.2f} MB")

# ===============================
# Curva de treino vs validação (acuracia)
# ===============================
plt.figure(figsize=(7, 4), dpi=300)
plt.plot(rna.history['accuracy'], label='Training')
plt.plot(rna.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Accuracy Curve - Training vs Validation')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# ===============================
# Curva de treino vs validação (loss)
# ===============================
plt.figure(figsize=(7, 4), dpi=300)
plt.plot(rna.history['loss'], label='Training')
plt.plot(rna.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve - Training vs Validation')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# ===============================
# Grafico da Matriz de confusão
# ===============================

def plotar_matriz_confusao(cm, nome_conjunto):
    plt.figure(figsize=(7, 4), dpi=300)
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Confusion Matrix - {nome_conjunto}')
    plt.colorbar()
    plt.xlabel('Predicted Class')
    plt.ylabel('Correct Class')

    # Marcar valores nas células
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.4f}",
                     ha="center", va="center")

    plt.xticks([0, 1], ['No Suicidal \nIdeation\n', 'Suicidal \nIdeation\n'])
    plt.yticks([0, 1], ['No Suicidal \nIdeation', 'Suicidal \nIdeation'])
    plt.tight_layout()
    plt.show()


# ===============================
# Avaliação do modelo
# ===============================

def avaliar_modelo(model, X, X_text, y, nome_conjunto="Teste"):
    print(f"\n Avaliação no conjunto de {nome_conjunto.upper()}:")
    resultados = model.evaluate(X, y, verbose=0)
    print(f"Loss:       {resultados[0]:.4f}")
    print(f"Acurácia:   {resultados[1]:.4f}")
    print(f"Precisão:   {resultados[2]:.4f}")
    print(f"Revocação:  {resultados[3]:.4f}")

    # Predição
    y_probs = model.predict(X).flatten()
    y_pred = np.round(y_probs)

    # Matriz de confusão
    print(f"\n🔹 Matriz de Confusão - {nome_conjunto}")
    cm = confusion_matrix(y, y_pred, normalize='true')
    print(cm)

    # Plot da matriz de confusão
    plotar_matriz_confusao(cm, nome_conjunto)

    # Identificação de FP e FN
    fp_idx = np.where((y == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y == 1) & (y_pred == 0))[0]

    print(f"\nFalsos Positivos (FP): {len(fp_idx)}")
    print(f"Falsos Negativos (FN): {len(fn_idx)}")

    # Exemplos qualitativos de erros
    def mostrar_exemplos(indices, X_texto, y_real, y_pred, titulo, n=1):
        print(f"\n📌 Exemplos de {titulo}:")
        for idx in indices[:n]:
            print("-" * 60)
            print("Texto:", X_texto[idx][:300], "...")
            print("Real:", y_real[idx], "| Previsto:", int(y_pred[idx]))

    mostrar_exemplos(fp_idx, X_text, y, y_pred, "FALSOS POSITIVOS")
    mostrar_exemplos(fn_idx, X_text, y, y_pred, "FALSOS NEGATIVOS")

    print(classification_report(y, y_pred, digits=4))

    # F1-score explícito
    f1 = f1_score(y, y_pred)
    print(f"F1-Score ({nome_conjunto}): {f1:.4f}")

# Chamada para treino, validação e teste
avaliar_modelo(model, X_train, X_text_train, y_train, "Training")
avaliar_modelo(model, X_val,   X_text_val,   y_val,   "Validation")
avaliar_modelo(model, X_test,  X_text_test,  y_test,  "Test")

# ===============================
# Curva ROC
# ===============================
y_test_probs = model.predict(X_test).flatten()
fpr, tpr, _ = roc_curve(y_test, y_test_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 4), dpi=300)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Set')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# ===============================
# Consolidação do custo computacional x desempenho
# ===============================
resultados_experimento = {
    "uso_stopwords": "Não" ,  # alterar manualmente para "Sim" na outra execução
    "tempo_treino_segundos": training_time,
    "memoria_MB": memory_used,
}

# Métricas no conjunto de teste
y_test_probs = model.predict(X_test).flatten()
y_test_pred = np.round(y_test_probs)

resultados_experimento["accuracy"] = np.mean(y_test_pred == y_test)
resultados_experimento["f1_score"] = f1_score(y_test, y_test_pred)

df_resultados = pd.DataFrame([resultados_experimento])
print("\n📊 Resumo custo x desempenho:")
print(df_resultados)

print("====================================")
print("Fim da Analise usando a RNA")
print("====================================")

# endregion

