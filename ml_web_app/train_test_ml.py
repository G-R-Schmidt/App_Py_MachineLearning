import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request

app = Flask(__name__)

# Carregar o conjunto de dados
data = pd.read_csv('C:\Users\gui15\Desktop\App_Py_MachineLearning\App_Py_MachineLearning\ml_web_app')

# Criar a variável de destino binária ('all_star')
data['all_star_binary'] = data['all_star'].apply(lambda x: 1 if x > 0 else 0)

# Definir variável de destino e recursos
X = data.drop(['first', 'last', 'team', 'year', 'all_star', 'all_star_binary'], axis=1)
y = data['all_star_binary']

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pré-processamento de dados
numeric_features = ['fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'turnover', 'pf', 'pts', 'fg_pct', 'fg3_pct', 'ft_pct']
categorical_features = ['team']  # Adicione outras variáveis categóricas se necessário

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Substitua 'most_frequent' se necessário
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Inicializar e treinar o classificador
def train_and_predict(classifier, X_train, y_train, X_test):
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Definir classificadores
classifiers = {
    'knn': KNeighborsClassifier(),
    'mlp': MLPClassifier(max_iter=1000),
    'dt': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(random_state=42)
}

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota para treinamento e teste
@app.route('/train_test', methods=['POST'])
def train_test():
    # Obter parâmetros do formulário
    classifier_name = request.form['classifier']
    k_value = request.form.get('k_value', None)

    # Pré-processar os dados
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Inicializar e treinar o classificador
    classifier = classifiers[classifier_name]
    y_pred = train_and_predict(classifier, X_train_transformed, y_train, X_test_transformed)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    # Criar matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não All-Star', 'All-Star'], yticklabels=['Não All-Star', 'All-Star'])
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.savefig('static/confusion_matrix.png')

    return render_template('index.html', accuracy=accuracy, macro_f1=macro_f1)

# Executar a aplicação
if __name__ == '__main__':
    app.run(debug=True)
