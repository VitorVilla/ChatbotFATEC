# Chatbot FATEC

Este projeto é um chatbot desenvolvido em Python, utilizando a biblioteca `customtkinter` para criar uma interface gráfica amigável. O chatbot foi projetado para responder perguntas sobre o curso de Desenvolvimento de Software Multiplataforma da FATEC.

## Funcionalidades

- Interface gráfica intuitiva com design responsivo.
- Respostas do chatbot aparecem de forma gradual, letra por letra.
- Possibilidade de fazer perguntas e obter respostas instantaneamente.
- Análise de sentimentos de acordo com sua pergunta ou resposta.

## Tecnologias Utilizadas

- **Python**: Linguagem de programação usada para o desenvolvimento do chatbot.
- **customtkinter**: Biblioteca para a criação de interfaces gráficas em Python.
- **Pillow**: Biblioteca para manipulação de imagens.

## Estrutura do Projeto

```
Chatbot/
│
├── chatbot.py        # Código principal do chatbot
├── interface.py      # Código da interface gráfica
├── papel_aviao.png   # Ícone do papel de avião
├── requirements.txt   # Lista de dependências do projeto
└── README.md         # Documentação do projeto
```

## Como Usar

1. Clone este repositório em sua máquina local:
   ```bash
   git clone https://github.com/SEU_USUARIO/chatbot-fatec.git
   cd chatbot-fatec
   ```

2. Crie um ambiente virtual e entre nele:
```bash
   virtualenv venv
   source venv/bin/activate (linux comand)
   cd venv/Scripts/activate (windows comand)
``` 

3. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   python -m spacy download pt_core_news_md
   ```

4. Execute o programa:
   ```bash
   python interface.py
   ```

