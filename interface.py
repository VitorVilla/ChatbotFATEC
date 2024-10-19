import customtkinter as ctk
import time
from PIL import Image, ImageTk
from chatbot import answer  # Importando a função 'answer' do arquivo chatbot.py

# Configurações iniciais
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Função para exibir a resposta com efeito de digitação
def mostrar_resposta_lentamente(texto, widget):
    widget.delete("1.0", ctk.END) 
    for letra in texto:
        widget.insert(ctk.END, letra)
        widget.update()
        time.sleep(0.05)

# Função chamada ao enviar a pergunta
def enviar_pergunta():
    pergunta = entry.get()
    if pergunta.strip(): 
        adicionar_mensagem("Você: " + pergunta)
        resposta = answer(pergunta)  # Chama a função 'answer' da lógica
        adicionar_mensagem("Chatbot: " + resposta)
        entry.delete(0, ctk.END)  # Limpa o campo de entrada após enviar a pergunta

# Função para adicionar mensagens à área de chat
def adicionar_mensagem(mensagem):
    chat_area.configure(state="normal")
    chat_area.insert(ctk.END, mensagem + "\n\n")
    chat_area.configure(state="disabled")
    chat_area.yview(ctk.END)

# Janela principal
janela = ctk.CTk()
janela.title("Chatbot FATEC")
janela.geometry("800x600")

# Paleta de cores
cor_base = "#092327"
cor_secundaria = "#0B5351"
cor_terciaria = "#00A9A5"
cor_texto = "#FFFFFF"
cor_fundo_entry = "#1B3A3B"

# Configurações da janela
fundo_frame = ctk.CTkFrame(janela, fg_color=cor_base)
fundo_frame.pack(fill="both", expand=True)

frame = ctk.CTkFrame(fundo_frame, corner_radius=15, fg_color=cor_secundaria)
frame.pack(padx=20, pady=20, fill="both", expand=True)

label = ctk.CTkLabel(frame, text="Olá! O que você deseja saber sobre o curso de Desenvolvimento de Software Multiplataforma da FATEC?",
                     font=("Helvetica", 20, "bold"), text_color=cor_texto, wraplength=700)
label.pack(pady=10)

chat_area = ctk.CTkTextbox(frame, font=("Helvetica", 14), fg_color=cor_base, text_color="white", corner_radius=10, state="disabled", height=400)
chat_area.pack(pady=10, fill="both", expand=True, padx=10)

entry_frame = ctk.CTkFrame(frame, fg_color=cor_secundaria)
entry_frame.pack(pady=10, fill="x")

entry = ctk.CTkEntry(entry_frame, font=("Helvetica", 16), fg_color=cor_fundo_entry, text_color="white", 
                     placeholder_text="Digite sua pergunta aqui...", placeholder_text_color=cor_texto)
entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)

imagem_papel_aviao = Image.open("./chatbot-pln/papel_aviao.png")
imagem_papel_aviao = imagem_papel_aviao.resize((20, 20)) 
foto_papel_aviao = ImageTk.PhotoImage(imagem_papel_aviao)

botao = ctk.CTkButton(entry_frame, image=foto_papel_aviao, command=enviar_pergunta, fg_color=cor_terciaria, hover_color="#008C8C", width=50, text="")
botao.pack(side="right", padx=5, pady=5)

# Inicia a interface gráfica
janela.mainloop()
