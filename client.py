import paramiko
import sys
import os

HOST = "ssh-sr006-jupyter.ai.cloud.ru"
PORT = 2222
USERNAME = "h100-zemskova-1.ai0001053-01066"
PRIVATE_KEY_PATH = "/Users/Zemskova/mlspace__private_key.txt"
REMOTE_PATH = "/home/jovyan/Tatiana_Z/bbq_demo/user_query/"

def send_data(text, image_path=None):
    """Отправляет текст и изображение на сервер."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USERNAME, key_filename=PRIVATE_KEY_PATH)
    
    # Отправка текста
    stdin, stdout, stderr = ssh.exec_command(f'echo "{text}" > {REMOTE_PATH}/text.txt')
    print(stdout.read().decode(), stderr.read().decode())

    # Отправка изображения (если указано)
    if image_path and os.path.exists(image_path):
        sftp = ssh.open_sftp()
        sftp.put(image_path, f"{REMOTE_PATH}/{os.path.basename(image_path)}")
        sftp.close()

    ssh.close()

if __name__ == "__main__":
    while True:
        user_input = input("Введите текст: ")
        if user_input.lower() == "exit":
            break
        send_data(user_input, "/path/to/image.jpg")
