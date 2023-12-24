import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import telebot

# Загрузка обученной модели
model = AutoModelForSequenceClassification.from_pretrained('intel/neural-chat-7b-v3-1', num_labels=4)
tokenizer = AutoTokenizer.from_pretrained('intel/neural-chat-7b-v3-1')

# Инициализация бота
bot_token = 'YOUR_BOT_TOKEN'  # замените на токен вашего бота
bot = telebot.TeleBot(bot_token)

# Функция для генерации рекомендаций и отчета по развитию клиентов
def generate_report(product_purpose, product_name):
    # Токенизация входных данных
    inputs = tokenizer(product_purpose, product_name, return_tensors='pt')

    # Прогноз модели
    outputs = model(**inputs)
    
    # Извлечение вероятностей для каждой области
    likelihoods = outputs.logits.detach().numpy()

    # Подготовка рекомендаций, включая метку и вероятность
    recommendations = [{'метка': label, 'вероятность': likelihood} for label, likelihood in zip(['финансы', 'эффективность', 'процесс покупки', 'поддержка'], likelihoods[0])]

    # Генерация отчета по развитию клиентов
    report = {
        'название_продукта': product_name,
        'цель_продукта': product_purpose,
        'рекомендации': recommendations
    }

    return report

# Функция для продолжения тестирования
def продолжить_тестирование():
    продолжить = input('Хотите продолжить тестирование? (да/нет): ')
    if продолжить.lower() == 'да':
        цель_продукта_новая = input('Введите новую цель продукта: ')
        название_продукта_новое = input('Введите название нового продукта: ')
        отчет_новый = generate_report(цель_продукта_новая, название_продукта_новое)
        print("\nОтчет по развитию клиентов для", отчет_новый['название_продукта'])
        print("Цель продукта:", отчет_новый['цель_продукта'])
        print("Рекомендации:")
        df_новый = pd.DataFrame(отчет_новый['рекомендации'])
        print(df_новый)
        продолжить_тестирование()

# Функция для отправки отчета в Telegram
def send_report_to_telegram(report):
    chat_id = 'YOUR_CHAT_ID'  # замените на ID чата, в который нужно отправить отчет
    message = f"Отчет по развитию клиентов для {report['название_продукта']}\n"
    message += f"Цель продукта: {report['цель_продукта']}\n"
    message += "Рекомендации:\n"
    for recommendation in report['рекомендации']:
        message += f"{recommendation['метка']}: {recommendation['вероятность']}\n"
    bot.send_message(chat_id, message)

# Главная функция
def main():
    while True:
        цель_продукта = input('Введите цель продукта: ')
        название_продукта = input('Введите название продукта: ')
        отчет = generate_report(цель_продукта, название_продукта)
        print("\nОтчет по развитию клиентов для", отчет['название_продукта'])
        print("Цель продукта:", отчет['цель_продукта'])
        print("Рекомендации:")
        df = pd.DataFrame(отчет['рекомендации'])
        print(df)
        send_report_to_telegram(отчет)
        продолжить_тестирование()

if __name__ == "__main__":
    main()