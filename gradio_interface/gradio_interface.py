import gradio as gr

# Функция для взаимодействия с моделью через Gradio
def gradio_interface(product_purpose, product_name):
    отчет = generate_report(product_purpose, product_name)
    recommendations = pd.DataFrame(отчет['рекомендации'])
    return recommendations

# Создание интерфейса
iface = gr.Interface(
    fn=gradio_interface,
    inputs=["text", "text"],
    outputs=gr.outputs.Dataframe(headers=["метка", "вероятность"]),
    layout="vertical",
    title="Predictive Customer Pain Bot Interface",
    description="Enter the product purpose and name to generate recommendations."
)

iface.launch()