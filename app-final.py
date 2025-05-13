import streamlit as st
import pandas as pd
import os
from llama_cpp import Llama
from tqdm import tqdm
import llama_cpp.llama as raw_llama
from datetime import datetime

# ==============================
# Кастомный конфиг и главный диалог
# P.S. конфиг нужен для того, чтобы увеличить максимально возможный размер загружаемого файла до определенного числа т.к. изначально лок на 200мб
# ==============================
st.set_page_config(page_title="Тестирование LLM-моделей", layout="wide")
st.title("🔍 Тестирование LLM-моделей на галлюцинации")

# ==============================
# Блок основных функций с которыми будем работать
# ==============================
def load_model_list():
    models_dir = os.path.join(os.getcwd(), "Models")
    return [f for f in os.listdir(models_dir) if f.endswith(".gguf")]

def load_llama_model(path):
    return Llama(
        model_path=path,
        n_ctx=4096, #из-за величины текстов в контекстном датасете пришлось сильно увеличить эту величину
        n_gpu_layers=-1,
        n_threads=4,
        low_vram=True #чтобы не погибал комп
    )

def evaluate_model(model, df, system_prompt="", judge_model=None):
    answers = []
    correct_answers, wrong_answers, instruction_violations = 0, 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["prompt"]
        expected_answer = row["answer"]

        prompt = f"""Вопрос: {question}
        Ответ:"""

        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1,
        )
        generated_answer = response["choices"][0]["message"]["content"].strip()
        answers.append(generated_answer)


        if judge_model is not None:
            # 💬 Оценка 1: правильность по смыслу
            judge_prompt_1 = f"""Ты — судья, оценивающий ответ другой модели. Учитывай, что она видела системный промт.
            Системный промт: {system_prompt}
            Вопрос: {question}
            Ожидаемый ответ: {expected_answer}
            Ответ модели: {generated_answer}
            Вопрос: Является ли ответ правильным по смыслу, даже если он не совпадает дословно? Отвечай только Да или Нет."""

            judge_response_1 = judge_model.create_chat_completion(
                messages=[{"role": "user", "content": judge_prompt_1}],
                max_tokens=2,
                temperature=0.0,
            )
            judge_decision_1 = judge_response_1["choices"][0]["message"]["content"].strip().lower()

            # if judge_decision_1 not in {"да", "нет"}: Т.К. модели слабые, логировать придется практически все
            #     unclear_cases.append({
            #         "question": question,
            #         "expected": expected_answer,
            #         "answer": generated_answer,
            #         "judge_raw": judge_decision_1
            #     })

            if 'да' in judge_decision_1:
                correct_answers += 1
            else:
                wrong_answers += 1

            # 💬 Оценка 2: следование инструкции
            judge_prompt_2 = f"""Ты — судья, оценивающий, выполнила ли другая модель инструкцию из системного промта.
            Системный промт: {system_prompt}
            Вопрос: {question}
            Ответ модели: {generated_answer}
            Вопрос: Выполнена ли инструкция, заданная в системном промте? Отвечай только Да или Нет."""

            judge_response_2 = judge_model.create_chat_completion(
                messages=[{"role": "user", "content": judge_prompt_2}],
                max_tokens=2,
                temperature=0.0,
            )
            judge_decision_2 = judge_response_2["choices"][0]["message"]["content"].strip().lower()

            if 'да' in judge_decision_2:
                instruction_violations += 1

        else:
            # fallback: сравниваем по вхождению
            if str(generated_answer).lower() in str(expected_answer).lower():
                correct_answers += 1
            else:
                wrong_answers += 1
                
        if judge_model is not None:
            st.markdown(f"**🧑‍⚖️ Судья (смысл):** `{judge_decision_1}`")
            st.markdown(f"**📏 Судья (инструкция):** `{judge_decision_2}`")    
            st.markdown("---")
            st.markdown(f"### 🧪 Пример {len(answers)}")
            st.markdown(f"**🟢 Вопрос:** {question}")
            st.markdown(f"**📥 Системный промт:** `{system_prompt}`")
            st.markdown(f"**🤖 Ответ модели:** `{generated_answer}`")
            st.markdown(f"**🎯 Эталон:** `{expected_answer}`")
        instruction_violations = 0
           
    return answers, correct_answers, wrong_answers, instruction_violations

# ==============================
# Тест кастомного датасета пользователя
# ==============================

st.header("🧪 Тест пользовательского датасета по типам галлюцинаций")

st.write("Кастомный датасет, который Вы формируете самостоятельно должен соответствовать определенным требованиям, " \
"как это указано у нас в шаблоне ниже. Каждая строка каждого столбца должна быть заполнена. " \
"Даже если вы уже вводили системный промпт для этого типа тестирования скопируйте и вставьте его в пропущенные ячейки, " \
"иначе модель не сможет прочитать ваш файл.")

sample_path = os.path.join("Sample", "SampleDataset.xlsx") #делаем возможность скачать образец того, как должен выглядеть кастомный датасет

with open(sample_path, "rb") as f:
    sample_bytes = f.read()

st.download_button( #сделаем саму кнопку
    label="📥 Скачать шаблон кастомного датасета",
    data=sample_bytes,
    file_name="SampleDataset.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

model_names = load_model_list()
selected_model_name = st.selectbox("Выберите модель для пользовательского теста:", model_names, key="custom_model")

user_file = st.file_uploader("Загрузите Excel-файл (Тип запроса | Системный промпт | Промпт | Ответ)", type="xlsx")
if user_file and st.button("🚀 Запустить кастомный тест"):
    df_all = pd.read_excel(user_file)

    expected_columns = {"Тип запроса", "Системный промпт", "Промпт", "Ответ"}
    if not expected_columns.issubset(df_all.columns):
        st.error(f"❌ В файле должны быть колонки: {expected_columns}")
        st.stop()

    model_path = os.path.join("Models", selected_model_name)
    model = load_llama_model(model_path)

    judge_model_dir = os.path.join("Models", "JudgeModel")
    judge_files = [f for f in os.listdir(judge_model_dir) if f.endswith(".gguf")]
    if not judge_files:
        st.error("❌ Не найдена модель-судья в 'Models/JudgeModel'")
        st.stop()
    judge_model = load_llama_model(os.path.join(judge_model_dir, judge_files[0]))

    metric_results = {}
    violation_total = 0
    total_rows = 0

    for halluc_type in ["Контекстный", "Фактологический", "Логический"]:
        df_type = df_all[df_all["Тип запроса"].str.lower() == halluc_type.lower()]
        if df_type.empty:
            continue

        df_eval = pd.DataFrame({
            "prompt": df_type["Промпт"].astype(str),
            "answer": df_type["Ответ"].astype(str)
        })
        system_prompt = df_type["Системный промпт"].iloc[0] if pd.notna(df_type["Системный промпт"].iloc[0]) else ""

        answers, correct, wrong, violations = evaluate_model(
            model=model,
            df=df_eval,
            system_prompt=system_prompt,
            judge_model=judge_model
        )

        #оцениваем результат от 0 до 100%
        total = correct + wrong
        score = round(correct / total * 100, 2) if total else 0
        metric_results[halluc_type] = f"{score}%"

        violation_total += violations
        total_rows += len(df_eval)

    instruction_score = round((1 - violation_total / total_rows) * 100, 2) if total_rows else "N/A"
    metric_results["Следование инструкциям"] = f"{instruction_score}%"

    result_df = pd.DataFrame([{
        "Название модели": selected_model_name,
        "Контекстное соответствие": metric_results.get("Контекстный", "N/A"),
        "Фактологическое соответствие": metric_results.get("Фактологический", "N/A"),
        "Логическое соответствие": metric_results.get("Логический", "N/A"),
        "Следование инструкциям": metric_results.get("Следование инструкциям", "N/A"),
        "Название датасета": user_file.name,
        "Дата и время теста": datetime.now().strftime("%d.%m.%Y %H:%M")
    }])

    # Сохраняем в custom_leaderboard.xlsx
    custom_leaderboard_path = os.path.join("leaderboard", "custom_leaderboard.xlsx")
    os.makedirs("leaderboard", exist_ok=True)
    if os.path.exists(custom_leaderboard_path):
        existing_custom = pd.read_excel(custom_leaderboard_path)
    else:
        existing_custom = pd.DataFrame()
    updated_custom_df = pd.concat([existing_custom, result_df], ignore_index=True)
    updated_custom_df.to_excel(custom_leaderboard_path, index=False)

    st.success("✅ Результаты добавлены в custom_leaderboard.xlsx")
    st.dataframe(result_df, use_container_width=True)

# ==============================
# 📈 Лидерборды моделей
# ==============================

st.header("📊 Лидерборд: кастомные датасеты")
try:
    custom_leaderboard_df = pd.read_excel(os.path.join("leaderboard", "custom_leaderboard.xlsx"))
    sort_custom = st.selectbox("🔽️ Отсортировать по (кастомные):", options=custom_leaderboard_df.columns[1:])
    sorted_custom_df = custom_leaderboard_df.sort_values(by=sort_custom, ascending=False, ignore_index=True)
    st.dataframe(sorted_custom_df, use_container_width=True)
except Exception as e:
    st.warning(f"Не удалось загрузить custom_leaderboard.xlsx: {e}")

# ==============================
# ✅ Блок 2 — Тест на предзагруженных датасетах
# ==============================

st.header("🧪 Тестирование модели на предзагруженных датасетах")
selected_model_name2 = st.selectbox("Выберите модель для оценки:", model_names, key="model2")

dataset_map = {
    "ContextDataset.xlsx": "Контекстное соответствие",
    "FactsDataset.xlsx": "Фактологическое соответствие",
    "LogicDataset.xlsx": "Логическое соответствие"
}

selected_datasets = st.multiselect(
    "Выберите датасеты для оценки:",
    options=list(dataset_map.keys()),
    default=list(dataset_map.keys())
)

if st.button("🔢 Запустить оценку модели для лидерборда"):
    datasets_dir = os.path.join(os.getcwd(), "Datasets")

    model_path = os.path.join("Models", selected_model_name2)
    model = load_llama_model(model_path)

    # Загрузка модели-судьи
    judge_model_dir = os.path.join("Models", "JudgeModel")
    judge_files = [f for f in os.listdir(judge_model_dir) if f.endswith(".gguf")]
    if not judge_files:
        st.error("❌ В папке 'Models/JudgeModel' не найдено модели-судьи (.gguf)")
        st.stop()
    judge_model_path = os.path.join(judge_model_dir, judge_files[0])
    judge_model = load_llama_model(judge_model_path)

    results = {}
    instruction_violations_total = 0
    instruction_rows_total = 0

    for filename, label in dataset_map.items():
        if filename not in selected_datasets:
            results[label] = "N/A"
            continue

        file_path = os.path.join(datasets_dir, filename)

        try:
            full_df = pd.read_excel(file_path, header=None)
            system_prompt = str(full_df.iloc[0, 0]) if pd.notna(full_df.iloc[0, 0]) else ""
            df = pd.read_excel(file_path, header=1)

            df["prompt"] = df["prompt"].astype(str)
            df["answer"] = df["answer"].astype(str)

            answers, correct, wrong, violations = evaluate_model(
                model=model,
                df=df,
                system_prompt=system_prompt,
                judge_model=judge_model
            )

            total = correct + wrong
            score = round(correct / total * 100, 2) if total else 0
            results[label] = f"{score}%"

            instruction_violations_total += violations
            instruction_rows_total += len(df)

        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла '{filename}': {e}")
            results[label] = "N/A"

    # Подсчёт общей метрики следования инструкциям
    if instruction_rows_total > 0:
        instruction_score = round((1 - instruction_violations_total / instruction_rows_total) * 100, 2)
        results["Следование инструкциям"] = f"{instruction_score}%"
    else:
        results["Следование инструкциям"] = "N/A"

    # Сохраняем в leaderboard
    if results:
        leaderboard_path = os.path.join("leaderboard", "leaderboard.xlsx")
        os.makedirs("leaderboard", exist_ok=True)
        if os.path.exists(leaderboard_path):
            existing_df = pd.read_excel(leaderboard_path)
        else:
            existing_df = pd.DataFrame()

        new_row = {
            "Название модели": selected_model_name2,
            "Контекстное соответствие": results.get("Контекстное соответствие", "N/A"),
            "Фактологическое соответствие": results.get("Фактологическое соответствие", "N/A"),
            "Логическое соответствие": results.get("Логическое соответствие", "N/A"),
            "Следование инструкциям": results.get("Следование инструкциям", "N/A"),
            "Дата и время теста": datetime.now().strftime("%d.%m.%Y %H:%M")
        }

        updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        updated_df.to_excel(leaderboard_path, index=False)

        st.success("✅ Результаты модели добавлены в leaderboard.xlsx")
        st.dataframe(updated_df)


# ================================================ #
# Лидерборд предзагруженных моделей и их датасетов #
# ================================================ #

st.header("📊 Лидерборд: предзагруженные датасеты")
try:
    leaderboard_df = pd.read_excel(os.path.join("leaderboard", "leaderboard.xlsx"))
    sort_option = st.selectbox("🔽️ Отсортировать по (предзагруженные):", options=leaderboard_df.columns[1:])
    sorted_df = leaderboard_df.sort_values(by=sort_option, ascending=False, ignore_index=True)
    st.dataframe(sorted_df, use_container_width=True)
except Exception as e:
    st.warning(f"Не удалось загрузить leaderboard.xlsx: {e}")

# ==============================
# 🗂 Опционально: загрузка пользовательской модели
# ==============================
# st.header("📂 Загрузка пользовательской модели")
# uploaded_model = st.file_uploader("Загрузите вашу LLM модель (.gguf)", type=["gguf"], key="custom")
# if uploaded_model:
#     model_folder = os.path.join(os.getcwd(), "Models")
#     os.makedirs(model_folder, exist_ok=True)
#     model_path = os.path.join(model_folder, uploaded_model.name)
#     with open(model_path, "wb") as f:
#         f.write(uploaded_model.getbuffer())
#     st.success(f"Модель успешно сохранена в: {model_path}")
