---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- transformers
language:
- ru
---

# Адаптер type2

QLoRA-адаптер для городского новостного медиа. База: `Qwen/Qwen2.5-1.5B-Instruct`, ревизия `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Адаптер применяется поверх этой базы; он не является самостоятельной моделью.

Обучение выполнено на Tesla T4 в Google Colab: 20 обучающих и 5 validation-пар, 3 эпохи, learning rate `1e-4`, batch size 1, gradient accumulation 2, seed 42. База при обучении — NF4 с double quantization, вычисления — FP16. LoRA: rank 16, alpha 32, dropout 0.05; модули `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Loss считается только по assistant-ответу.

Сохранённый validation loss: `0.8789710998535156`. Это метрика обучения, а не оценка достоверности новостных постов. Полные параметры и версии: `artifacts/train_type2.json` в корне проекта.

В `submission.zip` рядом с `adapter_config.json` лежит исходный `adapter_model.safetensors` (73 911 112 байт; 18 464 768 параметров в FP32). В браузерном комплекте для GitHub этот бинарный файл исключён. Полный архив с весами следует приложить к GitHub Release или передать вместе с репозиторием.

Примеры и оценка находятся в корневых `RESULTS.md` и `REVIEW.md`. Сохраняются фактические и языковые ошибки; текущие ответы требуют редакторской проверки перед публикацией. Запуск vLLM и нагрузочный тест на сервере с 24 GB не выполнялись.
