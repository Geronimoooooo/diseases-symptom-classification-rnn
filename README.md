# Diseases Symptom Classification — RNN-классификатор болезней

> **RU.** Классификация десяти ЖКТ-заболеваний по текстовым
> описаниям симптомов, сеть `Embedding → LSTM → Dense`.
>
> **EN.** Classifying ten GI diseases from free-form symptom
> descriptions with an `Embedding → LSTM → Dense` network.

---

## Описание / Overview

**RU.** Для каждого класса (аппендицит, гастрит, гепатит, дуоденит,
колит, панкреатит, холицестит, эзофагит, энтерит, язва) дан один
текстовый файл. Ноутбук читает все файлы, делит каждый текст 80/20
на train/test (чтобы выборки не пересекались), балансирует классы
обрезкой до 90-го перцентиля длины, токенизирует текст, нарезает
скользящим окном по 50 слов с шагом 10 и обучает сеть
`Embedding → LSTM → Dense` с `SpatialDropout1D`. Целевые метрики
задания — точность ≥ 30 % и корректное распознавание не менее 6
классов из 10.

**EN.** Each class (appendicitis, gastritis, hepatitis, duodenitis,
colitis, pancreatitis, cholecystitis, esophagitis, enteritis, ulcer)
is represented by a single text file. The notebook reads every file,
splits each text 80/20 into train/test internally (so the two
subsets never overlap), balances classes by truncating to the 90th
percentile of length, tokenises the text, slices it into 50-word
windows with stride 10, and trains an `Embedding → LSTM → Dense`
network with `SpatialDropout1D`. The assignment targets are
accuracy ≥ 30 % and correctly classifying at least 6 of the 10
diseases.

## Датасет / Dataset

- **Источник / Source:**
  <https://storage.yandexcloud.net/aiueducation/Content/base/l8/diseases.zip>
- ZIP скачивается и распаковывается автоматически в первой секции
  ноутбука.
- **Целевая переменная / Target:** `CLASS_LIST` — список из 10
  классов.

## Стек / Stack

- Python 3.11
- `numpy`, `matplotlib`, `gdown`
- `scikit-learn` (`confusion_matrix`, `ConfusionMatrixDisplay`)
- `tensorflow` / `keras` (`Embedding`, `LSTM`, `SpatialDropout1D`,
  `Dense`, `Dropout`, `Tokenizer`, `utils.to_categorical`)

## Структура / Structure

```
diseases-symptom-classification-rnn/
├── README.md
└── diseases_symptom_classification_rnn.ipynb
```

Логические разделы / notebook sections:

1. Импорты / Imports
2. Загрузка датасета / Dataset download
3. Чтение текстов и train/test split
4. Статистика и балансировка / Statistics & balancing
5. Токенизация / Tokenisation
6. Векторизация скользящим окном / Sliding-window vectorisation
7. Helpers: `plot_history`, `evaluate_classifier`,
   `compile_train_evaluate`
8. Модель `Embedding → LSTM → Dense`
9. Выводы / Conclusions

## Результаты / Results

**RU.**

- Сеть уверенно проходит целевой уровень ≥ 30 % точности и корректно
  распознаёт 6 и более классов из 10.
- Обрезка до 90-го перцентиля длины уравнивает объёмы классов —
  особенно убирает доминирование «длинных» описаний (Колит,
  Панкреатит).
- `SpatialDropout1D` на эмбеддингах лучше обычного `Dropout` для
  текстов: он обнуляет целые каналы признаков, а не отдельные токены.

**EN.**

- The network comfortably beats the 30 % accuracy target and
  correctly classifies six or more of the ten classes.
- Truncating to the 90th percentile of length equalises class sizes,
  specifically removing the dominance of the long descriptions
  (Колит, Панкреатит).
- `SpatialDropout1D` on the embeddings is more appropriate for text
  than plain `Dropout`: it zeros out entire feature channels rather
  than individual tokens.

## Как запустить / How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy matplotlib gdown scikit-learn tensorflow jupyter

jupyter notebook diseases_symptom_classification_rnn.ipynb
```

**RU.** Архив (~0.5 МБ) скачивается автоматически в первой секции
ноутбука.

**EN.** The archive (~0.5 MB) is downloaded automatically in the
notebook's first section.
