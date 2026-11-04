# Модуль 1: что делать из этого задания

PDF (`cs336_assignment1_basics.pdf`, версия 26.0.3) описывает весь A1 Стэнфорда. В модуль 1 сборного курса входит **только часть**: §3-6 и один прогон на TinyStories. Всё остальное пропускай.

Полное задание и критерии проверки: [задание.md](../../../emigration/обучение/курсы/сборный_курс_LLM/01_трансформер/задание.md). Разборы по лекции 3 (RMSNorm, z-loss, QK-norm, KV-кэш): [заметки.md](../../../emigration/обучение/курсы/сборный_курс_LLM/01_трансформер/заметки.md). Решение сдаётся в `решение.md` рядом с заданием (шаблон в конце `задание.md`).

## Сроки (план от 2026-11-03)

* **к 2026-11-08** -- часть A, модель: тесты `test_model.py` зелёные
* **к 2026-11-15 (КТ1)** -- части B и C: обучение, прогон, генерация

## Часть A. Модель (§3, стр. 13-27)

| # | Задача PDF | Стр. | Адаптер в `tests/adapters.py` | Тест |
|---|---|---|---|---|
| 1 | `linear` | 17-18 | `run_linear` | `test_linear` |
| 2 | `embedding` | 18 | `run_embedding` | `test_embedding` |
| 3 | `rmsnorm` | 19-20 | `run_rmsnorm` | `test_rmsnorm` |
| 4 | `positionwise_feedforward` (SiLU, SwiGLU) | 21-22 | `run_silu`, `run_swiglu` | `test_silu_matches_pytorch`, `test_swiglu` |
| 5 | `rope` | 22-23 | `run_rope` | `test_rope` |
| 6 | `softmax` | 24 | `run_softmax` | `test_softmax_matches_pytorch` (в `test_nn_utils.py`) |
| 7 | `scaled_dot_product_attention` | 23-24 | `run_scaled_dot_product_attention` | `test_scaled_dot_product_attention`, `test_4d_...` |
| 8 | `multihead_self_attention` | 25 | `run_multihead_self_attention`, `..._with_rope` | `test_multihead_self_attention`, `..._with_rope` |
| 9 | `transformer_block` | 26 | `run_transformer_block` | `test_transformer_block` |
| 10 | `transformer_lm` | 26 | `run_transformer_lm` | `test_transformer_lm`, `..._truncated_input` |

Перед задачами прочитать §3.1-3.3.1 (стр. 13-17): схема модели, einsum, порядок осей, инициализация параметров.

**Пропустить:** `transformer_accounting` (стр. 27, письменный подсчёт FLOPs).

**Замер времени:** засеки, сколько уходит на задачи 1-4 до зелёных тестов. По этой цифре уточним срок всего модуля.

## Часть B. Обучение (§4-5, стр. 28-37)

| # | Задача PDF | Стр. | Адаптер | Тест |
|---|---|---|---|---|
| 11 | `cross_entropy` | 28-29 | `run_cross_entropy` | `test_cross_entropy` |
| 12 | `adamw` | 31-32 | `get_adamw_cls` | `test_adamw` |
| 13 | `learning_rate_schedule` | 33 | `run_get_lr_cosine_schedule` | `test_get_lr_cosine_schedule` |
| 14 | `gradient_clipping` | 34 | `run_gradient_clipping` | `test_gradient_clipping` |
| 15 | `data_loading` | 34 | `run_get_batch` | `test_get_batch` |
| 16 | `checkpointing` | 35-36 | `run_save_checkpoint`, `run_load_checkpoint` | `test_checkpointing` |
| 17 | `training_together`: скрипт цикла обучения | 36 | -- | -- |

§4.2 (SGD, стр. 29-31) прочитать: там пример, как устроен `torch.optim.Optimizer`, он нужен для AdamW. Задачу `learning_rate_tuning` (стр. 31) можно сделать быстро, но не обязательно.

**Пропустить:** `adamw_accounting` (стр. 32).

## Часть C. Прогон и генерация (§6-7.2, стр. 37-42)

* 18. `decoding` (стр. 37-38): генерация с temperature и top-p.
* 19. Прогон TinyStories по **Low-Resource Tip на стр. 40**: 40 млн токенов (например 32 × 5000 × 256), цель val loss ≤ 2,00. Конфиг модели -- §7.2.1 (стр. 39). У авторов на M4 Max: 1 ч 22 мин на CPU; на обычном ноутбуке закладывай 1,5-3 ч или бесплатную T4 в Kaggle.
* 20. Советы по отладке -- §7.2.3 (стр. 40), читать до прогона.
* Токенизатор: GPT-2 из `tiktoken` вместо своего BPE. Отметить это в `решение.md`. Vocab size тогда 50257, а не 10000 из §7.2.1.
* Данные: команды `wget` для TinyStories в `README.md` (только две строки TinyStories, OpenWebText не нужен).

**Пропустить:** `experiment_log`, `learning_rate` (подбор, кроме одного рабочего LR), `batch_size_experiment`, абляции §7.3, OpenWebText §7.4, лидерборд §7.5.

## Не делать в модуле 1

* §2 целиком: Unicode, BPE, токенизатор (`test_train_bpe.py`, `test_tokenizer.py`) -- это модуль 3, по желанию.
* `make_submission.sh`, `writeup.pdf`, Gradescope -- это для студентов Стэнфорда.

## Команды

```bash
uv run pytest tests/test_model.py -k "linear or embedding"   # одна-две задачи
uv run pytest tests/test_model.py tests/test_nn_utils.py tests/test_optimizer.py tests/test_data.py tests/test_serialization.py   # весь модуль 1
```

## Учёт времени

| Дата | Задачи | Часы | Где застрял |
|---|---|---|---|
| | | | |
