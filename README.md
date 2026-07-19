# Contest Helper

`contest-helper` — Python-библиотека и набор CLI-инструментов для подготовки задач к импорту в Яндекс Контест.

Библиотека помогает:

- создавать заготовку новой задачи;
- генерировать входные данные и ответы с помощью эталонного решения;
- сохранять текстовые, бинарные и SQLite-тесты;
- запускать решение на локальном наборе тестов;
- добавлять примеры в условие;
- настраивать метаданные задачи и собирать ZIP-архив;
- создавать checker и postprocessor из шаблонов;
- типографировать условие с сохранением формул и inline-кода.

## Требования

- Python 3.10 или новее;
- доступ к интернету для команды `ch-typograf`;
- пакет `tabulate`, устанавливаемый автоматически как зависимость.

## Установка

Из PyPI:

```bash
python3 -m pip install contest-helper
```

Из исходного кода для разработки:

```bash
git clone https://github.com/kirillcskaslezin/contest-helper.git
cd contest-helper
python3 -m pip install -e .
```

После установки становятся доступны команды с префиксом `ch-`.

## Быстрый старт

Создайте директорию задачи:

```bash
ch-start-problem sum -l ru
cd sum
```

Команда создаст:

```text
sum/
├── generator.py
├── meta.json
└── statement.md
```

С флагом `--checker` также будет создан `checker.py`:

```bash
ch-start-problem sum -l ru --checker
```

Опишите тип входных данных, эталонное решение, генератор и адаптеры в `generator.py`, после чего запустите:

```bash
python3 generator.py
```

Сгенерированные тесты появятся в директории `tests`:

```text
tests/
├── 01
├── 01.a
├── 02
└── 02.a
```

Проверьте решение и соберите архив:

```bash
ch-test ./solution --timeout 2
ch-typograf statement.md
cd ..
ch-combine sum --time-limit 2000 --memory-limit 268435456
```

Результатом будет `sum.zip`.

## Генерация тестов

Основной класс библиотеки — `Generator` из модуля `contest_helper.basic`. Он:

1. очищает или создаёт директорию `tests`;
2. обрабатывает указанные примеры;
3. генерирует случайные тесты;
4. передаёт каждый тест эталонному решению;
5. проверяет результат через необязательный validator;
6. записывает входной файл и ответ с суффиксом `.a`.

Минимальный пример для задачи сложения двух чисел:

```python
from typing import Iterable

from contest_helper.basic import Generator, TextInputAdapter, TextOutputAdapter
from contest_helper.values import CombineValues, RandomNumber


class PairInputAdapter(TextInputAdapter):
    def parse_lines(self, lines: Iterable[str]) -> list[int]:
        a, b = list(lines)[0].split()
        return [int(a), int(b)]

    def input_lines(self, data: list[int]) -> Iterable[str]:
        a, b = data
        return [f"{a} {b}"]


class IntegerOutputAdapter(TextOutputAdapter):
    def output_lines(self, result: int) -> Iterable[str]:
        return [str(result)]


def solution(data: list[int]) -> int:
    return sum(data)


generator = Generator(
    solution=solution,
    tests_generator=CombineValues([
        RandomNumber(1, 101),
        RandomNumber(1, 101),
    ]),
    tests_count=20,
    input_adapter=PairInputAdapter(),
    output_adapter=IntegerOutputAdapter(),
)

generator.run()
```

`RandomNumber(start, stop, step)` использует полуинтервал `[start, stop)`. Например, `RandomNumber(1, 101)` генерирует целые числа от 1 до 100.

### Примеры из файлов

Пути к входным файлам можно передать через `samples`. Адаптер разбирает содержимое, эталонное решение вычисляет ответ, а исходный формат примера сохраняется:

```python
generator = Generator(
    solution=solution,
    samples=["samples/01.txt", "samples/02.txt"],
    tests_generator=CombineValues([
        RandomNumber(1, 101),
        RandomNumber(1, 101),
    ]),
    tests_count=10,
    input_adapter=PairInputAdapter(),
    output_adapter=IntegerOutputAdapter(),
)
```

Примеры получают имена `sample01`, `sample02` и так далее.

### Несколько групп генераторов

Для нескольких наборов входных данных передайте параллельные списки генераторов и количества тестов:

```python
from contest_helper.basic import Generator, TextInputAdapter, TextOutputAdapter
from contest_helper.values import RandomNumber


generator = Generator(
    solution=lambda value: value * value,
    tests_generator=[
        RandomNumber(1, 11),
        RandomNumber(1_000, 10_001),
    ],
    tests_count=[5, 10],
    input_adapter=TextInputAdapter(),
    output_adapter=TextOutputAdapter(),
)

generator.run()
```

Тесты всех групп нумеруются последовательно: `01`, `02`, `03` и далее.

### Проверка сгенерированных тестов

`validator` получает входные данные и ответ эталонного решения. Если он возвращает `False`, тест отбрасывается и генерируется заново:

```python
def validator(data: list[int], result: int) -> bool:
    return data[0] != data[1] and result >= 0
```

Тест также можно отклонить непосредственно из эталонного решения:

```python
from contest_helper.exceptions import BadTestException


def solution(data: list[int]) -> int:
    if data[1] == 0:
        raise BadTestException("division by zero")
    return data[0] // data[1]
```

## Генераторы значений

Генераторы находятся в `contest_helper.values` и являются вызываемыми объектами без аргументов.

### Константы и функции

```python
import time

from contest_helper.values import Lambda, Value

constant = Value(42)
timestamp = Lambda(time.time)

print(constant())
print(timestamp())
```

### Случайные значения

```python
from contest_helper.values import RandomNumber, RandomValue, RandomWord

color = RandomValue(["red", "green", "blue"])
number = RandomNumber(0, 100)
word = RandomWord(min_length=5, max_length=12)
```

Доступны:

- `RandomValue(sequence)` — случайный элемент последовательности;
- `RandomNumber(start, stop, step=1)` — число из заданного диапазона;
- `RandomWord(...)` — случайная строка;
- `RandomSentence(...)` — последовательность случайных слов;
- `RandomList(generator, length)` — список;
- `RandomSet(generator, length)` — множество уникальных элементов;
- `RandomDict(key_generator, value_generator, length)` — словарь;
- `CombineValues(sequence)` — список результатов нескольких генераторов.

Пример матрицы 5 × 5:

```python
from contest_helper.values import RandomList, RandomNumber

matrix = RandomList(
    RandomList(RandomNumber(0, 10), length=5),
    length=5,
)

print(matrix())
```

Размер коллекции также может быть генератором:

```python
values = RandomList(
    RandomNumber(0, 100),
    length=RandomNumber(1, 11),
)
```

При использовании `RandomSet`, `RandomDict` или уникальных столбцов базы данных генератор должен уметь выдать достаточное количество разных значений.

## Даты и время

Дополнительные генераторы импортируются из `contest_helper.extra.datetime`:

```python
from contest_helper.extra.datetime import RandomDate, RandomDateTime, RandomTime

date_generator = RandomDate(
    "2025-01-01",
    "2025-12-31",
    strftime="%d.%m.%Y",
)

time_generator = RandomTime("09:00", "18:00", step_seconds=60)

datetime_generator = RandomDateTime(
    "2025-01-01 00:00:00",
    "2025-01-31 23:59:59",
)
```

Границы диапазонов даты и времени включаются.

## Адаптеры ввода и вывода

Адаптеры отделяют структуру данных Python от формата тестовых файлов.

### Текстовые адаптеры

Для входных данных наследуйте `TextInputAdapter` и при необходимости переопределите:

- `parse_lines(lines)` — чтение примера из файла;
- `input_lines(data)` — сериализацию сгенерированного теста.

Для ответа наследуйте `TextOutputAdapter` и переопределите `output_lines(result)`.

```python
from typing import Iterable

from contest_helper.basic import TextInputAdapter, TextOutputAdapter


class ListInputAdapter(TextInputAdapter):
    def parse_lines(self, lines: Iterable[str]) -> list[int]:
        return [int(value) for value in list(lines)[1].split()]

    def input_lines(self, values: list[int]) -> Iterable[str]:
        return [str(len(values)), " ".join(map(str, values))]


class ListOutputAdapter(TextOutputAdapter):
    def output_lines(self, values: list[int]) -> Iterable[str]:
        return [" ".join(map(str, values))]
```

### Бинарные адаптеры

Для бинарных тестов используются `BinaryInputAdapter` и `BinaryOutputAdapter`:

```python
from contest_helper.basic import BinaryInputAdapter, BinaryOutputAdapter


class BytesInputAdapter(BinaryInputAdapter):
    def parse_bytes(self, blob: bytes) -> bytes:
        return blob

    def input_bytes(self, data: bytes):
        return [data]


class BytesOutputAdapter(BinaryOutputAdapter):
    def output_bytes(self, result: bytes):
        return [result]
```

## Генерация SQLite-баз

Модуль `contest_helper.extra.db` позволяет генерировать связанные таблицы и использовать SQLite-файл как вход или ответ задачи.

```python
import random

from contest_helper.extra.db import (
    ColumnSpec,
    ForeignKey,
    SQLiteConnectionDataBase,
    Table,
)


users = Table(
    name="users",
    rows=10,
    columns={
        "id": ColumnSpec(lambda: random.randint(1, 1_000_000), unique=True),
        "name": ColumnSpec(lambda: f"user_{random.randint(1, 9999)}"),
    },
)

posts = Table(
    name="posts",
    rows=20,
    columns={
        "id": ColumnSpec(lambda: random.randint(1, 1_000_000), unique=True),
        "user_id": ColumnSpec(ForeignKey("users", "id")),
        "title": ColumnSpec(lambda: f"post_{random.randint(1, 9999)}"),
    },
)

database = SQLiteConnectionDataBase(users, posts)
connection = database()
```

Для записи базы в тестовый файл используйте `SQLiteConnInputAdapter` или `SQLiteConnOutputAdapter`.

## CLI-команды

### `ch-start-problem`

Создаёт новую директорию задачи из встроенных шаблонов:

```bash
ch-start-problem DIRECTORY [options]
```

Основные параметры:

| Параметр | Назначение |
|---|---|
| `-l, --language {en,ru}` | Язык шаблона условия |
| `-c, --checker` | Создать `checker.py` |
| `-i, --input-type {text,binary}` | Тип входного адаптера |
| `-o, --output-type {text,binary}` | Тип выходного адаптера |

Пример:

```bash
ch-start-problem graph -l ru -c
```

### `ch-test`

Запускает решение на всех входных файлах из локальной директории `tests`. Для каждого входного файла ожидается файл ответа с тем же именем и суффиксом `.a`.

```bash
ch-test SOLUTION [-t SECONDS] [-c CHECKER] [-i INTERPRETER]
```

Примеры:

```bash
ch-test ./solution
ch-test ./solution --timeout 3 --checker ./checker
ch-test solution.py --interpreter python3
```

Без checker результаты сравниваются как текст после удаления пробельных символов по краям всего вывода.
Файл решения должен быть исполняемым даже при использовании `--interpreter`; при необходимости выполните `chmod +x solution.py`.

### `ch-statement-preview`

Находит файлы `tests/sampleNN` и `tests/sampleNN.a` и добавляет оформленные примеры в условие:

```bash
ch-statement-preview DIRECTORY --lang ru
```

Результат можно записать в другой файл:

```bash
ch-statement-preview DIRECTORY --lang ru --output preview.md
```

Команда дописывает примеры к существующему `statement.md`. При повторном запуске в тот же файл ранее добавленные примеры автоматически не удаляются.

### `ch-typograf`

Отправляет содержимое файла в веб-сервис «Типограф» Студии Артемия Лебедева и перезаписывает файл результатом:

```bash
ch-typograf statement.md
```

Команда сохраняет без изменений:

- формулы `$...$`;
- блочные формулы `$$...$$`;
- inline-code в обратных кавычках.

Перед запуском рекомендуется сохранить изменения в системе контроля версий: файл обновляется на месте. Команде требуется сетевой доступ к `typograf.artlebedev.ru`.

### `ch-combine`

Обновляет `meta.json` по содержимому директории и параметрам командной строки, после чего создаёт ZIP-архив для импорта:

```bash
ch-combine DIRECTORY [options]
```

Пример:

```bash
ch-combine sum \
  --time-limit 2000 \
  --memory-limit 268435456 \
  --checker-files checker.py \
  --solutions python3_13:solution.py
```

Поддерживаемые настройки:

| Параметр | Назначение |
|---|---|
| `--checker-files ...` | Файлы checker |
| `--compile-files ...` | Файлы, добавляемые при компиляции |
| `--run-files ...` | Файлы, добавляемые при запуске |
| `--post-files ...` | Файлы postprocessor |
| `--solutions ...` | Авторские решения в формате `compiler_id:path` |
| `--time-limit` | Ограничение времени, мс |
| `--idleness-limit` | Ограничение бездействия, мс |
| `--memory-limit` | Ограничение памяти, байты |
| `--output-limit` | Ограничение вывода, байты |
| `--input-file` | Имя входного файла |
| `--output-file` | Имя выходного файла |
| `--disable-stdin` | Отключить перенаправление stdin |
| `--disable-stdout` | Отключить перенаправление stdout |
| `--hide-limits` | Скрыть ограничения в условии |
| `--hide-io` | Скрыть секции ввода и вывода |
| `--hide-samples` | Скрыть примеры |

Команда изменяет `DIRECTORY/meta.json` перед упаковкой. Архив создаётся рядом с директорией и получает имя `DIRECTORY.zip`.

### `ch-make-checker`

Создаёт `checker.py` из встроенного шаблона в текущей директории:

```bash
ch-make-checker
```

### `ch-make-postprocessor`

Создаёт настроенный `postprocessor.py` в текущей директории:

```bash
ch-make-postprocessor --max-value 100 --groups "3,5,2" --by-groups
```

Параметры:

- `--max-value` — максимальный балл;
- `--groups` — размеры групп через запятую;
- `--different` — дифференцированное оценивание;
- `--by-groups` — оценивание по группам.

### `ch-compilers`

Ищет доступные компиляторы по части названия без учёта регистра:

```bash
ch-compilers python
ch-compilers "c++"
```

Команда выводит идентификатор компилятора, который можно использовать в `--solutions` при сборке задачи.

## Структура проекта задачи

Типичная рабочая директория выглядит так:

```text
problem/
├── checker.py
├── generator.py
├── meta.json
├── postprocessor.py
├── solution.py
├── statement.md
└── tests/
    ├── sample01
    ├── sample01.a
    ├── 01
    ├── 01.a
    └── ...
```

Обязательный для `ch-combine` файл — `meta.json`. Остальные файлы добавляются в архив при наличии.

## Разработка

Запуск тестов:

```bash
python3 -m unittest discover -s tests -v
```

Проверка синтаксиса пакета:

```bash
python3 -m compileall -q contest_helper tests
```

## Лицензия

Проект распространяется по лицензии [MIT](LICENSE).
