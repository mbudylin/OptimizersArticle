Материалы для трёх статей по оптимизаторам.

Необходимо предварительно установить солверы, как описано ниже.


## Для статьи 3:
### Локальный запуск

Установить солверы и requirements (python 3.8)

1) cbc: https://ampl.com/products/solvers/open-source/
    прокинуть в .bash_profile путь к бинарнику.

2) ipopt: https://ampl.com/products/solvers/open-source/
    прокинуть в .bash_profile путь к бинарнику.

3) cvxopt отсюда https://cvxopt.org/install/

### Запуск в докере

1. Собрать контейнер из Dockerfile с тегом opt из текущей директории:
```
docker build -t opt .
```

2. Запустить контейнер с mount текущей директории <-> контейнер:
```
docker run -dp 3000:3000 -w /app -v "$(pwd):/app" -i -t opt
```

3. Вставить CONTAINER ID (из команды docker ps) в команду:
```
docker attach ..
```

4. Далее, в контейнере запустить расчёты:
```
python runner.py
```

5. Выйти из контейнера: 
```
exit
```

