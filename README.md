Решение задачи CartPole-v1 с помощью DQN.

Цель достичь reward >= 475.

Основные идеи взяты из статьи: [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
реализация упрощена.

Основные идеи:

**Replay buffer** - последовательный сэмплинг из среды плохо сказывается на процессе обучения за счет корреляции. 
Поэтому мы храним в буфере состояния среды из разных временных промежутков и сэмплим из него случайно размером 
BATCH_SIZE. Вначале необходимо инициализировать буфер, чтобы он был заполнен до MIN_BUFFER_SIZE.
Также в процессе обучения буффер обновляется последними состояниями среды.

**Epsilon Greedy Policy** - стратегия выбора действия. Вначале мы хотим собрать больше информации о среде, поэтому 
выбираем случайное действие с вероятностью EPSILON_START, которая постепенно уменьшается до EPSILON_END.
Тем самым скармливаем сетке больше информации о среде и даем ей экспериментировать с различными действиями.
Далее уменьшаем вероятность случайного действия, чтобы сеть стала более оптимистичной и начала выбирать нужное действие
на основе опыта.

**Online Net and Target Net** - для устойчивости обучения используется две сети. Одна сеть обучается, а другая
используется для вычисления целевого значения. Параметры сети обновляются с периодом UPDATE_TARGET_NET.

**Immediate reward and Discounted max future reward** - в качестве целевого значения используется сумма наград за
текущий шаг и максимальная награда за следующий шаг. Таким образом мы учитываем награду за текущий шаг и 
предсказываем награду за следующий шаг. За баланс между текущей и будущей наградой отвечает параметр GAMMA.
