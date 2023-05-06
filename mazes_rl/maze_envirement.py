from abc import ABC
from typing import Tuple

import gym
from gym import spaces
import pygame
from mazes_generator.maze_generator import generate_mazes


class MazeEnvironment(gym.Env, ABC):
    def __init__(self):
        super(MazeEnvironment, self).__init__()
        self.maze = self.generate_maze()
        self.start_position = (1, 0)
        self.end_position = (19, 20)
        self.current_position = self.start_position
        self.previous_position = self.start_position

        self.observation_space = spaces.Box(low=0, high=4, shape=(21, 21), dtype=int)
        self.action_space = spaces.Discrete(4)

    @staticmethod
    def generate_maze():
        """
        Сгенерировать случайный лабиринт размером 10
        :return: лабиринт
        """
        return generate_mazes(1, 10)[0]

    def reset(self) -> Tuple[int, int]:
        """
        Сбросить состояние среды и сгенерировать лабиринт
        :return: состояние среды на начало эпизода
        """
        self.current_position = self.start_position
        self.previous_position = self.start_position
        self.maze = self.generate_maze()
        return self.get_observation()

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Сделать шаг в среде
        :param action: экшен от 0 до 3
        :return:
        """
        new_position = self.get_new_position(action)

        # Если упираемся в стену, то стоим на месте
        if self.maze[new_position] == 0:
            return self.get_observation(), -1, False, {}

        self.previous_position = self.current_position
        self.current_position = new_position
        # Если дошли до финиша, то получаем +1
        if self.current_position == self.end_position:
            return self.get_observation(), 1, True, {}
        # Если не дошли до финиша, то получаем -0.01
        return self.get_observation(), -0.01, False, {}

    def get_new_position(self, action: int) -> Tuple[int, int]:
        """
        Получить новую позицию агента на основе экшена

        :param action: Действие агента (0 - вправо, 1 - вверх, 2 - влево, 3 - вниз)
        :return: координаты x, y новой позиции
        """
        x, y = self.current_position

        if action == 0:
            return x + 1, y
        elif action == 1:
            return x, y + 1
        elif action == 2:
            return x - 1, y
        elif action == 3:
            return x, y - 1

    def get_observation(self):
        """
        Получить состояние среды
        :return: состояние среды
        """
        # обновляем прошлую последнюю позицию, как посещенная клетка
        self.maze[self.maze == 3] = 2
        # обновляем последнюю посещенную клетку
        self.maze[self.previous_position] = 3
        # обновляем текущую позицию
        self.maze[self.current_position] = 4
        return self.maze

    def render(self, mode="human", close=False):
        """
        Отрисовать среду
        """
        if close:
            pygame.quit()
            return

        if mode == "human":
            cell_size = 30

            if not hasattr(self, "screen"):
                pygame.init()
                self.screen = pygame.display.set_mode((21 * cell_size, 21 * cell_size))
                pygame.display.set_caption("Maze Environment")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.screen.fill((255, 255, 255))

            for i in range(21):
                for j in range(21):
                    color = (255, 255, 255)

                    if self.maze[i, j] == 0:
                        color = (0, 0, 0)
                    elif self.maze[i, j] == 2:
                        color = (128, 128, 128)
                    elif self.maze[i, j] == 3:
                        color = (0, 0, 255)
                    elif self.maze[i, j] == 4:
                        color = (0, 255, 0)

                    pygame.draw.rect(
                        self.screen, color, pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                    )
                    pygame.draw.rect(
                        self.screen, (0, 0, 0), pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size), 1
                    )

            pygame.display.flip()
            pygame.time.wait(100)
