

"""
### NOTICE ###
You DO NOT need to upload this file.
"""
import random, sys
from ale_python_interface import ALEInterface

class ALE(object):
    def __init__(self, init_seed, init_rand):
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', init_seed)
        #self.ale.setFloat('repeat_action_probability', 0.0)
        self.ale.loadROM('./breakout.bin')
        self.action_size = 4

        self.screen = None
        self.reward = 0
        self.terminal = True
        self.init_rand = init_rand

    def setSetting(self, action_repeat, screen_type):
        self.action_repeat = action_repeat
        self.screen_type = screen_type

    def _step(self, action):
        self.reward = self.ale.act(action)
        self.terminal = self.ale.game_over()

        if self.screen_type == 0:
            self.screen = self.ale.getScreenRGB()
        elif self.screen_type == 1:
            self.screen = self.ale.getScreenGrayscale()
        else:
            sys.stderr.write('screen_type error!')
            exit()


    def state(self):
        return self.reward, self.screen, self.terminal

    def act(self, action):
        cumulated_reward = 0
        for _ in range(self.action_repeat):
            self._step(action)
            cumulated_reward += self.reward
            if self.terminal:
                break
        self.reward = cumulated_reward
        return self.state()

    def new_game(self):
        if self.ale.game_over():
            self.ale.reset_game()

            if self.screen_type == 0:
                self.screen = self.ale.getScreenRGB()
            elif self.screen_type == 1:
                self.screen = self.ale.getScreenGrayscale()
            else:
                sys.stderr.write('screen_type error!')
                exit()

        for _ in range(self.init_rand):
            self._step(0)

        return self.screen

