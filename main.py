# import pybullet_envs
from agent2 import Agent
from utils import plot_learning_curve
# from gym import wrappers
import random
import pygame
from pygame.locals import *
import numpy as np
import gym
import sys
from pygame import Vector2
from math import *

pygame.init()
print('___ START _____')

# Initialize our game
window_width, window_height = 1920, 1080
rotation_max, acceleration_max = 0.08, 0.5
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

SHIP_W, SHIP_H = 17, 21
TARGET_W, TARGET_H = 5, 5
SPEED_INCREMENT = 12
SPEED_DECREMENT = 12
MAX_SPEED = 30  # 5m/s
MAX_SPEED_TARGET = 3
ROT_SPEED = 12

# reward functions parameters
p1, p2, p3, p4, p5, p6 = 0.01*100, 9, 1, 0.01*3, 0.002*3, 10*30


class CustomEnv(gym.Env):
    def __init__(self, env_config=None):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        if env_config is None:
            env_config = {}
        self.clock = None
        self.window = None
        self.font = pygame.font.Font('SFPixelate.ttf', 24)
        self.score = 0

        # ─── Ship ──────────────────────────────────────────
        self.pos = Vector2(window_width / 2, window_height / 2)
        self.speed = 0
        self.rot = 0.
        self.rot_speed = ROT_SPEED
        self.surf = pygame.Surface((SHIP_W, SHIP_H))

        # ─── Target ──────────────────────────────────────────
        n1 = random.choice((random.randint(300, 800), random.randint(1120, 1620)))
        n2 = random.choice((random.randint(200, 550), random.randint(630, 880)))
        self.target_pos = Vector2(n1, n2)
        self.target_speed = random.randint(int(MAX_SPEED_TARGET / 4), MAX_SPEED_TARGET)
        self.target_rot = 0.
        self.target_rot_speed = ROT_SPEED
        self.target_surf = pygame.Surface((TARGET_H, TARGET_H))

    def init_render(self):
        # Set the game
        pygame.init()
        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()

    def draw_text(self, txt, pos, color=WHITE):
        # Allows the display of text on screen
        text_surf = self.font.render(txt, 1, color)
        self.window.blit(text_surf, pos)

    def reset(self):
        # Reset the positions and velocities of the ship and target
        self.score = 0
        # ─── Ship ──────────────────────────────────────────
        self.pos = Vector2(window_width / 2, window_height / 2)
        self.speed = 0
        self.rot = 0.
        self.rot_speed = ROT_SPEED
        self.surf = pygame.Surface((SHIP_W, SHIP_H))

        # ─── Target ──────────────────────────────────────────
        n1 = random.choice((random.randint(300, 800), random.randint(1120, 1620)))
        n2 = random.choice((random.randint(200, 550), random.randint(630, 880)))
        self.target_pos = Vector2(n1, n2)
        self.target_speed = random.randint(int(MAX_SPEED_TARGET / 4), MAX_SPEED_TARGET)
        self.target_rot = 0.
        self.target_rot_speed = ROT_SPEED
        self.target_surf = pygame.Surface((TARGET_H, TARGET_H))

        obs = self.get_obs()
        return obs

    def get_obs(self):
        auv_x = self.pos.x
        auv_y = self.pos.y
        auv_speed = self.speed
        auv_rot = self.rot
        sin_psi = sin(auv_rot * pi / 180.0)  # Convert degrees to rads then calculate x component
        cos_psi = cos(auv_rot * pi / 180.0)

        target_x = self.target_pos.x
        target_y = self.target_pos.y
        target_speed = self.target_speed
        target_rot = self.target_rot
        target_rot_speed = self.target_rot_speed
        sin_psi_target = sin(target_rot * pi / 180.0)  # Convert degrees to rads then calculate x component
        cos_psi_target = cos(target_rot * pi / 180.0)
        distance_x = target_x-auv_x
        distance_y = target_y - auv_y
        obs = [target_x, target_y, distance_x, distance_y, cos_psi_target, sin_psi_target, target_speed,
               target_rot_speed, cos_psi, sin_psi]
        # obs = [target_x, target_y, auv_x, auv_y, cos_psi_target,
        # sin_psi_target, cos_psi, sin_psi, target_speed, target_rot_speed]
        return np.array(obs)

    def step(self, ac, epoch, i_done_):
        self.movements(time_passed_secs, epoch, ac)

        # ─── Calculate reward ──────────────────────────────────
        # reward = 0
        game_over = False
        dist_ = sqrt((self.pos.x - self.target_pos.x) ** 2 + (self.pos.y - self.target_pos.y) ** 2)
        on_wall = 0
        if self.pos.x == 0 or self.pos.x == window_width or self.pos.y == 0 or self.pos.y == window_height:
            on_wall = 10
        diff_speed = abs(self.speed - self.target_speed)
        rot_speed = ac[1] * self.rot_speed
        diff_speed_ang = abs((rot_speed - self.target_rot_speed) % 180)
        d_los = 200

        r = -1 * p1 * (sqrt(dist_) - sqrt(d_los)) - p4 * diff_speed - p5 * diff_speed_ang - on_wall + p6 * i_done_  # r1
        # if epoch < 500:
        #     if dist_ < d_los:
        #         game_over = True
        # else:
        if dist_ < d_los and i_done_ == 1:
            game_over = True
        #    LOS_link = 'LOS Link activated !'

        reward_, d = r, game_over
        obs = self.get_obs()
        return obs, reward_, d

    def movements(self, time_passed_, epoch, ac=np.zeros(2, dtype=np.float64)):
        # action[0]: acceleration | action[1]: rotation
        mov_direction = -1

        # ─── APPLY ACCELERATION ──────────────────────────────────────────
        acceleration = ac[0]
        # self.speed = self.speed + acceleration * SPEED_INCREMENT
        self.speed = acceleration * MAX_SPEED
        if self.speed > MAX_SPEED:
            self.speed = MAX_SPEED

        if self.speed < -MAX_SPEED:
            self.speed = -MAX_SPEED

        # ─── APPLY ROTATION ──────────────────────────────────────────────
        rot_direction = ac[1]
        # move rocket

        rotated_ship_surf = pygame.transform.rotate(self.surf, self.rot)
        sw, sh = rotated_ship_surf.get_size()

        # This is for the spaceship's rotation--time based
        self.rot += (rot_direction * self.rot_speed * time_passed_)
        self.rot = self.rot % 360

        # Returns the drawing position and where it's heading
        heading_x = sin(self.rot * pi / 180.0)  # Convert degrees to rads then calculate x component
        heading_y = cos(self.rot * pi / 180.0)  # Convert degrees to rads then calculate y component

        ship_draw_pos, s_heading = Vector2(self.pos.x - sw / 2, self.pos.y - sh / 2), Vector2(heading_x, heading_y)
        s_heading *= mov_direction
        self.pos += s_heading * time_passed_ * self.speed

        # ─── Mouvements target ──────────────────────────────────────────────
        acc_target = random.randint(-1, 1)
        # acc_target = 1
        if acc_target == 1:
            self.target_speed = self.target_speed + acc_target * SPEED_INCREMENT
        elif acc_target == -1:
            self.target_speed = self.target_speed + acc_target * SPEED_DECREMENT

        if self.target_speed > MAX_SPEED_TARGET:
            self.target_speed = MAX_SPEED_TARGET

        if self.target_speed < 0:
            self.target_speed = 0

        # self.target_speed = self.target_speed + acc_target * SPEED_INCREMENT

        acc_rot_dir = random.randint(-1, 1)

        self.target_rot_speed = random.randint(0, 180)
        rotated_target_surf = pygame.transform.rotate(self.target_surf, self.target_rot)
        tw, th = rotated_target_surf.get_size()

        # This is for the spaceship's rotation--time based
        self.target_rot += acc_rot_dir * self.target_rot_speed * time_passed_
        self.target_rot = self.target_rot % 360

        # Returns the drawing position and where it's heading
        heading_target_x = sin(self.target_rot * pi / 180.0)  # Convert degrees to rads then calculate x component
        heading_target_y = cos(self.target_rot * pi / 180.0)  # Convert degrees to rads then calculate y component

        target_draw_pos, t_heading = Vector2(self.target_pos.x - tw / 2, self.target_pos.y - th / 2), \
                                     Vector2(heading_target_x, heading_target_y)

        t_heading *= mov_direction
        self.target_pos += t_heading * time_passed_ * self.target_speed

        # ─── Keep the objects on screen ──────────────────────────────────
        if self.pos.x > window_width:
            self.pos.x = self.pos.x - window_width
        elif self.pos.x < 0:
            self.pos.x = self.pos.x + window_width
        if self.pos.y > window_height:
            self.pos.y = self.pos.y - window_height
        elif self.pos.y < 0:
            self.pos.y = self.pos.y + window_height

        # if self.pos.y < 0:
        #     self.pos.y = 0
        # if self.pos.y > window_height:
        #     self.pos.y = window_height
        # if self.pos.x < 0:
        #     self.pos.x = 0
        # if self.pos.x > window_width:
        #     self.pos.x = window_width

        if self.target_pos.x > window_width:
            self.target_pos.x = self.target_pos.x - window_width
        elif self.target_pos.x < 0:
            self.target_pos.x = self.target_pos.x + window_width
        if self.target_pos.y > window_height:
            self.target_pos.y = self.target_pos.y - window_height
        elif self.target_pos.y < 0:
            self.target_pos.y = self.target_pos.y + window_height

        # if self.target_pos.x > window_width - 150:
        #     self.target_pos.x = window_width - 150
        # elif self.target_pos.x < 150:
        #     self.target_pos.x = 150
        # if self.target_pos.y > window_height - 120:
        #     self.target_pos.y = window_height - 120
        # elif self.target_pos.y < 120:
        #     self.target_pos.y = 120

        self.window.blit(rotated_target_surf, target_draw_pos)
        self.window.blit(rotated_ship_surf, ship_draw_pos)

    def render(self):
        pygame.draw.aaline(self.surf, GREEN, [0, 20], [8, 0])
        pygame.draw.aaline(self.surf, GREEN, [8, 0], [16, 20])

        pygame.draw.circle(self.target_surf, RED, [3, 3], 2)


def exit_game():
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    env = CustomEnv()
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_games = 1000
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    # env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'AUV.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        cntr = 0
        i_done = 0
        total_time_passed_secs = 0
        env.init_render()
        while not done:
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[K_ESCAPE]:
                exit_game()

            env.window.fill(BLACK)
            time_passed = env.clock.tick(30)
            time_passed_secs = time_passed / 1000.0
            total_time_passed_secs += time_passed_secs
            if total_time_passed_secs > 300:
                break

            action = agent.choose_action(observation)

            distance = sqrt((env.pos.x - env.target_pos.x) ** 2 + (env.pos.y - env.target_pos.y) ** 2)
            diff_speed_ = abs(env.speed - env.target_speed)
            diff_speed_ang_ = abs(((action[1] * env.rot_speed) - env.target_rot_speed) % 180)
            LOS_link = 'LOS Link waiting'
            if distance < 200:
                cntr += time_passed_secs
                LOS_link = 'LOS Link activated !'
            ep_min = i/100
            if cntr > ep_min:
                i_done = 1

            observation_, reward, done = env.step(action, i, i_done)
            score += reward
            printed_epoch = env.font.render('Episode %.1f' % i, 1, WHITE).get_rect()
            printed_score = env.font.render('Score %.1f' % score, 1, WHITE).get_rect()
            printed_time = env.font.render('Time = %s' % str(int(total_time_passed_secs)), 1, WHITE).get_rect()
            printed_link = env.font.render(LOS_link, 1, WHITE).get_rect()
            printed_dist = env.font.render('distance = %.1f ' % distance, 1, WHITE).get_rect()
            printed_r1 = env.font.render('p1*sqrt(dist-100) = %.3f ' % (p1*(sqrt(distance) - 10)), 1, WHITE).get_rect()
            printed_r2 = env.font.render('p4 * diff_speed = %.3f ' % (p4*diff_speed_), 1, WHITE).get_rect()
            printed_r3 = env.font.render('p5 * diff_speed_ang_ = %.3f ' % (p5 * diff_speed_ang_), 1, WHITE).get_rect()
            printed_actions1 = env.font.render('vitesse = %.3f ' % action[0], 1, WHITE).get_rect()
            printed_actions0 = env.font.render('rotation = %.3f ' % action[1], 1, WHITE).get_rect()

            env.draw_text('Episode %.1f' % i, (10, 10))
            env.draw_text('Score %.1f' % score, (10, 50))
            env.draw_text('Time = %s s' % str(int(total_time_passed_secs)), (10, 90))
            env.draw_text(LOS_link, (10, 130))
            env.draw_text('distance = %.1f ' % distance, (1710, 10))
            env.draw_text('p1*sqrt(dist)-14 = %.3f ' % (p1*(sqrt(distance) - 10)), (1550, 960))
            env.draw_text('p4 * diff_speed = %.3f ' % (p4*diff_speed_), (1550, 1000))
            env.draw_text('p5 * diff_speed_ang_ = %.3f ' % (p5 * diff_speed_ang_), (1500, 1040))
            env.draw_text('vitesse = %.3f ' % action[0], (10, 1000))
            env.draw_text('rotation = %.3f ' % action[1], (10, 1040))

            env.render()
            pygame.display.update()
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_per_sec = score / total_time_passed_secs
        score_history.append(score_per_sec)

        avg_score_per_sec = np.mean(score_history[-100:])

        if avg_score_per_sec > best_score:
            best_score = avg_score_per_sec
        if not load_checkpoint:
            agent.save_models()

        print('episode ', i,
              'score %.1f' % score,
              'score/s %.1f' % score_per_sec,
              'avg_score/s %.1f' % avg_score_per_sec,
              'Time = %.1f' % total_time_passed_secs)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
