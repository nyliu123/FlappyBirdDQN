import os

import warnings
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import random

from pathlib import Path
import time
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent
PRETRAINED_MODEL_DIRNAME = 'pretrained_model'
PRETRAINED_MODEL_DIR = os.path.join(BASE_DIR, PRETRAINED_MODEL_DIRNAME)
BASE_MODEL_DIR = os.path.join(PRETRAINED_MODEL_DIR, 'base')
TARGET_MODEL_DIR = os.path.join(PRETRAINED_MODEL_DIR, 'target')
REWARD_COMPARE_PATH = os.path.join(BASE_DIR, 'reward_compare.png')
REWARD_LOG_BASE_PATH = os.path.join(BASE_DIR, 'reward_base.npy')
REWARD_LOG_TARGET_PATH = os.path.join(BASE_DIR, 'reward_target.npy')

# Game configuration: set to False to disable score display or sound effects
TRAIN_GAME_SHOW_SCORE = False
TRAIN_GAME_PLAY_SOUND = False

TEST_GAME_SHOW_SCORE = True
TEST_GAME_PLAY_SOUND = True

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32
        self.target_sync_interval = 1000
        self.model_save_interval = 50000
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def image_to_tensor(image, device):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor).to(device)
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def get_model_dir(model_type):
    if model_type == 'base':
        return BASE_MODEL_DIR
    if model_type == 'target':
        return TARGET_MODEL_DIR
    raise ValueError('invalid model type')


def ensure_model_dirs():
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    os.makedirs(TARGET_MODEL_DIR, exist_ok=True)


def get_model_path(iteration, model_type):
    return os.path.join(get_model_dir(model_type), f'current_model_{iteration}.pth')


def list_model_iterations(model_type):
    model_dir = get_model_dir(model_type)
    if not os.path.exists(model_dir):
        return []
    iterations = []
    for filename in os.listdir(model_dir):
        match = re.fullmatch(r'current_model_(\d+)\.pth', filename)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def ask_scratch_or_existing(model_type):
    model_dir = get_model_dir(model_type)
    checkpoints = []
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            match = re.fullmatch(r'checkpoint_(\d+)\.pth', filename)
            if match:
                checkpoints.append((int(match.group(1)), os.path.join(model_dir, filename)))
    checkpoints.sort(key=lambda x: x[0])
    if not checkpoints:
        print(f'No checkpoint found for {model_type}, starting from scratch')
        return None
    choice = choose_from_list(f'Start {model_type} training:', ['scratch', 'existing checkpoint'])
    if choice.startswith('scratch'):
        return None
    checkpoint = choose_from_list('Select a checkpoint:', [f'checkpoint at iteration {it}' for it, _ in checkpoints])
    it = int(re.search(r'iteration (\d+)', checkpoint).group(1))
    return os.path.join(get_model_dir(model_type), f'checkpoint_{it}.pth')


def make_noop_action(model, device):
    action = torch.zeros([model.number_of_actions], dtype=torch.float32, device=device)
    action[0] = 1
    return action


def initialize_state(game_state, model, device):
    action = make_noop_action(model, device)
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data, device)
    return torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)


def select_action(model, state, epsilon, device):
    with torch.no_grad():
        output = model(state)[0]
    action = torch.zeros([model.number_of_actions], dtype=torch.float32, device=device)
    random_action = random.random() <= epsilon
    action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.long, device=device) if random_action else torch.argmax(output)
    action[action_index] = 1
    return action, action_index, output


def optimize(model, optimizer, criterion, replay_memory, target_model=None):
    minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))
    state_batch = torch.cat(tuple(d[0] for d in minibatch))
    action_batch = torch.cat(tuple(d[1] for d in minibatch))
    reward_batch = torch.cat(tuple(d[2] for d in minibatch))
    state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
    with torch.no_grad():
        eval_model = target_model if target_model is not None else model
        output_1_batch = eval_model(state_1_batch)
    y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4] else reward_batch[i] + model.gamma * torch.max(output_1_batch[i]) for i in range(len(minibatch))))
    q_value = torch.sum(model(state_batch) * action_batch, dim=1)
    optimizer.zero_grad()
    loss = criterion(q_value, y_batch.detach())
    loss.backward()
    optimizer.step()
    return loss


def step_environment(game_state, state, action, device):
    image_data_1, reward, terminal = game_state.frame_step(action)
    image_data_1 = resize_and_bgr2gray(image_data_1)
    image_data_1 = image_to_tensor(image_data_1, device)
    state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
    reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=device)
    return state_1, action.unsqueeze(0), reward_tensor, terminal, float(reward)


def train(model_type):
    ensure_model_dirs()
    print('Using Device:', device)
    set_seed(2026, device)
    model = NeuralNetwork().to(device)
    model.apply(init_weights)
    resume_from = ask_scratch_or_existing(model_type)
    start = time.time()
    print("Training DQN", "+ Target Network" if model_type == 'target' else "", "...")
    from game.flappy_bird import GameState
    target_model = NeuralNetwork().to(device) if model_type == 'target' else None
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    game_state = GameState(show_score=TRAIN_GAME_SHOW_SCORE, play_sound=TRAIN_GAME_PLAY_SOUND)
    replay_memory = []
    reward_history = []
    episode_reward = 0.0
    episode_count = 0
    iteration = 0
    epsilon = model.initial_epsilon
    if resume_from is not None:
        print(f'Resuming from checkpoint: {resume_from}')
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
        epsilon = checkpoint['epsilon']
        replay_memory = [(s.to(device), a.to(device), r.to(device), s1.to(device), t) for s, a, r, s1, t in checkpoint['replay_memory']]
        reward_history = checkpoint['reward_history']
        episode_reward = checkpoint['episode_reward']
        episode_count = checkpoint['episode_count']
        if target_model is not None:
            target_model.load_state_dict(checkpoint['target_model_state_dict'])
    elif target_model is not None:
        target_model.load_state_dict(model.state_dict())
    if target_model is not None:
        target_model.eval()
    state = initialize_state(game_state, model, device)
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
    try:
        while iteration < model.number_of_iterations:
            selected_action, action_index, output = select_action(model, state, epsilon, device)
            state_1, action, reward, terminal, reward_value = step_environment(game_state, state, selected_action, device)
            replay_memory.append((state, action, reward, state_1, terminal))
            if len(replay_memory) > model.replay_memory_size:
                replay_memory.pop(0)
            epsilon = epsilon_decrements[iteration]
            optimize(model, optimizer, criterion, replay_memory, target_model)
            state = state_1
            episode_reward += reward_value
            if terminal:
                print('iteration:', iteration, 'elapsed time:', time.time() - start, 'epsilon:', epsilon, 'action:', action_index.detach().cpu().numpy(), 'reward:', reward.detach().cpu().numpy()[0][0], 'Q max:', np.max(output.detach().cpu().numpy()))
                reward_history.append(episode_reward)
                episode_reward = 0.0
                state = initialize_state(game_state, model, device)
                episode_count += 1
            if target_model is not None and iteration > 0 and iteration % model.target_sync_interval == 0:
                target_model.load_state_dict(model.state_dict())
            if iteration > 0 and iteration % model.model_save_interval == 0:
                torch.save(model.state_dict(), get_model_path(iteration, model_type))
            iteration += 1
        print("done")
    except KeyboardInterrupt:
        print('\nTraining interrupted by user at iteration', iteration)
        if iteration > 0:
            cpu_replay = [(s.cpu(), a.cpu(), r.cpu(), s1.cpu(), t) for s, a, r, s1, t in replay_memory]
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration,
                'epsilon': epsilon,
                'replay_memory': cpu_replay,
                'reward_history': reward_history,
                'episode_reward': episode_reward,
                'episode_count': episode_count,
            }
            if target_model is not None:
                checkpoint_data['target_model_state_dict'] = target_model.state_dict()
            checkpoint_path = os.path.join(get_model_dir(model_type), f'checkpoint_{iteration}.pth')
            torch.save(checkpoint_data, checkpoint_path)
            print('Checkpoint saved to', checkpoint_path)
            exit_or_return()
    return reward_history


def moving_average(values, window=100):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode='valid')



def plot_reward_compare(base_rewards, target_rewards):
    base_rewards = np.asarray(base_rewards, dtype=np.float32)
    target_rewards = np.asarray(target_rewards, dtype=np.float32)

    smooothed_base_rewards = moving_average(base_rewards)
    smooothed_target_rewards = moving_average(target_rewards)

    base_x = np.arange(len(smooothed_base_rewards))
    target_x = np.arange(len(smooothed_target_rewards))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(base_x, smooothed_base_rewards, color='blue', label='DQN')
    ax.plot(target_x, smooothed_target_rewards, color='red', label='DQN + Target Network')

    ax.set_title('Reward Comparison')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    fig.tight_layout()
    fig.savefig(REWARD_COMPARE_PATH)
    plt.close(fig)


def train_both():
    try:
        base_rewards = train('base')
        np.save(REWARD_LOG_BASE_PATH, np.asarray(base_rewards, dtype=np.float32))
        target_rewards = train('target')
        np.save(REWARD_LOG_TARGET_PATH, np.asarray(target_rewards, dtype=np.float32))
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
        exit_or_return()



def read_single_key():
    if os.name == 'nt':
        import msvcrt
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            ch2 = msvcrt.getch()
            if ch2 == b'H':
                return 'up'
            if ch2 == b'P':
                return 'down'
            if ch2 == b'K':
                return 'left'
            if ch2 == b'M':
                return 'right'
        if ch in (b'\r', b'\n'):
            return 'enter'
        if ch == b'\x1b':
            return 'esc'
        return ch.decode(errors='ignore')

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def choose_from_list(title, items, page_size=10):
    enable_paging = len(items) > page_size
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    page = 0
    index = 0
    while True:
        clear_console()
        start = page * page_size
        stop = min(start + page_size, len(items))
        page_items = list(enumerate(items[start:stop], start=start))
        if enable_paging:
            print(f'{title}  [{page + 1}/{total_pages}]')
        else:
            print(title)
        print()
        for i, item in page_items:
            if i == index:
                print(f'\033[96m> {item}\033[0m')
            else:
                print(f'  {item}')
        print()
        if enable_paging:
            print('NOTE: ↑/↓ select, ←/→ switch page, Enter confirm, Esc exit')
        else:
            print('NOTE: ↑/↓ select, Enter confirm, Esc exit')
        key = read_single_key()
        if key == 'up':
            index = start + ((index - start - 1) % (stop - start))
        elif key == 'down':
            index = start + ((index - start + 1) % (stop - start))
        elif key == 'left' and enable_paging:
            page = (page - 1) % total_pages
            index = page * page_size
        elif key == 'right' and enable_paging:
            page = (page + 1) % total_pages
            index = page * page_size
        elif key == 'enter':
            clear_console()
            return items[index]
        elif key == 'esc':
            clear_console()
            exit(0)

def choose_mode():
    return choose_from_list('Available modes:', ['test', 'train', 'plot', 'play'])

def choose_training_model_type():
    return choose_from_list('Choose training model type:', ['base', 'target', 'both'])

def choose_testing_model_type():
    return choose_from_list('Choose testing model type:', ['base', 'target'])

def choose_iteration(model_type):
    iterations = list_model_iterations(model_type)
    if len(iterations) == 0:
        print(f'error: no checkpoint found in {get_model_dir(model_type)}')
        exit_or_return()
    labels = [f'model at iteration {iteration}' for iteration in iterations]
    selected = choose_from_list(f'Choose iteration from {model_type}:', labels).split('iteration ')[1]
    return int(selected)

def test(model_type):
    it = choose_iteration(model_type)
    path = get_model_path(it, model_type)
    model = NeuralNetwork().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print('Using Device:', device)
    print('Loaded:', path)
    model.eval()
    from game.flappy_bird import GameState
    game_state = GameState(show_score=TEST_GAME_SHOW_SCORE, play_sound=TEST_GAME_PLAY_SOUND)
    state = initialize_state(game_state, model, device)
    scores = []
    print('Testing model... (press Ctrl+C to stop)')
    try:
        with torch.no_grad():
            while True:
                output = model(state)[0]
                action = torch.zeros([model.number_of_actions], dtype=torch.float32, device=device)
                action_index = torch.argmax(output)
                action[action_index] = 1
                state_1, action_tensor, reward_tensor, terminal, _ = step_environment(game_state, state, action, device)
                state = state_1
                if terminal:
                    score = game_state.final_score
                    scores.append(score)
                    print(f'Score: {score}')
                    state = initialize_state(game_state, model, device)
    except KeyboardInterrupt:
        avg = sum(scores) / len(scores) if scores else 0
        print(f'\nEpisodes played: {len(scores)} | Average score: {avg:.1f}')
        exit_or_return()


def choose_plot_type():
    return choose_from_list('Choose data to plot:', ['base', 'target', 'both'])


def plot_single(data, title, save_path):
    rewards = np.asarray(data, dtype=np.float32)
    smoothed = moving_average(rewards)
    x = np.arange(len(smoothed))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, smoothed, color='blue')
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot():
    plot_type = choose_plot_type()

    if plot_type in ('base', 'both'):
        if not os.path.exists(REWARD_LOG_BASE_PATH):
            print(f'Error: {REWARD_LOG_BASE_PATH} not found, run training first')
            exit_or_return()
            return
        base_data = np.load(REWARD_LOG_BASE_PATH)

    if plot_type in ('target', 'both'):
        if not os.path.exists(REWARD_LOG_TARGET_PATH):
            print(f'Error: {REWARD_LOG_TARGET_PATH} not found, run training first')
            exit_or_return()
            return
        target_data = np.load(REWARD_LOG_TARGET_PATH)

    if plot_type == 'base':
        save_path = os.path.join(BASE_DIR, 'reward_base.png')
        plot_single(base_data, 'DQN', save_path)
        print('Plot saved to:', save_path)
    elif plot_type == 'target':
        save_path = os.path.join(BASE_DIR, 'reward_target.png')
        plot_single(target_data, 'DQN + Target Network', save_path)
        print('Plot saved to:', save_path)
    elif plot_type == 'both':
        plot_reward_compare(base_data, target_data)
        print('Plot saved to:', REWARD_COMPARE_PATH)

    exit_or_return()


def main():
    mode = choose_mode()
    match mode:
        case 'test':
            model_type = choose_testing_model_type()    
            test(model_type)
        case 'train':
            model_type = choose_training_model_type()
            if model_type == 'both':
                train_both()
            else:
                reward_history = train(model_type)
                if model_type == 'base':
                    np.save(REWARD_LOG_BASE_PATH, np.asarray(reward_history, dtype=np.float32))
                else:
                    np.save(REWARD_LOG_TARGET_PATH, np.asarray(reward_history, dtype=np.float32))
        case 'play':
            play()
        case 'plot':
            plot()


def play():
    from game.flappy_bird import GameState
    game = GameState(show_score=True, play_sound=True)
    print('Playing Flappy Bird... (press Space to flap, Ctrl+C to stop)')
    print('NOTE: You may tap the window to focus before playing')
    try:
        while True:
            action = [1, 0]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print('\nGame stopped by user')
                    exit_or_return()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = [0, 1]
            _, _, terminal = game.frame_step(action)
            if terminal:
                game = GameState(show_score=True, play_sound=True)
    except KeyboardInterrupt:
        print('\nGame stopped by user')
        exit_or_return()


def exit_or_return():
    print('NOTE: Enter return to main page, Esc exit')
    key = read_single_key()
    clear_console()
    if key == 'enter':
        main()
    else:
        pygame.quit()
        exit(0)

if __name__ == '__main__':
    device = get_device()
    main()
