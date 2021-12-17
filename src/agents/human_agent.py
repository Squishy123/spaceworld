from pynput.keyboard import Listener, Key
# 1234 for lunarlander


class HumanAgent():
    '''
    def __init__(self):
        self.default_action = 0
        self.selected_action = 0

        listener = Listener(
            on_press=self.on_press,

        )
        listener.start()

    def act(self, observation, reward, done):
        print(f"REWARD: {reward}")

        if self.selected_action != 0:
            temp = self.selected_action
            self.selected_action = 0
            return temp

        return self.default_action

    def on_press(self, key):
        if key == Key.w:
            self.selected_action = 2
        elif key == Key.a:
            self.selected_action = 1
        elif key == Key.d:
            self.selected_action = 3
        elif key == Key.q:
            exit(1)
    '''

    def act(self, *args):
        key = input("Enter input:")
        if key == "w":
            return 2
        elif key == "a":
            return 1
        elif key == "d":
            return 3
        elif key == "q":
            return None
        else:
            return 0
