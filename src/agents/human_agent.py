# 1234 for lunarlander

class HumanAgent(object):

    def act(self, observation, reward, done):
        action = input("wasd:")
        if action == "w":
            return 2
        if action == "a":
            return 1
        if action == "d":
            return 3
        return 0
