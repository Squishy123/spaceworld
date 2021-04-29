# 1234 for lunarlander

class HumanAgent(object):

    def act(self, observation, reward, done):
        action = input("Type an Action 1-4:")
        return int(action)
