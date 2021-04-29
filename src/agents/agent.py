
#RL agent that uses its imagination (dreamed predicted states) to select the best action in a given time step. The predicted states are the results of a pretrained encoder/decoder that is able to approximate the expected pixel representation of a future state given an input pixel space and an action.

class Agent(object):

    #each Agent has an imagination and a given action space
    def __init__(self,action_space,imagination_model):
        self.action_space = action_space
        self.imagination = imagination_model

    # in order for the agent to intelligently make a decision about what to do right now, he uses his imagination to predict what will happen for each available action to him. "If I go left what will happen?" etc. Following a greedy mindset he takes the best action right now.
    def act(self,observation,reward, done):

        imagined_rewards=[]
        for action in self.action_space:
            #query the imagination
            imagined_image,imagined_reward=self.imagination.no_update_step(action)
            imagined_rewards[imagined_reward]=action
        
        #get the action that had the highest reward as an imagined outcome
        return imagined_rewards[max(list(imagined_rewards.keys()))]

    
    #to be even smarter this agent can explore imagination trees, where each node is the result of a step through his imagination. The edges of this tree can be considered the various actions available to the agent. Initial trial of tree depth 2
    def smarter_act(self,observation,reward,done):

        #save imagination model starting state
        imagination_root=self.imagination.model.screen_stack
        imagined_rewards=[]
        for action in self.action_space:
            total_reward=0
            for second_action in self.action_space:
                #query the imagination
                imagined_image,imagined_reward=self.imagination.step(action)
                total_reward+=imagined_reward
                second_imagined_image,second_imagined_reward=self.imagination.step(second_action)
                total_reward+=second_imagined_reward
                #reset imagination to starting state
                self.imagination.model.screen_stack=imagination_root

            imagined_rewards[total_reward]=action

        return imagined_rewards[max(list(imagined_rewards.keys()))]
        







