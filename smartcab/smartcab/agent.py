import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pylab as pl

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.no_trials = 1       # Counter
        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed


    def reset(self, destination=None, testing=False, a = 0.99):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Increase trial counter
        self.no_trials += 1
        # Update epsilon using a decay function of your choice
        #self.epsilon = self.epsilon - 0.05
        self.epsilon = a**self.no_trials
        #self.epsilon = 1**(-self.no_trials**0.5) # Strecthed exponential decay
        #self.epsilon = 1/(self.no_trials**2)
        #self.epsilon = math.exp(-a*self.no_trials)
        #self.epsilon = math.cos(a*self.no_trials)
        #self.epsilon = math.atan(a/self.no_trials)
        
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing == True:
            self.epsilon = 0
            self.alpha = 0
        
        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent        
        state = (waypoint, inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'])

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########      
        # Calculate the maximum Q-value of all actions for a given state
        maxQ = 0

        for action in self.valid_actions:
            if self.Q[state][action] > maxQ:
                maxQ = self.Q[state][action]
               
        return maxQ 
        
        # Another way to find maxQ
        # maxQ = max(self.Q[state])


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0

        # For reference, self.Q = {state: {action:}}
        # Creating new dictionary for the state
        if self.learning == True:
            if state not in self.Q.keys():
                self.Q[state] = dict()
                self.Q[state] = {self.valid_actions[0]: 0.0, self.valid_actions[1]: 0.0,
                self.valid_actions[2]: 0.0, self.valid_actions[3]: 0.0}

        # Another approach using dict comprehension:
        # if (self.learning) and (state not in self.Q):
        #    self.Q[state] = {action: 0.0 for action in self.valid_actions}
        return


    def choose_action(self, state, maxQ_func):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()                                                   
        ########### 
        ## TO DO ##
        ###########
       
        # When not learning, choose a random action
        action = None
        
        # Action list to contain action associated with maxQ values
        action_list = []

        if self.learning == False:
            action = random.choice(self.valid_actions) 
            
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        else:
            if self.epsilon > random.random():
                action = random.choice(self.valid_actions)
            else:
                for final_action in self.valid_actions:
                    if self.Q[state][final_action] >= maxQ_func(state):
                        action = final_action
                        '''action_list.append[final_action]
                        action = random.choice(action_list) # Takes random action associated to highest Q value of the state
                action = random.choice({final_action for final_action in self.valid_actions if self.Q[state][final_action] >= maxQ_func(state)})'''
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        # Iteration using Bellman equation
        #self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + reward*self.alpha
        if self.learning == True:
            self.Q[state][action] += self.alpha*(reward-self.Q[state][action])
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state, self.get_maxQ)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """
    #for i in pl.frange(0.1, 1.0, 0.1):
        #for j in pl.frange(0.1, 1.0, 0.1):
    i = 0.6
    j = 0.6


    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose =True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, epsilon = i, alpha = j, learning =True)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline =True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay = 0.0, display =False, log_metrics =True, optimized =True, label_epsilon = i, label_alpha = j)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test = 10, tolerance = 0.05)


if __name__ == '__main__':
    run()
