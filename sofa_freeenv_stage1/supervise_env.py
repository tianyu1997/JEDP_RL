

from typing import Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Sofa
from sofa_scene import createScene
import SofaRuntime
import Sofa.Gui
# from util import getresetTranslated
import random

evaluation = False
# evaluation = True

class SoftRobotEnv(gym.Env):
    def __init__(self, nocontactpts, max_episode_steps=130, channel_last: bool = True, force_threshold=20):
        super(SoftRobotEnv, self).__init__()
        # self.current_target_index = 0 # track target index

        self.nocontactpts=nocontactpts
        self.force_thre = force_threshold
        self.sidelen = int(np.ceil(np.sqrt(nocontactpts)))  # Create square images

        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)

        # original state: 3+3 + 5
        # vec state: 3+3 -> 3 + 4 actual cable value + last axial value + contact binary indicator
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        import SofaRuntime
        self.root = Sofa.Core.Node("root")
        self.root = createScene(self.root)
        Sofa.Simulation.init(self.root)
        # self.addtarget()
        # self.addtip()
        self.root.animate = True
        self.max_episode_steps = max_episode_steps  # Maximum steps per episode
        self.steps = 0
        self.axialaction = np.array(0,dtype=np.float32).flatten()

        self.sucpt_file = 'sucpt.txt'
        self.distance_file = 'distance.txt'

        self.oldpos = []
        self.oldvel = []
        # ############################ for Evaluation , command it when training without GUI#######################
        if evaluation == True:
            # self.target_position = np.array(self.root.ElasticMaterialObject1.point_tar.point_tar.rest_position.value, dtype=np.float32)
            # self.target_position = np.array(self.root.ElasticMaterialObject1.point_tar.point_tar.position.value, dtype=np.float32)
            self.endo = self.root.ElasticMaterialObject2.Tip1.tip1
            self.resetendopos = self.root.ElasticMaterialObject2.dofs.rest_position.value
            # self.stopos = self.root.ElasticMaterialObject1.dofs.position.value
            # self.new_target_position = np.array(self.stopos[self.current_target_index])
        # ############################ for Evaluation , command it when training without GUI#######################



    def initilize_variable(self):
        # self.target_position = np.array(self.root.ElasticMaterialObject1.point_tar.point_tar.rest_position.value, dtype=np.float32)
        # self.target_position = np.array(self.root.ElasticMaterialObject1.point_tar.point_tar.position.value, dtype=np.float32)
        self.endo = self.root.ElasticMaterialObject2.Tip1.tip1
        self.resetendopos = self.root.ElasticMaterialObject2.dofs.rest_position.value
        # self.stopos = self.root.ElasticMaterialObject1.dofs.position.value
        # self.new_target_position = np.array(self.stopos[self.current_target_index])
    
    def normalize_vector(self, vector):
        """Normalize a vector to range 0-1."""
        magnitude = np.linalg.norm(vector)
        if magnitude == 0 or not np.isfinite(magnitude):
            return np.zeros_like(vector)  # Return zero vector if magnitude is zero or invalid
        return vector / magnitude

    def get_state(self) -> str:
        """
        Uses the state to get the observation mapping.

        :return: observation dict {'vec': ..., 'img': ...}
        """
        # Robot tip position
        robot_tip = np.array(self.endo.position.value, dtype=np.float32).flatten()

        vec_state = np.concatenate([
            robot_tip
        ]).astype(np.float32)

        return vec_state
            
    def log_reset_target(self):
        """
        Log the resetting of the target index to the `sucpt_file`.
        """
    #     with open(self.sucpt_file, 'a') as f:
    #         f.write(f"Reset Target Index: {self.current_target_index}\n")
    #     # print(f"Logged reset target index: {self.current_target_index}")

    def log_success(self):
        """
        Log the success event for the current target index and position.
        """


    def log_unsuccess(self):
        """
        Log the success event for the current target index and position.
        """

    def log_distance(self, distance):
        """
        Log the resetting of the target index to the `sucpt_file`.
        """
        with open(self.distance_file, 'a') as f:
            f.write(f"{distance}\n")


    def jacobian(self, action):
        """
        Calculate the Jacobian matrix for the endoscope.
        
        Args:
            action: [∂θ₁, ∂θ₂, ∂z] - joint velocities/increments
        
        Returns:
            J: 3x3 Jacobian matrix relating joint velocities to end-effector velocities
        """
        # Get current and previous positions
        new_vel = np.array(self.endo.velocity.value, dtype=np.float32).flatten()
        delta_vel = new_vel - self.oldvel
        
        # Initialize Jacobian matrix
        J = np.zeros((3, 3))
        
        # Threshold for considering action as zero
        zero_threshold = 1e-8
        
        # Calculate Jacobian using finite differences
        # J[i,j] = ∂pos_i/∂joint_j ≈ Δpos_i/Δjoint_j
        for i in range(3):  # For each position component (x, y, z)
            for j in range(3):  # For each joint (θ₁, θ₂, z)
                if np.abs(action[j]) < zero_threshold:
                    # Zero out the entire column if action is zero
                    J[:, j] = 0
                else:
                    J[i, j] = delta_vel[i] / action[j]
        
        return J

    def step(self, action):
        self.steps += 1  # Increment step counter
        # self.oldpos = np.array(self.endo.position.value, dtype=np.float32).flatten()
        self.oldvel = np.array(self.endo.velocity.value, dtype=np.float32).flatten()
        ################ dynamic modeling ###########################
        # Print the current simulation time
        current_time = self.root.time.value

        self.root.ElasticMaterialObject2.EndostepController.applyAction(action)
        self.root.ElasticMaterialObject2.cablestepcontrol.applyAction(action)
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        Sofa.Simulation.animate(self.root, self.root.dt.value)

        
        robot_tip = np.array(self.endo.position.value, dtype=np.float32).flatten()
        observation = self.get_state()
        reward = 0

        terminated = False
        # Cable lengths --> check if need to end epoch
        cablelength1 = np.array(self.root.ElasticMaterialObject2.cable1.CableConstraint.cableLength.value, dtype=np.float32).flatten()/ 13
        cablelength2 = np.array(self.root.ElasticMaterialObject2.cable2.CableConstraint.cableLength.value, dtype=np.float32).flatten()/ 13
        cablelength3 = np.array(self.root.ElasticMaterialObject2.cable3.CableConstraint.cableLength.value, dtype=np.float32).flatten()/ 13
        cablelength4 = np.array(self.root.ElasticMaterialObject2.cable4.CableConstraint.cableLength.value, dtype=np.float32).flatten()/ 13
        print(f"cable lengths are {cablelength1}, {cablelength2}, {cablelength3}, {cablelength4}")
        if np.any(cablelength1 < 1) or np.any(cablelength2 < 1) or np.any(cablelength3 < 1) or np.any(cablelength4 < 1):
            truncated = True
            print("Cable length exceeded limit, truncating episode.")
        else: 
            truncated = False
        J = self.jacobian(action)
        info = {"jocobian": J} # ground truth of jocobian matrix
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed,options=options)
        self.steps = 0  # Reset step counter
        # ############################ for training , command it when evaluate with GUI#######################
        # problem solving for fix boxing cannot be reset
        # Initialize SOFA and other parameters
        if evaluation == False:
            # problem solving for fix boxing cannot be reset
            # Initialize SOFA and other parameters
            self.root = Sofa.Core.Node("root")
            self.root = createScene(self.root)
            Sofa.Simulation.init(self.root)
            # self.addtarget()
            # self.addtip()
            self.root.animate = True
            self.initilize_variable()
            # Sofa.Simulation.reset(self.root)
            # self.root.init()
        # ############################ for training , command it when evaluate with GUI#######################
        # Reset the target position with seeding for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # self.resettarget(seed)

        self.root.ElasticMaterialObject2.dofs.position.value = self.resetendopos.copy()
        self.root.ElasticMaterialObject2.dofs.velocity.value = [[0, 0, 0]] * len(self.resetendopos)
        
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        Sofa.Simulation.animate(self.root, self.root.dt.value)
        initial_observation = self.get_state()
        # obs = initial_observation.get("vec")
        print(f"initial_obs is {initial_observation}")
        return initial_observation, {}

    def addevaluatecontroller(self, env):
        from controller import EndoEvalController
        self.root.ElasticMaterialObject2.addObject(EndoEvalController(self.root.ElasticMaterialObject2,env,"ppo_endogym_policy_60",self.root))


    ########################### for evaluation #######################
    def addtip(self,indice=int(518)):
        ################################ End effector ########################################
        # Todo1: add a end-effector, index 322, [ 12.3556 , -0.24437, -1.545 ] 136 [10.7638,0.55538,1.08942]; 517 [13.1954,0.3389,-0.8718]; 520[11.921,0.41796,-0.385926]
        endo = self.root.ElasticMaterialObject2
        position = np.array(endo.dofs.position.value[indice])
        tip1=endo.addChild('Tip1')
        tip1.addObject('MechanicalObject',name='tip1', position=position) # position after adjusting
        # tip1.addObject('RigidMapping')
        tip1.addObject('BarycentricMapping', mapForces=False, mapMasses=False)

