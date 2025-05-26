# -*- coding: utf-8 -*-
import Sofa.Core
import Sofa.constants.Key as Key


def getdiscreteTranslated(points, vec):
    r = []
    for v in points:
        r.append([v[0] + vec[0], v[1] + vec[1], v[2] + vec[2]])
    
    return r

def getcontinuousTranslated(points, action):
    r = []
    for v in points:
        r.append([v[0] + action[0], v[1] , v[2]])
        # r.append([v[0] + 0.6, v[1] , v[2]])
    # r1=np.array(r[390])
    # print(f"r:{r1}")
    return r

from stable_baselines3 import PPO

class EndoEvalController(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.endo = args[0]
        self.env = args[1]
        self.modelname = args[2]
        self.root = args[3]
        self.name = "EndoController"
        self.model = PPO.load(self.modelname,env=self.env,print_system_info=True)
        self.action = [0,0,0,0,0]
        self.obs,_ = self.env.reset()
        self.step = 0
    def onKeypressedEvent(self, e):
        direction = None

        if e["key"] == Key.uparrow:
            self.action, _ = self.model.predict(self.obs, deterministic=True)
            self.obs, reward, terminated, truncated, info = self.env.step(self.action)
        elif e["key"] == Key.rightarrow:
            self.env.reset()
        # elif e["key"] == Key.downarrow:
        #     self.env.animate(self.model,self.obs,self.step)
        self.step += 1    
        print("step is:", self.step)
        # if direction is not None and self.endo is not None:
        #     # m = finger.getChild("ElasticMaterialObject1")
        #     # mecaobject = self.endo.getObject("dofs")
        #     # mecaobject.findData('position').value = getTranslated(mecaobject.position.value, direction)
        #     mecaobject = self.endo.getObject("dofs")
        #     mecaobject.findData('position').value = getcontinuousTranslated(mecaobject.position.value, action)

        #         cable = m.getChild("PullingCable").getObject("CableConstraint")
        #         p = cable.pullPoint.value
        #         cable.findData("pullPoint").value = [p[0] + direction[0], p[1] + direction[1], p[2] + direction[2]]


class EndokeyboardController(Sofa.Core.Controller):
# discrete version
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.endo = args[0]
        self.name = "EndostepController"
        self.step=0
        self.root = args[1]
        self.resetendopos = self.root.ElasticMaterialObject2.dofs.rest_position.value
    def onKeypressedEvent(self, e):
        """
        Applies an action based on input action number.
        action_number: 0 for up, 1 for down, 2 for left, 3 for right
        """
        direction = None
        self.step += 1
        if e["key"] == Key.leftarrow:
            direction = [0.40, 0.0, 0.0]  # Move up
        elif e["key"] == Key.rightarrow:
            direction = [-0.40, 0.0, 0.0]  # Move down
        elif e["key"] == Key.uparrow:
            # for reset testing
            # from util import getresetTranslated
            # self.root.ElasticMaterialObject2.dofs.rest_position.value = getresetTranslated(self.resetendopos)
            Sofa.Simulation.reset(self.root)
            # Sofa.Simulation.load()
        # elif action_number == 3:
        #     direction = [0.1, 0.0, 0.0]  # Move right
        
        if direction is not None and self.endo is not None:
            mecaobject = self.endo.getObject("dofs")
            mecaobject.findData('rest_position').value = getdiscreteTranslated(mecaobject.rest_position.value, direction)
            # mecaobject.findData('position').value = getdiscreteTranslated(mecaobject.position.value, direction)
        print("step is", self.step)
        robot_tip = np.array(self.endo.Tip1.tip1.position.value, dtype=np.float32).flatten()
        print(f"robot_tip:{robot_tip}")

import numpy as np
class EndostepController(Sofa.Core.Controller):
# continuous version
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.endo = args[0]
        self.name = "EndostepController"

    def applyAction(self, action):
        """
        Applies an action based on input action like [-0.9552726   0.6214838  -0.36833474]
        """
        if action is not None and self.endo is not None:
            mecaobject = self.endo.getObject("dofs")
            # action = [action[0],0,0]
            mecaobject.findData('rest_position').value = getcontinuousTranslated(mecaobject.rest_position.value, action)
            # mecaobject.findData('position').value = getcontinuousTranslated(mecaobject.position.value, action)
            # robot_tip = np.array(self.endo.Tip1.tip1.position.value, dtype=np.float32).flatten()
            # print(f"robot_tip in controller:{robot_tip}")
            # print("applied action")



def createScene(rootNode):
    # Create the EndoController and pass the endoscope object
    endo = rootNode.getChild('endo')  # Assuming 'endo' is the endoscope node name
    rootNode.addObject(EndoController(endo)) 
    # or
    endo.addObject(EndostepController(endo))

    # Example: Manually apply actions instead of waiting for keyboard input
    rootNode.endo.EndostepController.applyAction(1)  # Example: Apply action to move up

    return