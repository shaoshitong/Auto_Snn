
class config:
    def __init__(self,**kwargs):
        self.iter_epoch=[0,70]
        self.iter_beta=[0,0.1]
        self.iter_size=[0.5,1]
        self.iter_drop=[0.0,0.05]
        for keys in kwargs.keys():
            setattr(self,keys,kwargs[keys])