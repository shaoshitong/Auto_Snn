
class config:
    def __init__(self,**kwargs):
        self.iter_epoch=[0,130]
        self.iter_beta=[0,0.2]
        self.iter_size=[1,1]
        self.iter_drop=[0.0,0.0]
        for keys in kwargs.keys():
            setattr(self,keys,kwargs[keys])