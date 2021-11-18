
class config:
    def __init__(self,**kwargs):
        self.iter_epoch=[50,100,150,200]
        self.iter_beta=[0.1,0.3,0.5,0.7]
        self.iter_size=[0.5,0.6,0.7,0.8]
        self.iter_drop=[0.02,0.04,0.06,0.08]
        for keys in kwargs.keys():
            setattr(self,keys,kwargs[keys])
